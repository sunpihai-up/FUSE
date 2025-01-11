import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_
import math
from .utils import token2feature, feature2token


class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        """
        x: [batch_size, features, k]
        """
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h * w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output


class Prompt_block(nn.Module):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(
            in_channels=inplanes,
            out_channels=hide_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv0_1 = nn.Conv2d(
            in_channels=inplanes,
            out_channels=hide_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1x1 = nn.Conv2d(
            in_channels=hide_channel,
            out_channels=inplanes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.fovea = Fovea(smooth=smooth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass with input x.
        x0: Feature map from encoder layer
        x1: Feature map from prompt layer
        """
        B, C, W, H = x.shape
        x0 = x[:, 0 : int(C / 2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C / 2) :, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) + x1

        return self.conv1x1(x0)


""" **************** Start of Feature Rectify Module **************** """


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(
            1, 0, 2, 3, 4
        )  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = (
            self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        )  # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = (
            x1
            + self.lambda_c * channel_weights[1] * x2
            + self.lambda_s * spatial_weights[1] * x2
        )
        out_x2 = (
            x2
            + self.lambda_c * channel_weights[0] * x1
            + self.lambda_s * spatial_weights[0] * x1
        )
        return out_x1, out_x2


""" **************** End of Feature Rectify Module **************** """

""" **************** Start of Feature Fusion Module **************** """


# Stage 1
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = (
            x1.reshape(B, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        q2 = (
            x2.reshape(B, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k1, v1 = (
            self.kv1(x1)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        k2, v2 = (
            self.kv2(x2)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


# Stage 2
class ChannelEmbed(nn.Module):
    def __init__(
        self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d
    ):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(
                out_channels // reduction,
                out_channels // reduction,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=out_channels // reduction,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels // reduction, out_channels, kernel_size=1, bias=True
            ),
            norm_layer(out_channels),
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2,
            out_channels=dim,
            reduction=reduction,
            norm_layer=norm_layer,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)

        return merge


""" **************** End of Feature Fusion Module **************** """


class Prompt_block_rf(nn.Module):
    def __init__(self, inplanes=None, reduction=1, num_heads=12):
        super(Prompt_block_rf, self).__init__()
        self.frm = FeatureRectifyModule(dim=inplanes, reduction=reduction)
        self.ffm = FeatureFusionModule(
            dim=inplanes, reduction=reduction, num_heads=num_heads
        )

    def forward(self, x):
        """
        Forward pass with input x.
        x0: Feature map from encoder layer
        x1: Feature map from prompt layer
        """
        B, C, W, H = x.shape
        x0 = x[:, 0 : int(C / 2), :, :].contiguous()
        x1 = x[:, int(C / 2) :, :, :].contiguous()

        x0, x1 = self.frm(x0, x1)
        x_fused = self.ffm(x0, x1)

        return x_fused


class MaxVar_Feat_Rect(nn.Module):
    def __init__(self):
        super(MaxVar_Feat_Rect, self).__init__()

    def forward(self, feat_a, feat_b):
        A = feat_a
        B = feat_b
        mean_A = torch.mean(A, dim=1, keepdim=True)
        mean_B = torch.mean(B, dim=1, keepdim=True)
        A_demeaned = A - mean_A
        B_demeaned = B - mean_B
        covariance = torch.sum(A_demeaned * B_demeaned, dim=1)
        std_A = torch.sqrt(torch.sum(A_demeaned**2, dim=1))
        std_B = torch.sqrt(torch.sum(B_demeaned**2, dim=1))
        correlation = covariance / (std_A * std_B)
        A_flat = A.view(A.size(0), A.size(1), -1)
        B_flat = B.view(B.size(0), B.size(1), -1)

        cosine_sim = (F.cosine_similarity(A_flat, B_flat, dim=1) + 1) / 2

        cosine_sim = cosine_sim.view(A.size(0), A.size(2), A.size(3))

        std_A = torch.std(A, dim=1, keepdim=True)
        std_B = torch.std(B, dim=1, keepdim=True)

        var_A = torch.var(A, dim=1, keepdim=True)
        var_B = torch.var(B, dim=1, keepdim=True)

        var_A = var_A / torch.sum(var_A)
        var_B = var_B / torch.sum(var_B)

        high_sim_threshold = 0.7
        average = (A + B) / 2

        fuse_based_on_variance1 = torch.where(
            var_A >= var_B, A, torch.div(B * std_A, std_B)
        )
        fuse_based_on_variance2 = torch.where(
            var_A >= var_B, torch.div(A * std_B, std_A), B
        )

        # Decide which values to take based on the cosine similarity
        fused_tensor1 = torch.where(
            correlation.unsqueeze(1) > high_sim_threshold,
            average,
            fuse_based_on_variance1,
        )
        fused_tensor2 = torch.where(
            correlation.unsqueeze(1) > high_sim_threshold,
            average,
            fuse_based_on_variance2,
        )

        rect_feat_a = fused_tensor1
        rect_feat_b = fused_tensor2

        return rect_feat_a, rect_feat_b



class MaxVar_Feat_Fuse(nn.Module):
    def __init__(self):
        super(MaxVar_Feat_Fuse, self).__init__()
    
    def forward(self, feat_a, feat_b):
        A = feat_a
        B = feat_b
        mean_A = torch.mean(A, dim=1, keepdim=True)
        mean_B = torch.mean(B, dim=1, keepdim=True)
        A_demeaned = A - mean_A
        B_demeaned = B - mean_B
        covariance = torch.sum(A_demeaned * B_demeaned, dim=1)
        std_A = torch.sqrt(torch.sum(A_demeaned**2, dim=1))
        std_B = torch.sqrt(torch.sum(B_demeaned**2, dim=1))
        correlation = covariance / (std_A * std_B)
        A_flat = A.view(A.size(0), A.size(1), -1)
        B_flat = B.view(B.size(0), B.size(1), -1)

        cosine_sim = (F.cosine_similarity(A_flat, B_flat, dim=1) + 1) / 2
        cosine_sim = cosine_sim.view(A.size(0), A.size(2), A.size(3))

        std_A = torch.std(A, dim=1, keepdim=True)
        std_B = torch.std(B, dim=1, keepdim=True)

        var_A = torch.var(A, dim=1, keepdim=True)
        var_B = torch.var(B, dim=1, keepdim=True)
        var_A = var_A / torch.sum(var_A)
        var_B = var_B / torch.sum(var_B)

        high_sim_threshold = 0.7
        average = (A + B) / 2

        fuse_based_on_variance = torch.where(var_A >= var_B, A, B)
        fused_tensor = torch.where(
            correlation.unsqueeze(1) > high_sim_threshold, average, fuse_based_on_variance
        )

        return fused_tensor


class FeatureFusionWeight(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        weight_layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=2,
            )
        ]
        weight_layers.append(nn.GELU())
        weight_layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=2,
                kernel_size=kernel_size,
            )
        )
        self.get_weight = nn.Sequential(*weight_layers)
    
    def forward(self, joint_feas):
        # The shape of x1 and x2 is [B, C, H, W]
        weight = self.get_weight(joint_feas)
        weight = F.softmax(weight, dim=-1)
        return weight
    

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = self.head_dim**-0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, L, C = query.shape

        # (B, num_heads, L, head_dim)
        Q = self.query(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.query(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (B, num_heads, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        attn_weight = F.softmax(scores, dim=-1)
        attn_weight = self.dropout(attn_weight)

        # (B, num_heads, L, head_dim)
        context = torch.matmul(attn_weight, V)
        # (B, L, dim)
        context = context.transpose(1, 2).contiguous().view(B, L, self.dim)

        return self.out(context)


class FeatureInteraction(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super(FeatureInteraction, self).__init__()
        self.cross_attention_1 = CrossAttentionBlock(dim, num_heads, dropout)
        self.cross_attention_2 = CrossAttentionBlock(dim, num_heads, dropout)
    
    def forward(self, x1, x2):
        inter_x1 = self.cross_attention_1(query=x1, key=x2, value=x2)
        inter_x2 = self.cross_attention_2(query=x2, key=x1, value=x1)
        
        return inter_x1, inter_x2


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.upsample = nn.PixelShuffle(upscale_factor=2)
        self.interact = FeatureInteraction(dim=dim//4, num_heads=num_heads)
        self.downsample_1 = nn.Conv2d(
            in_channels=dim // 4, out_channels=dim, kernel_size=1, stride=2
        )
        self.downsample_2 = nn.Conv2d(
            in_channels=dim // 4, out_channels=dim, kernel_size=1, stride=2
        )
        self.fuse_weight = FeatureFusionWeight(
            in_channels=dim // 2, out_channels=dim, kernel_size=1
        )
    
    def fuse_feature_maps(self, weight_map, feature_map1, feature_map2):
        # Split the weight map into two components, [B, 1, H, W]
        weight1 = weight_map[:, 0:1, :, :]
        weight2 = weight_map[:, 1:2, :, :]
        
        # Apply weights to the feature maps, [B, C, H, W] * [B, 1, H, W]
        weighted_feature1 = feature_map1 * weight1
        weighted_feature2 = feature_map2 * weight2

        # Compute the fused feature map, [B, C, H, W]        
        return weighted_feature1 + weighted_feature2

    def forward(self, x1, x2, patch_grid_size):
        assert x1.shape == x2.shape, "The shape of x1 and x2 does not match."
        # [B, L, C] --> [B, C, H, W]
        x1 = token2feature(x1, patch_grid_size)
        x2 = token2feature(x2, patch_grid_size)

        # [B, C, H, W] --> [B, C/4, H*2, W*2]
        H, W = patch_grid_size
        upsample_patch_grid_size = (H * 2, W * 2)
        x1 = self.upsample(x1)
        x2 = self.upsample(x2)
        
        # [B, C/4, H*2, W*2] --> [B, L*4, C/4]
        x1 = feature2token(x1)
        x2 = feature2token(x2)
        x1, x2 = self.interact(x1, x2)
        
        # [B, L*4, C/4] --> [B, C/4, H*2, W*2]
        x1 = token2feature(x1, upsample_patch_grid_size)
        x2 = token2feature(x2, upsample_patch_grid_size)
        
        # [B, C/4, H*2, W*2] --> [B, C, H, W]
        x1_down = self.downsample_1(x1)
        x2_down = self.downsample_2(x2)
        
        # [B, C/4, H*2, W*2] * 2 --> [B, C/2, H*2, W*2]
        joint_feas = torch.cat((x1, x2), dim=1)
        weight = self.fuse_weight(joint_feas)
        fused_feature_map = self.fuse_feature_maps(weight, x1_down, x2_down)
        return feature2token(fused_feature_map)