import torch
import torch.nn.functional as F
import torch.nn as nn

from .utils import token2feature, feature2token


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, reduction=1, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads // reduction

        # Linear Projection Layers
        self.W_q = nn.Linear(embed_dim, embed_dim // reduction)  # x1 -> Query
        self.W_k = nn.Linear(embed_dim, embed_dim // reduction)  # x2 -> Key
        self.W_v = nn.Linear(embed_dim, embed_dim // reduction)  # x2 -> Value

        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.out_proj = nn.Linear(embed_dim // reduction, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x1, x2):
        B, N, _ = x1.shape
        """
        x1/x2: [B, N, C]
        masks: [B, N]
        """
        x1 = self.norm1(x1)
        x2 = self.norm1(x2)

        q = self.W_q(x1)
        k = self.W_k(x2)
        v = self.W_v(x2)

        # [B, H, N, D]
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Step 3: Calculate attention score (incorporating relative credibility)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Calculating attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Step 4: Aggregate Value
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # [B, H, N, D]

        # Merge multiple heads and project
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        attn_output = self.out_proj(attn_output)  # [B, N, C]
        return x1 + attn_output


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        # Shuffle the channels: split channels into groups and then concatenate them
        x = x.view(B, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(B, C, H, W)


class GaussianLaplacianDecoupling(nn.Module):
    def __init__(
        self, channels, num_layers=3, scale=2, kernel_size=3, sigma=1.0, groups=8
    ):
        """
        Args:
          channels: number of channels in the feature map.
          num_layers: number of pyramid layers.
          scale: downsampling factor (e.g., 2).
          kernel_size: size of the Gaussian kernel.
          sigma: standard deviation for Gaussian blur.
        """
        super().__init__()
        self.num_layers = num_layers
        self.scale = scale
        self.channels = channels
        self.groups = groups

        # Create a fixed 2D Gaussian kernel.
        self.register_buffer(
            "gaussian_kernel", self.create_gaussian_kernel(kernel_size, sigma)
        )

        # Fusion modules for top-down fusion.
        # One module per level except the coarsest level.
        self.fusion_convs_g = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels * 2,
                        channels,
                        kernel_size=1,
                        groups=self.groups,
                        # bias=False,
                    ),
                    ChannelShuffle(self.groups),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=1,
                        groups=self.groups,
                        # bias=False,
                    ),
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.fusion_convs_l = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels * 2,
                        channels,
                        kernel_size=1,
                        groups=self.groups,
                        # bias=False,
                    ),
                    ChannelShuffle(self.groups),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=1,
                        groups=self.groups,
                        # bias=False,
                    ),
                )
                for _ in range(num_layers - 1)
            ]
        )

    def create_gaussian_kernel(self, kernel_size, sigma):
        # Create a 2D Gaussian kernel.
        ax = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return kernel  # shape: [1, 1, kernel_size, kernel_size]

    def gaussian_blur(self, x):
        # Apply the same Gaussian kernel to each channel (depthwise convolution).
        B, C, H, W = x.shape
        kernel = self.gaussian_kernel.expand(C, 1, -1, -1)  # shape: [C, 1, k, k]
        x = F.conv2d(x, kernel, padding=self.gaussian_kernel.shape[-1] // 2, groups=C)
        return x

    def downsample(self, x):
        # Use bilinear interpolation to downsample.
        B, C, H, W = x.shape
        return F.interpolate(
            x,
            size=(H // self.scale, W // self.scale),
            mode="bilinear",
            align_corners=False,
        )

    def upsample(self, x, size):
        # Upsample to a given spatial size.
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def build_gaussian_pyramid(self, x):
        gp = [x]
        for _ in range(1, self.num_layers):
            blurred = self.gaussian_blur(gp[-1])
            down = self.downsample(blurred)
            gp.append(down)
        return gp

    def build_laplacian_pyramid(self, gp):
        lp = []
        for i in range(len(gp) - 1):
            up = self.upsample(gp[i + 1], size=gp[i].shape[-2:])
            lap = gp[i] - up
            lp.append(lap)
        # The last level is retained as is.
        lp.append(gp[-1])
        return lp

    def top_down_fusion(self, pyramid, fusion_convs):
        # Fuse pyramid levels from coarsest to finest.
        fused = pyramid[-1]
        for i in range(self.num_layers - 2, -1, -1):
            up = self.upsample(fused, size=pyramid[i].shape[-2:])
            # Concatenate along channel dimension.
            concat = torch.cat([pyramid[i], up], dim=1)
            fused = fusion_convs[i](concat)
        return fused

    def forward(self, x, patch_grid_size):
        x = token2feature(x, patch_grid_size=patch_grid_size)
        # Build the Gaussian pyramid.
        gp = self.build_gaussian_pyramid(x)
        # Build the Laplacian pyramid.
        lp = self.build_laplacian_pyramid(gp)

        # Top-down fusion for low-frequency (Gaussian) part.
        low_freq = self.top_down_fusion(gp, self.fusion_convs_g)
        # Top-down fusion for high-frequency (Laplacian) part.
        high_freq = self.top_down_fusion(lp, self.fusion_convs_l)
        return feature2token(low_freq), feature2token(high_freq)


class FreDFuse(nn.Module):
    """
    Fequency Decoupled Feature Integration Module
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        reduction=1,
    ):
        super().__init__()
        self.img_feature_decouple = GaussianLaplacianDecoupling(channels=embed_dim)
        self.vox_feature_decouple = GaussianLaplacianDecoupling(channels=embed_dim)

        self.low_cross_att = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            reduction=reduction,
        )
        self.high_cross_att = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            reduction=reduction,
        )

        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x1,
        x2,
        patch_grid_size,
    ):
        """
        Shape of x1/x2: [B, N, C]
        Shape of cofs:  [B, N]
        """

        x1_low, x1_high = self.img_feature_decouple(x1, patch_grid_size)
        x2_low, x2_high = self.vox_feature_decouple(x2, patch_grid_size)

        low_fused = self.low_cross_att(x1_low, x2_low)
        high_fused = self.high_cross_att(x2_high, x1_high)

        fused = low_fused + high_fused
        out_proj = self.out_proj(self.norm1(fused))
        return self.norm2(out_proj)
