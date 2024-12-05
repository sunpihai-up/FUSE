from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Tuple, Union

from model.depth_anything_v2.dpt import DepthAnythingV2
from model.layers.patch_embed import PatchEmbed
from model.epde.prompt_module import Prompt_block

'''
add token transfer to feature
'''
def token2feature(tokens):
    B,L,D=tokens.shape
    H=W=int(L**0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x

'''
feature2token
'''
def feature2token(x):
    B,C,W,H = x.shape
    L = W*H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens


class EPDE(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        embed_layer=PatchEmbed,
        encoder="vitl",
        norm_layer=None,
        prompt_type=None,
    ):
        super(EPDE, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.embed_layer = embed_layer

        self.encoder = encoder

        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.prompt_type = prompt_type

        depth_anything_model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.foundation = DepthAnythingV2(**depth_anything_model_configs[self.encoder])

        # self.foundation.load_state_dict(torch.load(args.pretrained_depthanything, map_location='cpu')['net'])

        '''
        prompt
        ! inchans: Number of input channels
        '''
        self.patch_embed_prompt = embed_layer(
            img_size=img_size, patch_size=patch_size, inchans=in_chans, embed_dim=embed_dim
        )

        if self.prompt_type in ['epde_shaw', 'epde_deep']:
            prompt_blocks = []
            block_nums = depth if self.prompt_type == 'epde_deep' else 1
            for i in range(block_nums):
                prompt_blocks.append(Prompt_block(inplanes=embed_dim, hide_channel=8, smooth=True))
            self.prompt_blocks = nn.Sequential(*prompt_blocks)

            prompt_norms = []
            for i in range(block_nums):
                prompt_norms.append(self.norm_layer(embed_dim))
            self.prompt_norms = nn.Sequential(*prompt_norms)

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        image = x[:, :3, :, :]
        event = x[:, 3:, :, :]
        
        # Compute event and image embedding
        image = self.foundation.pretrained.patch_embed(image)
        event_prompted = self.patch_embed_prompt(event)
        
        # Injecting modal supplementary information
        if self.prompt_type in ['epde_shaw', 'epde_deep']:
            image_feat = token2feature(self.prompt_norms[0](image))
            event_prompted_feat = token2feature(self.prompt_norms[0](event_prompted))
            
            prompted_feat = self.prompt_blocks[0](torch.cat([image_feat, event_prompted_feat], dim=1))
            event_prompted = feature2token(prompted_feat)
            
            image = image + event_prompted
        else:
            image = image + event_prompted
        
        # Adding cls_token
        image = torch.cat(
            (self.foundation.pretrained.cls_token.expand(image.shape[0], -1, -1), x), dim=1
        )
        
        # Add positional encoding
        B, nc, w, h = image.shape
        image += self.foundation.pretrained.interpolate_pos_encoding(image, w, h)
        
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        
        for i, blk in enumerate(self.foundation.pretrained.blocks):
            if i >= 1 and self.prompt_type == 'epde_deep':
                # Add Prompt information from 1st layer                
                image_feat = token2feature(self.prompt_norms[i](image))
                event_prompted_feat = token2feature(self.prompt_norms[i](event_prompted))
                
                prompted_feat = self.prompt_blocks[i](torch.cat([image_feat, event_prompted_feat], dim=1))
                event_prompted = feature2token(prompted_feat)
                
                image = image + event_prompted
            
            image = blk(image)
            if i in blocks_to_take:
                output.append(image)
        
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)

        if norm:
            outputs = [self.foundation.pretrained.norm(out) for out in outputs]
            
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder], return_class_token=True
        )

        depth = self.foundation.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)

        depth = self.forward(image)

        depth = F.interpolate(
            depth[:, None], (h, w), mode="bilinear", align_corners=True
        )[0, 0]

        return depth.cpu().numpy()

def epde_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = EPDE(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        encoder='vits',
        **kwargs,
    )
    return model


def epde_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = EPDE(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        encoder='vitb',
        **kwargs,
    )
    return model


def epde_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = EPDE(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        encoder='vitl',
        **kwargs,
    )
    return model


def epde_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = EPDE(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        encoder='vitg',
        **kwargs,
    )
    return model


def EPDE(
    model_name, in_chans=3, embed_layer=PatchEmbed, norm_layer=None, prompt_type=None
):
    model_zoo = {
        "vits": epde_small,
        "vitb": epde_base,
        "vitl": epde_large,
        "vitg": epde_giant2,
    }

    return model_zoo[model_name](
        img_size=518,
        patch_size=14,
        in_chans=3,
        embed_layer=embed_layer,
        norm_layer=norm_layer,
        prompt_type=prompt_type,
    )