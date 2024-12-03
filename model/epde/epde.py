from functools import partial

import torch
import torch.nn as nn

from model.depth_anything_v2.dpt import DepthAnythingV2
# from .epde_backbone import EPDEBackbone
from model.layers.patch_embed import PatchEmbed
from model.epde.prompt_module import Prompt_block

class EPDE(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        
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
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.encoder = encoder
        self.features = features
        self.outchannels = out_channels
        self.use_bn = use_bn
        self.use_clstoken = use_clstoken
        
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
        self.num_patches = self.patch_embed_prompt.num_patches
        # Consider whether use the position coding in depth anything v2
        self.pos_embed = self.foundation.pretrained.pos_embed
        
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

    def forward_features(self, x, mask=None):
        rgb = x[:, :3, :, :]
        event = x[:, 3:, :, :]
        
        # Compute event embedding
        event = self.patch_embed_prompt(event)
        B, nc, w, h = event.shape
        event = event + self.foundation.pretrained.interpolate_pos_encoding(event, w, h)
        
        
    def forward():
        pass
