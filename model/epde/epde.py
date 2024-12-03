import torch
import torch.nn as nn

from model.depth_anything_v2.dpt import DepthAnythingV2
# from .epde_backbone import EPDEBackbone
from model.layers.patch_embed import PatchEmbed

class EPDE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EPDE, self).__init__(*args, **kwargs)

        self.args = args
        
        self.foundation = DepthAnythingV2(args)
        self.foundation.load_state_dict(torch.load(args.pretrained_depthanything, map_location='cpu')['net'])
        
        self.patch_embed_prompt = args.embed_layer(
            img_size=args.img_size, patch_size=args.patch_size, inchans=args.in_chans, embed_dim=args.embed_dim
        )
        # self.backbone = EPDEBackbone()
    
    def forward():
        pass
    