import torch.nn as nn
from torch.nn.init import trunc_normal_

def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def token2feature(tokens, patch_grid_size):
    """add token transfer to feature"""
    B, L, D = tokens.shape
    # H = W = int(L**0.5)
    H, W = patch_grid_size[0], patch_grid_size[1]
    x = tokens.permute(0, 2, 1).view(B, D, H, W).contiguous()
    return x


def feature2token(x):
    B, C, H, W = x.shape
    L = W * H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens
