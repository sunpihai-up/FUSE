import torch.nn as nn
from torch.nn.init import trunc_normal_

def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def token2feature(tokens, patch_grid_size):
    assert tokens.dim() == 3, "The shape of the token should be [B, N, C]"
    B, N, C = tokens.shape

    if isinstance(patch_grid_size, int):
        patch_grid_size = (patch_grid_size, patch_grid_size)
    H, W = patch_grid_size
    assert H * W == N, f"Number of patches {N} does not match grid size {H}x{W}!"
    # [B, N, C] --> [B, C, N] --> [B, C, H, W]
    return tokens.permute(0, 2, 1).view(B, C, H, W).contiguous()


def feature2token(feature):
    B, C, H, W = feature.shape
    L = H * W
    # [B, C, H, W] --> [B, C, L] --> [B, L, C]
    return feature.view(B, C, L).permute(0, 2, 1).contiguous()
