from functools import partial
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from typing import Sequence, Tuple, Union
from model.depth_anything_v2.dpt import DepthAnythingV2

# from model.layers.patch_embed import PatchEmbed
from model.depth_anything_v2.dinov2_layers import (
    PatchEmbed,
    Mlp,
    NestedTensorBlock as Block,
)
from model.epde.prompt_module import (
    FeatureRectifyModule,
    FeatureFusionModule,
    MaxVar_Feat_Fuse,
    MaxVar_Feat_Rect,
    Prompt_block,
    Prompt_block_rf,
)
from dataset.transform import Resize, NormalizeImage, PrepareForNet
from .utils import token2feature, feature2token, init_weights_vit_timm


class EPDEVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        event_voxel_chans=5,
        embed_dim=768,
        depth=12,
        embed_layer=PatchEmbed,
        encoder="vitl",
        dataset="mvsec",
        max_depth=1,
        norm_layer=None,
        prompt_type=None,
        depth_anything_pretrained=None,
    ):
        super(EPDEVisionTransformer, self).__init__()

        self.encoder = encoder
        self.img_size = img_size
        self.patch_size = patch_size
        self.event_voxel_chans = event_voxel_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.prompt_depth = self.depth // 2
        self.max_depth = max_depth
        self.embed_layer = embed_layer
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.prompt_type = prompt_type
        self.depth_anything_pretrained = depth_anything_pretrained
        self.dataset = dataset

        depth_anything_model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }

        self.foundation = DepthAnythingV2(**depth_anything_model_configs[self.encoder])
        self.blocks_to_take = self.foundation.intermediate_layer_idx[encoder]
        self.num_heads = self.foundation.pretrained.num_heads

        self.patch_embed_prompt = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=event_voxel_chans,
            embed_dim=embed_dim,
        )

        prompt_blocks = []
        # prompt_rectify = []
        prompt_fuse = []
        for i in range(self.prompt_depth):
            prompt_blocks.append(
                Block(
                    dim=embed_dim,
                    num_heads=self.num_heads,
                    norm_layer=self.norm_layer,
                    qkv_bias=True,
                )
            )
            # prompt_rectify.append(MaxVar_Feat_Rect())

            if i in self.blocks_to_take:
                prompt_fuse.append(MaxVar_Feat_Fuse())
            else:
                prompt_fuse.append(nn.Identity())
        self.prompt_blocks = nn.Sequential(*prompt_blocks)
        # self.prompt_rectify = nn.Sequential(*prompt_rectify)
        self.prompt_fuse = nn.Sequential(*prompt_fuse)

        self.init_weights()

    def init_weights(self):
        if self.depth_anything_pretrained is not None:
            pretrained_weights = torch.load(
                self.depth_anything_pretrained, map_location="cpu"
            )
            # Initialize self.foundation
            self.foundation.load_state_dict(
                {k: v for k, v in pretrained_weights.items() if "pretrained" in k},
                strict=False,
            )
            # self.foundation.load_state_dict(pretrained_weights, strict=False)

            # Initialize self.prompt_blocks with corresponding layers
            for i, layer in enumerate(self.prompt_blocks):
                pretrained_layer_prefix = f"pretrained.blocks.{i}."
                layer_state_dict = {
                    k: v
                    for k, v in pretrained_weights.items()
                    if k.startswith(pretrained_layer_prefix)
                }
                layer.load_state_dict(layer_state_dict, strict=False)

            # Initialize self.prompt_patch using pretrained.patch_embed
            patch_embed_state_dict = {
                k: v for k, v in pretrained_weights.items() if "patch_embed" in k
            }
            self.patch_embed_prompt.load_state_dict(
                patch_embed_state_dict, strict=False
            )
            print(f"Loaded pretrained weights from {self.depth_anything_pretrained}")
        else:
            self.foundation.pretrained.init_weights()
            init_weights_vit_timm(self.patch_embed_prompt)
            init_weights_vit_timm(self.prompt_blocks)
            print(f"Initializing encoder parameters without pre-trained weights")

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        image = x[:, :3, :, :]
        event = x[:, 3:, :, :]
        B, nc, w, h = image.shape
        patch_grid_size = (w // self.patch_size, h // self.patch_size)

        # Compute event and image embedding
        image_token = self.foundation.pretrained.patch_embed(image)
        prompt_token = self.patch_embed_prompt(event)

        # Adding cls_token
        image_token = torch.cat(
            (
                self.foundation.pretrained.cls_token.expand(
                    image_token.shape[0], -1, -1
                ),
                image_token,
            ),
            dim=1,
        )
        prompt_token = torch.cat(
            (
                self.foundation.pretrained.cls_token.expand(
                    prompt_token.shape[0], -1, -1
                ),
                prompt_token,
            ),
            dim=1,
        )

        # Add positional encoding
        image_token += self.foundation.pretrained.interpolate_pos_encoding(
            image_token, w, h
        )
        prompt_token += self.foundation.pretrained.interpolate_pos_encoding(
            prompt_token, w, h
        )
        # prompt_token = self.prompt_blocks[0](prompt_token)

        output, total_block_len = [], len(self.foundation.pretrained.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )

        for i, blk in enumerate(self.foundation.pretrained.blocks):
            if i < self.prompt_depth:
                # use [:, 1:] to exclude the cls_token
                # image_feat = token2feature(image_token[:, 1:], patch_grid_size)
                # prompt_feat = token2feature(prompt_token[:, 1:], patch_grid_size)

                # image_feat, prompt_feat = self.prompt_rectify[i](
                #     image_feat, prompt_feat
                # )
                # prompt_token = prompt_token.clone()
                # image_token = image_token.clone()
                # prompt_token[:, 1:] = feature2token(prompt_feat)
                # image_token[:, 1:] = feature2token(image_feat)

                # prompt block
                prompt_token = self.prompt_blocks[i](prompt_token)

            image_token = blk(image_token)
            if i < self.prompt_depth and i in blocks_to_take:
                # Feature Fuse
                image_feat = token2feature(image_token[:, 1:], patch_grid_size)
                prompt_feat = token2feature(prompt_token[:, 1:], patch_grid_size)
                fuse_token = feature2token(self.prompt_fuse[i](image_feat, prompt_feat))
                cls_token = image_token[:, 0].unsqueeze(1)
                output.append(torch.cat((cls_token, fuse_token), dim=1))

            if i >= self.prompt_depth and i in blocks_to_take:
                output.append(image_token)

        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
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
            x,
            self.foundation.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )

        depth = self.foundation.depth_head(features, patch_h, patch_w) * self.max_depth

        return depth.squeeze(1)

    @torch.no_grad()
    def infer(self, image, event, input_size=518):
        image, event, (h, w) = self.input2tensor(image, event, input_size)

        input = torch.cat([image, event], dim=0)
        input = input.unsqueeze(0)

        depth = self.forward(input)
        depth = F.interpolate(
            depth[:, None], (h, w), mode="bilinear", align_corners=True
        )[0, 0]

        return depth.cpu().numpy()

    def input2tensor(self, image, event, input_size=518):
        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = image.shape[:2]
        sample = transform({"image": image, "event_voxel": event})
        image = torch.from_numpy(sample["image"])
        event = torch.from_numpy(sample["event_voxel"])

        DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        image = image.to(DEVICE)
        event = event.to(DEVICE)

        return image, event, (h, w)


def epde_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = EPDEVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        encoder="vits",
        **kwargs,
    )
    return model


def epde_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = EPDEVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        encoder="vitb",
        **kwargs,
    )
    return model


def epde_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = EPDEVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        encoder="vitl",
        **kwargs,
    )
    return model


def epde_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = EPDEVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        encoder="vitg",
        **kwargs,
    )
    return model


def EPDE(
    model_name,
    prompt_type,
    dataset="dense",
    max_depth=1,
    depth_anything_pretrained=None,
    event_voxel_chans=5,
    embed_layer=PatchEmbed,
    norm_layer=None,
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
        event_voxel_chans=event_voxel_chans,
        dataset=dataset,
        max_depth=max_depth,
        depth_anything_pretrained=depth_anything_pretrained,
        embed_layer=embed_layer,
        norm_layer=norm_layer,
        prompt_type=prompt_type,
    )
