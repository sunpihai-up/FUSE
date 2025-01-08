from functools import partial
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from typing import Sequence, Tuple, Union
from model.depth_anything_v2.dpt_align import DPTHead
from model.depth_anything_v2.dinov2 import DINOv2
from model.depth_anything_v2.dinov2_lora import DINOv2_lora
from model.epde.prompt_module import FeatureFusionModule
from dataset.transform import Resize, NormalizeImage, PrepareForNet
from model.epde.utils import token2feature, feature2token, init_weights_vit_timm

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

intermediate_layer_idx = {
    "vits": [2, 5, 8, 11],
    "vitb": [2, 5, 8, 11],
    "vitl": [4, 11, 17, 23],
    "vitg": [9, 19, 29, 39],
}


class READ_OUT(nn.Module):
    def __init__(self, in_channels, use_clstoken=False):
        super(READ_OUT, self).__init__()
        self.use_clstoken = use_clstoken
        if use_clstoken:
            self.readout_project = nn.Sequential(
                nn.Linear(2 * in_channels, in_channels), nn.GELU()
            )

    def forward(self, tokens):
        cls_token = tokens[:, 0]
        feature_token = tokens[:, 1:]
        if self.use_clstoken:
            cls_token = cls_token.unsqueeze(1).expand_as(x)
            x = self.readout_project(torch.cat((feature_token, cls_token), -1))
        else:
            x = feature_token
        return x


class EPDEVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        event_voxel_chans=5,
        embed_dim=768,
        depth=12,
        encoder="vitl",
        dataset="mvsec",
        max_depth=1,
        norm_layer=None,
        prompt_type=None,
        depth_anything_pretrained=None,
        return_feature=False,
    ):
        super(EPDEVisionTransformer, self).__init__()

        self.encoder = encoder
        self.img_size = img_size
        self.patch_size = patch_size
        self.event_voxel_chans = event_voxel_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.max_depth = max_depth
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.prompt_type = prompt_type
        self.depth_anything_pretrained = depth_anything_pretrained
        self.dataset = dataset
        self.blocks_to_take = intermediate_layer_idx[encoder]
        self.depth_anything_config = depth_anything_model_configs[encoder]
        self.return_feature = return_feature

        self.image_encoder = DINOv2(model_name=encoder)
        self.prompt_encoder = DINOv2_lora(model_name=encoder)
        self.num_heads = self.image_encoder.num_heads
        self.depth_head = DPTHead(
            in_channels=embed_dim,
            features=self.depth_anything_config["features"],
            use_bn=False,
            out_channels=self.depth_anything_config["out_channels"],
            use_clstoken=False,
        )

        prompt_fuse = []
        img_read_out, prompt_read_out = [], []
        for i in range(depth):
            if i in self.blocks_to_take:
                prompt_fuse.append(
                    FeatureFusionModule(
                        dim=embed_dim, num_heads=self.num_heads, reduction=1
                    )
                )
                img_read_out.append(READ_OUT(in_channels=embed_dim))
                prompt_read_out.append(READ_OUT(in_channels=embed_dim))

            else:
                prompt_fuse.append(nn.Identity())
                img_read_out.append(nn.Identity())
                prompt_read_out.append(nn.Identity())

            self.prompt_fuse = nn.Sequential(*prompt_fuse)
            self.img_read_out = nn.Sequential(*img_read_out)
            self.prompt_read_out = nn.Sequential(*prompt_read_out)

        self.init_weights()

    def init_weights(self):
        # Load pre-trained foundation model weights
        if self.depth_anything_pretrained is not None:
            pretrained_weights = torch.load(
                self.depth_anything_pretrained, map_location="cpu"
            )
            # Initialize self.image_encoder
            self.image_encoder.load_state_dict(
                {k: v for k, v in pretrained_weights.items() if "pretrained" in k},
                strict=False,
            )

            # Initialize self.prompt_blocks with corresponding layers
            for i, layer in enumerate(self.prompt_blocks):
                if i in self.blocks_to_take:
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
            self.image_encoder.init_weights()
            init_weights_vit_timm(self.patch_embed_prompt)
            init_weights_vit_timm(self.prompt_blocks)
            print(f"Initializing encoder parameters without pre-trained weights")

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        image = x[:, :3, :, :]
        event = x[:, 3:, :, :]
        B, nc, w, h = image.shape
        patch_grid_size = (w // self.patch_size, h // self.patch_size)

        # Compute event and image embedding
        image_token = self.image_encoder.patch_embed(image)
        prompt_token = self.prompt_encoder.patch_embed(event)

        # Adding cls_token
        image_token = torch.cat(
            (
                self.image_encoder.cls_token.expand(image_token.shape[0], -1, -1),
                image_token,
            ),
            dim=1,
        )
        prompt_token = torch.cat(
            (
                self.prompt_encoder.cls_token.expand(prompt_token.shape[0], -1, -1),
                prompt_token,
            ),
            dim=1,
        )

        # Add positional encoding
        image_token += self.image_encoder.interpolate_pos_encoding(image_token, w, h)
        prompt_token += self.prompt_encoder.interpolate_pos_encoding(prompt_token, w, h)

        output, total_block_len = [], len(self.image_encoder.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )

        for i in range(self.depth):
            image_token = self.image_encoder.blocks[i](image_token)
            prompt_token = self.prompt_encoder.blocks[i](prompt_token)

            if i in self.blocks_to_take:
                # Feature Fuse
                img_read_out = self.img_read_out[i](image_token)
                pro_read_out = self.prompt_read_out[i](prompt_token)
                image_feat = token2feature(img_read_out, patch_grid_size)
                prompt_feat = token2feature(pro_read_out, patch_grid_size)
                fuse_token = feature2token(self.prompt_fuse[i](image_feat, prompt_feat))
                cls_token = image_token[:, 0].unsqueeze(1)
                output.append(torch.cat((cls_token, fuse_token), dim=1))

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
            outputs = [self.image_encoder.norm(out) for out in outputs]

        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.get_intermediate_layers(
            x,
            self.blocks_to_take,
            return_class_token=True,
        )

        depth = self.depth_head(features, patch_h, patch_w)
        depth = 1.0 / depth + 1e-3

        if self.return_feature:
            fea_maps = []
            for i, fea in enumerate(features):
                fea_maps.append(fea[0])
            return depth.squeeze(1), fea_maps
        else:
            return depth.squeeze(1)

    # @torch.no_grad()
    # def infer(self, image, event, input_size=518):
    #     image, event, (h, w) = self.input2tensor(image, event, input_size)

    #     input = torch.cat([image, event], dim=0)
    #     input = input.unsqueeze(0)

    #     depth = self.forward(input)
    #     depth = F.interpolate(
    #         depth[:, None], (h, w), mode="bilinear", align_corners=True
    #     )[0, 0]

    #     return depth.cpu().numpy()

    # def input2tensor(self, image, event, input_size=518):
    #     transform = Compose(
    #         [
    #             Resize(
    #                 width=input_size,
    #                 height=input_size,
    #                 resize_target=False,
    #                 keep_aspect_ratio=True,
    #                 ensure_multiple_of=14,
    #                 resize_method="lower_bound",
    #                 image_interpolation_method=cv2.INTER_CUBIC,
    #             ),
    #             NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #             PrepareForNet(),
    #         ]
    #     )

    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    #     h, w = image.shape[:2]
    #     sample = transform({"image": image, "event_voxel": event})
    #     image = torch.from_numpy(sample["image"])
    #     event = torch.from_numpy(sample["event_voxel"])

    #     DEVICE = (
    #         "cuda"
    #         if torch.cuda.is_available()
    #         else "mps" if torch.backends.mps.is_available() else "cpu"
    #     )
    #     image = image.to(DEVICE)
    #     event = event.to(DEVICE)

    #     return image, event, (h, w)


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
    norm_layer=None,
    return_feature=False,
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
        norm_layer=norm_layer,
        prompt_type=prompt_type,
        return_feature=return_feature,
    )
