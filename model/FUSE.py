from functools import partial
import torch
import torch.nn as nn

from typing import Sequence, Tuple, Union
from model.depth_anything_v2.dpt_metric import DPTHead
from model.depth_anything_v2.dinov2 import DINOv2
from model.depth_anything_v2.dinov2_lora import DINOv2_lora
from model.fuse.utils import clean_pretrained_weight
from model.fuse.decouple import FreDFuse

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


class FUSEVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        event_voxel_chans=5,
        embed_dim=768,
        depth=12,
        encoder="vitl",
        max_depth=1,
        norm_layer=None,
        depth_anything_pretrained=None,
        prompt_encoder_pretrained=None,
        return_feature=False,
    ):
        super(FUSEVisionTransformer, self).__init__()

        self.encoder = encoder
        self.img_size = img_size
        self.patch_size = patch_size
        self.event_voxel_chans = event_voxel_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.max_depth = max_depth
        print(f"max_depth: {max_depth}")

        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.depth_anything_pretrained = depth_anything_pretrained
        self.prompt_encoder_pretrained = prompt_encoder_pretrained
        self.blocks_to_take = intermediate_layer_idx[encoder]
        self.depth_anything_config = depth_anything_model_configs[encoder]
        self.return_feature = return_feature

        self.image_encoder = DINOv2(model_name=encoder)
        # self.image_encoder = DINOv2_lora(model_name=encoder)
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
                    FreDFuse(
                        embed_dim=embed_dim,
                        num_heads=self.num_heads,
                        reduction=2,
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
            self.init_from_pretrained_weight()
        else:
            self.image_encoder.init_weights()
            self.prompt_encoder.init_weights()
            print("Initializing encoder parameters without pre-trained weights")

    def init_from_pretrained_weight(self):
        self.image_encoder.eval()
        self.prompt_encoder.eval()

        pretrained_weights = torch.load(
            self.depth_anything_pretrained, map_location="cpu"
        )
        prompt_encoder_weights = torch.load(
            self.prompt_encoder_pretrained, map_location="cpu"
        )
        pretrained_weights = clean_pretrained_weight(pretrained_weights)
        prompt_encoder_weights = clean_pretrained_weight(prompt_encoder_weights)

        # Initialize pretrained encoder
        self.image_encoder.load_state_dict(
            {
                k.replace("pretrained.", ""): v
                for k, v in pretrained_weights.items()
                # if "pretrained" in k and "blocks" in k
                if "pretrained" in k
            },
            strict=False,
        )
        has_lora = any("lora" in key for key in pretrained_weights.keys())
        if has_lora:
            print("Loaded pretrained image encoder weights with lora!")
        else:
            print("Loaded pretrained image encoder weights without lora!")

        self.prompt_encoder.load_state_dict(
            {
                k.replace("pretrained.", ""): v
                for k, v in prompt_encoder_weights.items()
                # if "pretrained" in k and "blocks" in k
                if "pretrained" in k
            },
            strict=False,
        )
        has_lora = any("lora" in key for key in prompt_encoder_weights.keys())
        if has_lora:
            print("Loaded pretrained prompt encoder weights with lora!")
        else:
            print("Loaded pretrained prompt encoder weights without lora!")

        # # Initialize pretrained Decoder
        # self.depth_head.load_state_dict(
        #     {
        #         k.replace("depth_head.", ""): v
        #         for k, v in pretrained_weights.items()
        #         if "depth_head" in k
        #     },
        #     strict=True,
        # )
        print(f"Loaded pretrained image weights from {self.depth_anything_pretrained}")
        print(f"Loaded pretrained prompt weights from {self.prompt_encoder_pretrained}")

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        image = x[:, :3, :, :]
        event = x[:, 3:, :, :]
        B, nc, w, h = image.shape
        patch_grid_size = (w // self.patch_size, h // self.patch_size)

        image_token = self.image_encoder.prepare_tokens_with_masks(image)
        prompt_token = self.prompt_encoder.prepare_tokens_with_masks(event)

        output, total_block_len = [], len(self.image_encoder.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )

        for i in range(self.depth):
            image_token = self.image_encoder.blocks[i](image_token)
            prompt_token = self.prompt_encoder.blocks[i](prompt_token)

            if i in self.blocks_to_take:
                # Feature Fusion
                img_read_out = self.img_read_out[i](image_token)
                pro_read_out = self.prompt_read_out[i](prompt_token)
                fuse_token = self.prompt_fuse[i](
                    img_read_out,
                    pro_read_out,
                    patch_grid_size,
                )
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

        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth

        if self.return_feature:
            fea_maps = [fea[0] for fea in features]
            return depth.squeeze(1), fea_maps
        else:
            return depth.squeeze(1)


def fuse_small(patch_size=16, num_register_tokens=0, **kwargs):
    return FUSEVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        encoder="vits",
        **kwargs,
    )


def fuse_base(patch_size=16, num_register_tokens=0, **kwargs):
    return FUSEVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        encoder="vitb",
        **kwargs,
    )


def fuse_large(patch_size=16, num_register_tokens=0, **kwargs):
    return FUSEVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        encoder="vitl",
        **kwargs,
    )


def fuse_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    return FUSEVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        encoder="vitg",
        **kwargs,
    )


def FUSE(
    model_name,
    max_depth=1,
    depth_anything_pretrained=None,
    prompt_encoder_pretrained=None,
    event_voxel_chans=5,
    norm_layer=None,
    return_feature=False,
):
    model_zoo = {
        "vits": fuse_small,
        "vitb": fuse_base,
        "vitl": fuse_large,
        "vitg": fuse_giant2,
    }

    return model_zoo[model_name](
        img_size=518,
        patch_size=14,
        event_voxel_chans=event_voxel_chans,
        max_depth=max_depth,
        depth_anything_pretrained=depth_anything_pretrained,
        prompt_encoder_pretrained=prompt_encoder_pretrained,
        norm_layer=norm_layer,
        return_feature=return_feature,
    )
