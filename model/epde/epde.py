from functools import partial
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from typing import Sequence, Tuple, Union
from model.depth_anything_v2.dpt import DepthAnythingV2
from model.layers.patch_embed import PatchEmbed
from model.epde.prompt_module import Prompt_block
from dataset.transform import Resize, NormalizeImage, PrepareForNet
from .utils import token2feature, feature2token, init_weights_vit_timm

class EPDEVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        # in_chans=3,
        event_voxel_chans=5,
        embed_dim=768,
        depth=12,
        embed_layer=PatchEmbed,
        encoder="vitl",
        dataset='dense',
        norm_layer=None,
        prompt_type=None,
        depth_anything_pretrained=None,
    ):
        super(EPDEVisionTransformer, self).__init__()

        self.encoder = encoder
        self.img_size = img_size
        self.patch_size = patch_size
        # self.in_chans = in_chans
        self.event_voxel_chans = event_voxel_chans
        self.embed_dim = embed_dim
        self.depth = depth
        
        self.embed_layer = embed_layer
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.prompt_type = prompt_type
        self.depth_anything_pretrained = depth_anything_pretrained
        self.dataset = dataset
        
        depth_anything_model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.foundation = DepthAnythingV2(**depth_anything_model_configs[self.encoder])

        self.patch_embed_prompt = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=event_voxel_chans, embed_dim=embed_dim
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
        
        self.init_weights()

    def init_weights(self):
        # Load pre-trained foundation model weights
        if self.depth_anything_pretrained is not None:
            try:
                self.foundation.load_state_dict(
                    {
                        k: v
                        for k, v in torch.load(
                            self.depth_anything_pretrained, map_location="cpu"
                        ).items()
                        if "pretrained" in k
                    },
                    strict=False,
                )
                print(f"Loaded pretrained weights from {self.depth_anything_pretrained}")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
        else:
            self.foundation.pretrained.init_weights()
            print(f"Initializing encoder parameters without pre-trained weights")
        
        init_weights_vit_timm(self.patch_embed_prompt)

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        image = x[:, :3, :, :]
        event = x[:, 3:, :, :]
        B, nc, w, h = image.shape
        
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
            (self.foundation.pretrained.cls_token.expand(image.shape[0], -1, -1), image), dim=1
        )
        event_prompted = torch.cat(
            (self.foundation.pretrained.cls_token.expand(event.shape[0], -1, -1), event_prompted), dim=1
        )

        # Add positional encoding
        
        image += self.foundation.pretrained.interpolate_pos_encoding(image, w, h)
        event_prompted += self.foundation.pretrained.interpolate_pos_encoding(event_prompted, w, h)

        output, total_block_len = [], len(self.foundation.pretrained.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )

        for i, blk in enumerate(self.foundation.pretrained.blocks):
            if i >= 1 and self.prompt_type == 'epde_deep':
                # Add Prompt information from 1st layer
                # TODO: Why ViPT use i - 1
                image_feat = token2feature(self.prompt_norms[i - 1](image))
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
            x, self.foundation.intermediate_layer_idx[self.encoder], return_class_token=True
        )

        depth = self.foundation.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)

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
        encoder='vits',
        **kwargs,
    )
    return model


def epde_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = EPDEVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        encoder='vitb',
        **kwargs,
    )
    return model


def epde_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = EPDEVisionTransformer(
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
    model = EPDEVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        encoder='vitg',
        **kwargs,
    )
    return model

def EPDE(
    model_name,
    dataset='dense',
    depth_anything_pretrained=None,
    max_depth=80.0,
    embed_layer=PatchEmbed,
    norm_layer=None,
    prompt_type=None,
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
        # in_chans=3,
        event_voxel_chans=5,
        dataset=dataset,
        depth_anything_pretrained=depth_anything_pretrained,
        embed_layer=embed_layer,
        norm_layer=norm_layer,
        prompt_type=prompt_type,
    )
