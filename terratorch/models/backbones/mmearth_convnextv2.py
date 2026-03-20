# code from https://github.com/vishalned/MMEarth-train/blob/main/models/convnextv2.py
# https://github.com/vishalned/MMEarth-train?tab=readme-ov-file
# Copyright (c) Meta Platforms, Inc. and affiliates.
from argparse import Namespace

import torch
from timm.layers import DropPath, LayerNorm, trunc_normal_

# from .norm_layers import LayerNorm, GRN
from timm.layers import GlobalResponseNormMlp as GRN
from torch import Tensor, nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv: nn.Module = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depth-wise conv
        self.norm: nn.Module = LayerNorm(dim, eps=1e-6)
        self.pwconv1: nn.Module = nn.Linear(dim, 4 * dim)  # point-wise/1x1 convs, implemented with linear layers
        self.act: nn.Module = nn.GELU()
        self.grn: nn.Module = GRN(4 * dim)
        self.pwconv2: nn.Module = nn.Linear(4 * dim, dim)
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        patch_size: int = 32,
        img_size: int = 128,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: list[int] = None,
        dims: list[int] = None,
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        use_orig_stem: bool = False,
        args: Namespace = None,
    ):
        super().__init__()
        self.depths = depths
        if self.depths is None:  # set default value
            self.depths = [3, 3, 9, 3]
        self.img_size = img_size
        self.use_orig_stem = use_orig_stem
        self.num_stage = len(depths)
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layer
        self.patch_size = patch_size
        if dims is None:
            dims = [96, 192, 384, 768]

        if self.use_orig_stem:
            self.stem_orig = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    dims[0],
                    kernel_size=patch_size // (2 ** (self.num_stage - 1)),
                    stride=patch_size // (2 ** (self.num_stage - 1)),
                ),
                LayerNorm(dims[0], eps=1e-6),  # data_format="channels_first"),
            )
        else:
            self.initial_conv = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1),
                LayerNorm(dims[0], eps=1e-6),  # data_format="channels_first"),
                nn.GELU(),
            )
            # depthwise conv for stem
            self.stem = nn.Sequential(
                nn.Conv2d(
                    dims[0],
                    dims[0],
                    kernel_size=patch_size // (2 ** (self.num_stage - 1)),
                    stride=patch_size // (2 ** (self.num_stage - 1)),
                    padding=(patch_size // (2 ** (self.num_stage - 1))) // 2,
                    groups=dims[0],
                ),
                LayerNorm(dims[0], eps=1e-6),  # data_format="channels_first"),
            )

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6),  # data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        if self.use_orig_stem:
            x = self.stem_orig(x)
        else:
            x = self.initial_conv(x)
            x = self.stem(x)

        x = self.stages[0](x)
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i + 1](x)

        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** 0.5)
        return mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        if mask is not None:  # for the pretraining case
            num_patches = mask.shape[1]
            scale = int(self.img_size // (num_patches**0.5))
            mask = self.upsample_mask(mask, scale)

            mask = mask.unsqueeze(1).type_as(x)
            x *= 1.0 - mask
            if self.use_orig_stem:
                x = self.stem_orig(x)
            else:
                x = self.initial_conv(x)
                x = self.stem(x)

            x = self.stages[0](x)
            for i in range(3):
                x = self.downsample_layers[i](x)
                x = self.stages[i + 1](x)
            return x

        x = self.forward_features(x)
        x = self.head(x)
        return x


@TERRATORCH_BACKBONE_REGISTRY.register
def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


checkpoints = {
    "pt-S2_atto_1M_64_uncertainty_56-8": "https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-S2_atto_1M_64_uncertainty_56-8/checkpoint-199.pth",
    "pt-all_mod_atto_100k_128_uncertainty_112-16": "https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-all_mod_atto_100k_128_uncertainty_112-16/checkpoint-199.pth",
    "pt-all_mod_atto_1M_128_uncertainty_112-16": "https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-all_mod_atto_1M_128_uncertainty_112-16/checkpoint-199.pth",
    "pt-all_mod_atto_1M_64_uncertainty_56-8": "https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-all_mod_atto_1M_64_uncertainty_56-8/checkpoint-199.pth",
}
