# Adapted from https://github.com/bair-climate-initiative/scale-mae/blob/main/mae/models_vit.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import logging
from collections.abc import Callable
from functools import partial

import numpy as np
import timm.models.vision_transformer
import torch
from einops import rearrange
from timm.models import FeatureInfo
from timm.models.vision_transformer import PatchEmbed
from torch import nn

from terratorch.datasets.utils import HLSBands
from terratorch.models.backbones.select_patch_embed_weights import select_patch_embed_weights

logger = logging.getLogger(__name__)

scalemae_model_registry: dict[str, Callable] = {}

PRETRAINED_BANDS = [
    HLSBands.RED,
    HLSBands.GREEN,
    HLSBands.BLUE,
]


def register_scalemae_model(constructor: Callable):
    scalemae_model_registry[constructor.__name__] = constructor
    return constructor


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed_with_resolution(embed_dim, grid_size, res, cls_token=False, device="cpu"):
    """
    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),
    return:
    pos_embed: [n,grid_size*grid_size, embed_dim] or [n,1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # res = torch.FloatTensor(res).to(device)
    res = res.to(device)
    grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    # grid = grid.reshape([2, 1, grid_size, grid_size])
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)  #  # (nxH*W, D/2)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [
                torch.zeros([n, 1, embed_dim], dtype=torch.float32, device=pos_embed.device),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed


class PatchEmbedUnSafe(PatchEmbed):
    """Image to Patch Embedding"""

    def forward(self, x):
        B, C, H, W = x.shape
        # Dropped size check in timm
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer for ScaleMAE model

    Adapted from `https://github.com/bair-climate-initiative/scale-mae/blob/main/mae/models_vit.py`.
    Vision transformer with scale-dependent position embedding.
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=1024, out_indices=None, default_input_res=1, **kwargs):
        """
        Args:
            patch_size (int, optional): Patch size. Defaults to 16.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Embedding dimension. Defaults to 1024.
            out_indices (_type_, optional): Indices of transformer blocks to be output as features. Defaults to None.
            default_input_res (int, optional): GSD of the input. If not passed through
                the dataset, this value will be used by default. Defaults to 1.
        """
        super().__init__(embed_dim=embed_dim, **kwargs)

        self.patch_embed = PatchEmbedUnSafe(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # preserve old forward
        self.encode_decode_forward = self.forward

        self.default_input_res = default_input_res

        self.out_indices = out_indices if out_indices else [-1]

        self.out_channels = [embed_dim] * len(self.out_indices)

        del self.fc_norm
        del self.head_drop
        del self.head

    def forward(self, x, input_res=None):
        B, _, h, w = x.shape
        x = self.patch_embed(x)

        if input_res is None:
            input_res = self.default_input_res
            input_res = torch.FloatTensor([input_res] * B)

        num_patches = int((h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1]))
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches**0.5),
            input_res,
            cls_token=True,
            device=x.device,
        )

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        output = []
        for blk in self.blocks:
            x = blk(x)
            output.append(x.clone())

        return [output[i] for i in self.out_indices]

    def prepare_features_for_image_model(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        for x in features:
            x_no_token = x[:, 1:, :]
            number_of_tokens = x_no_token.shape[1]
            h = int(np.sqrt(number_of_tokens))
            encoded = rearrange(
                x_no_token,
                "batch (h w) e -> batch e h w",
                e=self.embed_dim,
                h=h,
            )
            out.append(encoded)
        return out


@register_scalemae_model
def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


@register_scalemae_model
def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


@register_scalemae_model
def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def load_scalemae_weights(
    model: nn.Module, ckpt_data: str, model_bands: list[HLSBands], input_size: int = 224
) -> nn.Module:
    checkpoint_model = torch.load(ckpt_data, map_location="cpu", weights_only=True)["model"]
    state_dict = model.state_dict()

    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            logger.info(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    if input_size != 224:
        if "pos_embed" in checkpoint_model and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape:
            logger.info("Removing key pos_embed from pretrained checkpoint")
            del checkpoint_model["pos_embed"]

    checkpoint_model = select_patch_embed_weights(checkpoint_model, model, PRETRAINED_BANDS, model_bands)
    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)

    logger.info(msg)
    return model


def create_model(
    model_name: str, ckpt_path: str | None = None, bands: list[HLSBands | int | str] | None = None, **kwargs
):
    input_size = kwargs.pop("input_size", 224)
    try:
        constructor: Callable = scalemae_model_registry[model_name]
    except KeyError as e:
        msg = f"Model {model_name} not in registry. Possible models are {list(scalemae_model_registry.keys())}"
        raise Exception(msg) from e

    if bands is None:
        bands: list[HLSBands | int | str] = PRETRAINED_BANDS
        logger.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {PRETRAINED_BANDS}.\
            Pretrained patch_embed layer may be misaligned with current bands"
        )
    else:
        bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in bands]
    kwargs["in_chans"] = len(bands)
    model = constructor(**kwargs)

    if ckpt_path:
        load_scalemae_weights(model, ckpt_path, bands, input_size)

    # ScaleMAE does not use the pos_embed within ViT
    model.pos_embed.requires_grad = False

    return model
