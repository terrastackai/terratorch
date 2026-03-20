# Copyright contributors to the Terratorch project

import torch
from timm.models.vision_transformer import Block, PatchEmbed
from torch import nn

from terratorch.registry import TERRATORCH_DECODER_REGISTRY


@TERRATORCH_DECODER_REGISTRY.register
class SatMAEHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = None,
        out_channels: int = None,
        bias: bool = True,
        num_heads: int = None,
        mlp_ratio: float = None,
        patch_size: int = None,
        num_patches: int = 1,
        depth: int = None,
        norm_layer=nn.LayerNorm,
        in_chans: int = None,
    ) -> None:

        super().__init__()

        if type(embed_dim) == tuple:
            self.embed_dim = embed_dim[1]
        else:
            self.embed_dim = embed_dim

        self.out_channels = out_channels
        self.out_channels = out_channels
        self.bias = bias
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.depth = depth

        self.decoder_embed = nn.Linear(self.embed_dim, self.out_channels, bias=self.bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.out_channels))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.out_channels), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    self.out_channels,
                    self.num_heads,
                    self.mlp_ratio,
                    qkv_bias=True,
                    qk_norm=None,
                    norm_layer=self.norm_layer,
                )
                for i in range(self.depth)
            ]
        )

        self.decoder_norm = self.norm_layer(self.out_channels)
        self.decoder_pred = nn.Linear(
            self.out_channels, self.patch_size**2 * self.in_chans, bias=True
        )  # decoder to patch

    def unpatchify(self, x, p, c):
        """
        x: (N, L, patch_size**2 *C)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, ids_restore) -> torch.Tensor:

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        x = self.unpatchify(x, self.patch_size, self.in_chans)

        return x


class SatMAEHeadViT(nn.Module):
    def __init__(
        self,
        embed_dim: int = None,
        out_channels: int = None,
        bias: bool = True,
        num_heads: int = None,
        mlp_ratio: float = None,
        patch_size: int = None,
        num_patches: int = 1,
        depth: int = None,
        norm_layer=nn.LayerNorm,
        in_chans: int = None,
    ) -> None:

        super().__init__()

        if type(embed_dim) == tuple:
            self.embed_dim = embed_dim[1]
        else:
            self.embed_dim = embed_dim

        self.out_channels = out_channels
        self.bias = bias
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.depth = depth

        self.decoder_embed = nn.Linear(self.embed_dim, self.out_channels, bias=self.bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.out_channels))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.out_channels), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    self.out_channels,
                    self.num_heads,
                    self.mlp_ratio,
                    qkv_bias=True,
                    qk_norm=None,
                    norm_layer=self.norm_layer,
                )
                for i in range(self.depth)
            ]
        )

        self.decoder_norm = self.norm_layer(self.out_channels)
        self.decoder_pred = nn.Linear(
            self.out_channels, self.patch_size**2 * self.in_chans, bias=True
        )  # decoder to patch

    def unpatchify(self, x, p, c):
        """
        x: (N, L, patch_size**2 *C)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x) -> torch.Tensor:

        x = torch.stack(x, dim=0)

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        # x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        x = self.unpatchify(x, self.patch_size, self.in_chans)

        return x
