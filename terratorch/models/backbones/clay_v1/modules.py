import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers import use_fused_attn
from torch import Tensor, nn

from terratorch.models.backbones.clay_v1.utils import posemb_sincos_1d, posemb_sincos_2d_with_gsd

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"

# central wavelengths of pretrained model
WAVELENGTHS = {
    "blue": 0.493,
    "green": 0.56,
    "red": 0.665,
    "rededge1": 0.704,
    "rededge2": 0.74,
    "rededge3": 0.783,
    "nir": 0.842,
    "nir08": 0.865,
    "swir16": 1.61,
    "swir22": 2.19,
    "vv": 3.5,  # TODO: choose one vv
    "vh": 4.0,  # TODO: choose one vh
    "COASTAL_AEROSOL": 0.44,
    "BLUE": 0.49,
    "GREEN": 0.56,
    "RED": 0.665,
    "RED_EDGE_1": 0.705,
    "RED_EDGE_2": 0.74,
    "RED_EDGE_3": 0.783,
    "NIR_BROAD": 0.832,
    "NIR_NARROW": 0.864,
    "WATER_VAPOR": 0.945,
    "CIRRUS": 1.373,
    "SWIR_1": 1.61,
    "SWIR_2": 2.20,
    "THERMAL_INFRARED_1": 10.90,
    "THERMAL_INFRARED_12": 12.00,
    "VV": 5.405,
    "VH": 5.405,
    "ASC_VV": 5.405,
    "ASC_VH": 5.405,
    "DSC_VV": 5.405,
    "DSC_VH": 5.405,
    "VV-VH": 5.405,
}


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if use_fused_attn():
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        vpt: bool = False,
        vpt_n_tokens: int | None = None,
        vpt_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.vpt = vpt
        self.vpt_n_tokens = vpt_n_tokens
        self.vpt_dropout = vpt_dropout
        if self.vpt:
            if self.vpt_n_tokens is None:
                msg = "vpt_n_tokens must be provided when using VPT"
                raise ValueError(msg)
            self.vpt_prompt_embeddings = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, self.vpt_n_tokens, dim)) for _ in range(depth)]
            )
            self.vpt_dropout_layers = nn.ModuleList([nn.Dropout(vpt_dropout) for _ in range(depth)])
            val = np.sqrt(6.0 / float(3 * 8**2 + dim))
            for emb in self.vpt_prompt_embeddings:
                nn.init.uniform_(emb, -val, val)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([Attention(dim, heads=heads, dim_head=dim_head), FeedForward(dim, mlp_dim)])
            )

    def forward(self, x) -> list[torch.Tensor]:
        bs = x.shape[0]
        out = []
        for idx, (attn, ff) in enumerate(self.layers):
            if self.vpt:
                x = torch.cat(
                    (
                        x[:, :1, :],
                        self.vpt_dropout_layers[idx](self.vpt_prompt_embeddings[idx].expand(bs, -1, -1)),
                        x[:, 1:, :],
                    ),
                    dim=1,
                )  # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
            x = attn(x) + x
            x = ff(x) + x
            if self.vpt:
                x = torch.cat(
                    (x[:, :1, :], x[:, (1 + self.vpt_n_tokens) :, :]),
                    dim=1,
                )  # (batch_size, cls_token + n_patches, hidden_dim)
            out.append(x.clone())
        x = self.norm(x)
        out[-1] = x.clone()
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        mask_ratio,
        patch_size,
        shuffle,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        vpt: bool = False,
        vpt_n_tokens: int | None = None,
        vpt_dropout: float = 0.0,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.patch_embedding = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=False,
        )

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
            vpt=vpt,
            vpt_n_tokens=vpt_n_tokens,
            vpt_dropout=vpt_dropout,
        )

    def to_patch_embed(self, cube, waves):
        """Split the input cube into patches & create embeddings per patch"""
        patches, waves_encoded = self.patch_embedding(cube, waves)  # [B L D]
        return patches, waves_encoded  # ([B L D], [N D])

    def add_encodings(self, patches, time, latlon, gsd):
        """Add position encoding to the patches"""
        B, L, D = patches.shape

        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2

        pos_encoding = (
            posemb_sincos_2d_with_gsd(
                h=grid_size,
                w=grid_size,
                dim=(self.dim - 8),
                gsd=gsd,
            )
            .to(patches.device)
            .detach()
        )  # [L (D - 8)]

        time_latlon = torch.hstack((time, latlon)).to(patches.device).detach()  # [B 8]

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)  # [B L (D - 8)]
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)  # [B L 8]
        pos_metadata_encoding = torch.cat((pos_encoding, time_latlon), dim=-1)  # [B L D]

        # [B L D] + [B L D] -> [B L D]
        patches = patches + pos_metadata_encoding
        return patches  # [B L D]

    def mask_out(self, patches):
        """
        Mask out patches randomly by shuffling the patches & masking out the
        first N patches

        Parameters
        ----------
        patches : torch.Tensor A tensor of shape (B, L, D)

        Returns
        -------
        unmasked_patches : torch.Tensor
            A tensor of shape (B, L:(1 - mask_ratio), D) containing the
            embeddings of the unmasked patches.
        unmasked_indices : torch.Tensor
            A tensor of shape (B, (1 - mask_ratio)) containing the indices of
            the unmasked patches.
        masked_indices : torch.Tensor
            A tensor of shape (B, mask_ratio) containing the indices of the
            masked patches.
        masked_matrix : torch.Tensor
            A tensor of shape (B, L) containing the mask matrix, 1 indicates a masked
            patch & 0 indicates an unmasked patch.
        """
        B, L, D = patches.shape
        # assert (
        #     L == self.num_patches
        # ), f"Expected {self.num_patches} patches, got {L} patches."

        if self.shuffle:  # Shuffle the patches
            noise = torch.randn((B, L), device=patches.device)  # [B L]
        else:  # Don't shuffle, useful for interpolation & inspection of embeddings
            noise = rearrange(torch.arange(B * L, device=patches.device), "(B L) -> B L", B=B, L=L)

        random_indices = torch.argsort(noise, dim=-1)  # [B L]
        reverse_indices = torch.argsort(random_indices, dim=-1)  # [B L]

        num_masked_patches = int(self.mask_ratio * self.num_patches)  # Number of patches to be masked out
        masked_indices, unmasked_indices = (
            random_indices[:, :num_masked_patches],  # [B mask_ratio * L]
            random_indices[:, num_masked_patches:],  # [B (1 - mask_ratio) * L]
        )

        # create a mask of shape B L, where 1 indicates a masked patch
        # and 0 indicates an unmasked patch
        masked_matrix = torch.zeros((B, L), device=patches.device)  # [B L] = 0
        masked_matrix[:, :num_masked_patches] = 1  # [B mask_ratio * L] = 1
        masked_matrix = torch.gather(
            masked_matrix, dim=1, index=reverse_indices
        )  # [B L] -> [B L] - reorder the patches

        # mask out the patches
        batch_indices = rearrange(torch.arange(B, device=patches.device), "B -> B 1")  # [B 1]
        unmasked_patches = patches[batch_indices, unmasked_indices, :]  # [B L:(1 - mask_ratio) D]
        _ = patches[batch_indices, masked_indices, :]  # [B L:mask_ratio D]

        return (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B L:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B L]

    def forward(self, datacube):
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )  # [B C H W]

        B, C, H, W = cube.shape

        patches, waves_encoded = self.to_patch_embed(cube, waves)  # [B L D] - patchify & create embeddings per patch
        patches = self.add_encodings(
            patches,
            time,
            latlon,
            gsd,
        )  # [B L D] - add position encoding to the embeddings

        # mask out patches
        (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.mask_out(patches)  # [B L:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B L]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        unmasked_patches = torch.cat((cls_tokens, unmasked_patches), dim=1)  # [B (1 + L) D]

        # pass the unmasked patches through the transformer
        encoded_unmasked_patches = self.transformer(unmasked_patches)  # [B ((1 + L)):(1 - mask_ratio)) D]

        return (
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B ((1 + L):(1 - mask_ratio)) D], [(1-mask_ratio)], [mask_ratio], [B L]


class EmbeddingEncoder(Encoder):
    """Clay Encoder without mask and shuffle."""

    def __init__(
        self,
        img_size,
        patch_size,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        vpt: bool = False,
        vpt_n_tokens: int | None = None,
        vpt_dropout: float = 0.0,
    ):
        super().__init__(
            mask_ratio=0.0,
            shuffle=False,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            vpt=vpt,
            vpt_n_tokens=vpt_n_tokens,
            vpt_dropout=vpt_dropout,
        )
        self.img_size = img_size

        # Using fixed grid       size for inference
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2

    def add_encodings(self, patches, time, latlon, gsd):
        """Add position encoding to the patches"""
        B, L, D = patches.shape

        grid_size = self.grid_size

        pos_encoding = (
            posemb_sincos_2d_with_gsd(
                h=grid_size,
                w=grid_size,
                dim=(self.dim - 8),
                gsd=gsd,
            )
            .to(patches.device)
            .detach()
        )  # [L (D - 8)]

        time_latlon = torch.hstack((time, latlon)).to(patches.device).detach()  # [B 8]

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)  # [B L (D - 8)]
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)  # [B L 8]
        pos_metadata_encoding = torch.cat((pos_encoding, time_latlon), dim=-1)  # [B L D]

        # [B L D] + [B L D] -> [B L D]
        patches = patches + pos_metadata_encoding
        return patches  # [B L D]

    # def forward(self, cube, time, latlon, waves, gsd):
    def forward(self, datacube):
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )  # [B C H W]
        B, C, H, W = cube.shape

        patches, _ = self.to_patch_embed(cube, waves)  # [B L D] - patchify & create embeddings per patch

        # Add time & latlon as encoding to patches
        patches = self.add_encodings(
            patches,
            time,
            latlon,
            gsd,
        )  # [B L D] - add position encoding to the embeddings

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [B (1 + L) D]

        # pass the patches through the transformer
        patches = self.transformer(patches)  # list of [B (1 + L) D]

        # # remove the cls token
        # embeddings = patches[:, 1: , :]  # [B L D]

        return patches  # list [B (1 + L) D]


class FCBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.l1 = nn.Linear(size, size)
        self.l2 = nn.Linear(size, size)

    def forward(self, x):
        y = F.gelu(self.l1(x))
        y = F.gelu(self.l2(y))
        return x + y


class WavesTransformer(nn.Module):
    def __init__(
        self,
        wave_dim,
        output_dim,
        num_latent_tokens,
        embed_dim,
        is_decoder,
        num_heads=4,
        num_layers=1,
    ):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.is_decoder = is_decoder
        layer = nn.TransformerEncoderLayer(
            d_model=wave_dim,
            nhead=num_heads,
            activation="gelu",
            dropout=0,
            norm_first=False,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.fc_weight = nn.Linear(wave_dim, output_dim)
        self.fc_bias = None if self.is_decoder else nn.Linear(wave_dim, embed_dim)

        self.weight_tokens = nn.Parameter(torch.randn(self.num_latent_tokens, wave_dim) * 0.02)
        self.bias_token = nn.Parameter(torch.randn(1, wave_dim) * 0.02)

    def forward(self, x):
        x = torch.cat([self.weight_tokens, x, self.bias_token], dim=0)
        out = self.encoder(x)
        weights = self.fc_weight(out[self.num_latent_tokens : -1] + x[self.num_latent_tokens : -1])
        bias = None if self.is_decoder else self.fc_bias(out[-1])
        return weights, bias


class DynamicEmbedding(nn.Module):
    def __init__(
        self,
        wave_dim,
        num_latent_tokens,
        patch_size,
        embed_dim,
        is_decoder=False,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.is_decoder = is_decoder
        self.output_dim = (patch_size**2) * embed_dim

        self.weight_generator = WavesTransformer(
            wave_dim,
            self.output_dim,
            self.num_latent_tokens,
            self.embed_dim,
            is_decoder,
        )
        self.fclayer = FCBlock(self.wave_dim)

        self.initialize_weights()

    def forward(self, batch, waves):
        waves = posemb_sincos_1d(waves, self.wave_dim)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)

        if self.is_decoder:
            dynamic_weight = rearrange(
                weight,
                "cin (k1 k2 cout) -> (cin k1 k2) cout",
                k1=self.patch_size,
                k2=self.patch_size,
                cout=self.embed_dim,
            )
            if bias is not None:
                bias = rearrange(bias, "b -> (b)")
            dynamic_out = F.linear(batch, dynamic_weight * 0.02, bias=bias)
            x = dynamic_out
        else:
            dynamic_weight = rearrange(
                weight,
                "cin (cout k1 k2) -> cout cin k1 k2",
                k1=self.patch_size,
                k2=self.patch_size,
            )
            if bias is not None:
                bias = rearrange(bias, "b -> (b)")
            dynamic_out = F.conv2d(batch, dynamic_weight * 0.02, bias=bias, stride=self.patch_size)
            x = rearrange(dynamic_out, "b c h w -> b (h w) c")

        return x, waves

    def initialize_weights(self):
        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Datacuber(nn.Module):
    def __init__(self, bands=None) -> None:
        super().__init__()
        self.bands = bands

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor | None = None,
        latlon: torch.Tensor | None = None,
        waves: torch.Tensor | None = None,
        gsd: float | None = None,
    ) -> dict[str, torch.Tensor | float]:
        datacube: dict[str, torch.Tensor | float] = {}
        datacube["pixels"] = x
        datacube["time"] = torch.zeros((x.shape[0], 4), device=x.device) if time is None else time
        datacube["latlon"] = torch.zeros((x.shape[0], 4), device=x.device) if latlon is None else latlon
        datacube["gsd"] = 1.0 if gsd is None else gsd
        datacube["waves"] = self._parse_wavelengths(self.bands, x.shape[1]).to(x.device) if waves is None else waves
        return datacube

    def _parse_wavelengths(self, bands, channels):
        waves = torch.tensor([WAVELENGTHS[band] if band in WAVELENGTHS.keys() else 0.0 for band in bands])
        return waves
