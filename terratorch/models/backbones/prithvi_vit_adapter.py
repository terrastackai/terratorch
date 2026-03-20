import logging
import math
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.nn.init import normal_, trunc_normal_

logger = logging.getLogger(__name__)

from terratorch.models.backbones.prithvi_mae import PrithviViT

try:
    from terratorch.models.backbones.detr_ops.modules.ms_deform_attn import MSDeformAttn
    from terratorch.models.backbones.vit_adapter_modules import (
        InteractionBlock,
        SpatialPriorModule,
        deform_inputs,
    )

    class PrithviViTAdapter(PrithviViT):  # type: ignore
        """Prithvi ViT Adapter Encoder
        Args:
            interaction_indexes (list[list[int]]): List of indexes for each interaction block.
            conv_inplane (int): Number of channels for Spatial Prior Module.
            add_vit_feature (bool): Whether to add ViT features to the output.
            deform_num_heads (int): Number of heads for the deformable attention.
            n_points (int): Number of points for the deformable attention.
            init_values (float): Initial values for the deformable attention.
            with_cffn (bool): Whether to use the cffn in the interaction block.
            cffn_ratio (float): Ratio for the cffn in the interaction block.
            deform_ratio (float): Ratio for the deformable attention.
            use_extra_extractor (bool): Whether to use the extra extractor in the interaction block.
            with_cp (bool): Whether to use checkpointing.
            img_size (int | tuple[int, int]): Size of the input image.
            patch_size (int | tuple[int, int, int]): Size of the patches.
            num_frames (int): Number of frames in the input.
            in_chans (int): Number of input channels.
            embed_dim (int): Dimension of the embedding.
            depth (int): Depth of the model.
            num_heads (int): Number of heads in the model.
            mlp_ratio (float): Ratio for the mlp in the model.
            norm_layer (type[nn.Module]): Normalization layer to use.
            coords_encoding (list[str] | None): List of coordinate encodings to use.
            coords_scale_learn (bool): Whether to learn the scale of the coordinates.
            drop_path (float): Drop path rate.
        """

        extra_layers = ("level_embed", "spm", "interactions", "up", "norm1", "norm2", "norm3", "norm4")

        def __init__(
            self,
            # Adapter params
            interaction_indexes: list[list[int]],
            conv_inplane: int,
            add_vit_feature: bool,
            deform_num_heads: int,
            n_points: int,
            init_values: float,
            with_cffn: bool,
            cffn_ratio: float,
            deform_ratio: float,
            use_extra_extractor: bool,
            with_cp: bool,
            # PrithviViT params
            img_size: int | tuple[int, int] = 224,
            patch_size: int | tuple[int, int, int] = (1, 16, 16),
            num_frames: int = 1,
            in_chans: int = 3,
            embed_dim: int = 1024,
            depth: int = 24,
            num_heads: int = 16,
            mlp_ratio: float = 4.0,
            norm_layer: type[nn.Module] = nn.LayerNorm,
            coords_encoding: list[str] | None = None,
            coords_scale_learn: bool = False,
            drop_path: float = 0.0,
            **kwargs: dict[str, Any],
        ) -> None:
            if num_frames != 1:
                msg = "PrithviViTAdapter only supports num_frames=1"
                raise ValueError(msg)
            if isinstance(patch_size, tuple) and patch_size[0] != 1:
                msg = "PrithviViTAdapter only supports patch_size[0]=1"
                raise ValueError(msg)
            super().__init__(
                img_size=img_size,
                patch_size=patch_size,
                num_frames=num_frames,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                coords_encoding=coords_encoding,
                coords_scale_learn=coords_scale_learn,
                drop_path=drop_path,
                **kwargs,
            )

            # We don't use the original cls_token and norm layer
            del self.cls_token
            del self.norm

            self.num_block = len(self.blocks)
            self.interaction_indexes = interaction_indexes
            self.add_vit_feature = add_vit_feature
            self.out_channels = [self.embed_dim * self.patch_embed.grid_size[0]] * 4

            embed_dim = self.embed_dim
            self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
            self.spm = SpatialPriorModule(
                inplanes=conv_inplane,
                embed_dim=embed_dim,
                in_channels=self.in_chans,
                with_cp=False,
            )
            self.interactions = nn.Sequential(
                *[
                    InteractionBlock(
                        dim=embed_dim,
                        num_heads=deform_num_heads,
                        n_points=n_points,
                        init_values=init_values,
                        drop_path=drop_path,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        extra_extractor=((i == len(interaction_indexes) - 1) and use_extra_extractor),
                        with_cp=with_cp,
                        drop=0.0,
                    )
                    for i in range(len(interaction_indexes))
                ]
            )
            self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
            self.norm1 = nn.SyncBatchNorm(embed_dim)
            self.norm2 = nn.SyncBatchNorm(embed_dim)
            self.norm3 = nn.SyncBatchNorm(embed_dim)
            self.norm4 = nn.SyncBatchNorm(embed_dim)

            self.up.apply(self._init_weights)
            self.spm.apply(self._init_weights)
            self.interactions.apply(self._init_weights)
            self.apply(self._init_deform_weights)
            normal_(self.level_embed)

        def _init_weights(self, m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm | nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

        def _init_deform_weights(self, m: nn.Module) -> None:
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        def _add_level_embed(
            self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            c2 = c2 + self.level_embed[0]
            c3 = c3 + self.level_embed[1]
            c4 = c4 + self.level_embed[2]
            return c2, c3, c4

        def forward_features(
            self,
            x: torch.Tensor,
            temporal_coords: None | torch.Tensor = None,
            location_coords: None | torch.Tensor = None,
        ) -> list[torch.Tensor]:
            # add time dim for patch embedding.
            if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
                # add time dim
                x = x.unsqueeze(2)
            if x.shape[2] != 1:
                msg = "Input tensor must have 1 frame"
                raise ValueError(msg)
            sample_shape = x.shape[-3:]

            deform_inputs1, deform_inputs2 = deform_inputs(x.squeeze(2))

            # SPM forward
            c1, c2, c3, c4 = self.spm(x.squeeze(2))
            c2, c3, c4 = self._add_level_embed(c2, c3, c4)
            c = torch.cat([c2, c3, c4], dim=1)

            # Patch Embedding forward
            H = int(x.shape[-2] / self.patch_embed.patch_size[1])  # noqa: N806
            W = int(x.shape[-1] / self.patch_embed.patch_size[2])  # noqa: N806
            x = self.patch_embed(x)
            bs, n, dim = x.shape
            pos_embed = self.interpolate_pos_encoding((sample_shape[0], sample_shape[1], sample_shape[2]))

            # We don't have dropout in Prithvi
            # x = self.pos_drop(x + pos_embed)
            # add pos embed w/o cls token
            x = x + pos_embed[:, 1:, :]

            if self.temporal_encoding and temporal_coords is not None:
                num_tokens_per_frame = x.shape[1] // self.num_frames
                temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
                x = x + temporal_encoding
            if self.location_encoding and location_coords is not None:
                location_encoding = self.location_embed_enc(location_coords)
                x = x + location_encoding

            # Interaction
            outs = []
            for i, layer in enumerate(self.interactions):
                indexes = self.interaction_indexes[i]
                x, c = layer(
                    x,
                    c,
                    self.blocks[indexes[0] : indexes[-1] + 1],
                    deform_inputs1,
                    deform_inputs2,
                    H,
                    W,
                )
                outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

            # Split & Reshape
            c2 = c[:, 0 : c2.size(1), :]
            c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
            c4 = c[:, c2.size(1) + c3.size(1) :, :]

            c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
            c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
            c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
            c1 = self.up(c2) + c1

            if self.add_vit_feature:
                x1, x2, x3, x4 = outs
                x1 = F.interpolate(x1, scale_factor=4, mode="bilinear", align_corners=False)
                x2 = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
                x4 = F.interpolate(x4, scale_factor=0.5, mode="bilinear", align_corners=False)
                c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

            # Final Norm
            f1 = self.norm1(c1)
            f2 = self.norm2(c2)
            f3 = self.norm3(c3)
            f4 = self.norm4(c4)
            return [f1, f2, f3, f4]

        def prepare_features_for_image_model(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
            return features

        def freeze(self):
            """Freeze the ViT, not the adapter."""
            for n, param in super().named_parameters():
                if not n.startswith(PrithviViTAdapter.extra_layers):
                    param.requires_grad_(False)
except ImportError as err:
    # If the import fails, it means that the required modules are not available.
    # This is expected if the user has not installed the optional dependencies.
    # We define a dummy class to avoid breaking the code.
    class PrithviViTAdapter(nn.Module):
        err: ImportError = err

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            msg = (
                "While defining PrithviViTAdapter, the following error occurred:\n"
                f"{self.err}.\nPlease install the optional dependencies to use this feature."
            )
            raise ImportError(msg)
