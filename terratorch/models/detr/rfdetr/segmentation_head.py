# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Adapted for TerraTorch: ruff compliance.

import torch
import torch.nn.functional as f_nn
from torch import nn


class DepthwiseConvBlock(nn.Module):
    """Simplified ConvNeXt block without the MLP subnet."""

    def __init__(self, dim, layer_scale_init_value=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        return x + residual


class MLPBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=0):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            ]
        )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        residual = x
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x)
        if self.gamma is not None:
            x = self.gamma * x
        return x + residual


class SegmentationHead(nn.Module):
    def __init__(self, in_dim, num_blocks: int, bottleneck_ratio: int = 1, downsample_ratio: int = 4):
        super().__init__()

        self.downsample_ratio = downsample_ratio
        self.interaction_dim = in_dim // bottleneck_ratio if bottleneck_ratio is not None else in_dim
        self.blocks = nn.ModuleList([DepthwiseConvBlock(in_dim) for _ in range(num_blocks)])
        self.spatial_features_proj = (
            nn.Identity() if bottleneck_ratio is None else nn.Conv2d(in_dim, self.interaction_dim, kernel_size=1)
        )

        self.query_features_block = MLPBlock(in_dim)
        self.query_features_proj = (
            nn.Identity() if bottleneck_ratio is None else nn.Linear(in_dim, self.interaction_dim)
        )

        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(
        self,
        spatial_features: torch.Tensor,
        query_features: list[torch.Tensor],
        image_size: tuple[int, int],
        skip_blocks: bool = False,  # noqa: FBT001, FBT002
    ) -> list[torch.Tensor]:
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        spatial_features = f_nn.interpolate(spatial_features, size=target_size, mode="bilinear", align_corners=False)

        mask_logits = []
        if not skip_blocks:
            for block, qf in zip(self.blocks, query_features, strict=False):
                spatial_features = block(spatial_features)
                spatial_features_proj = self.spatial_features_proj(spatial_features)
                qf_proj = self.query_features_proj(self.query_features_block(qf))
                mask_logits.append(torch.einsum("bchw,bnc->bnhw", spatial_features_proj, qf_proj) + self.bias)
        else:
            qf_proj = self.query_features_proj(self.query_features_block(query_features[0]))
            mask_logits.append(torch.einsum("bchw,bnc->bnhw", spatial_features, qf_proj) + self.bias)

        return mask_logits

    def sparse_forward(
        self,
        spatial_features: torch.Tensor,
        query_features: list[torch.Tensor],
        image_size: tuple[int, int],
        skip_blocks: bool = False,  # noqa: FBT001, FBT002
    ) -> list[torch.Tensor]:
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        spatial_features = f_nn.interpolate(spatial_features, size=target_size, mode="bilinear", align_corners=False)

        output_dicts = []

        if not skip_blocks:
            for block, qf in zip(self.blocks, query_features, strict=False):
                spatial_features = block(spatial_features)
                spatial_features_proj = self.spatial_features_proj(spatial_features)
                qf_proj = self.query_features_proj(self.query_features_block(qf))

                output_dicts.append(
                    {
                        "spatial_features": spatial_features_proj,
                        "query_features": qf_proj,
                        "bias": self.bias,
                    }
                )
        else:
            qf_proj = self.query_features_proj(self.query_features_block(query_features[0]))

            output_dicts.append(
                {
                    "spatial_features": spatial_features,
                    "query_features": qf_proj,
                    "bias": self.bias,
                }
            )

        return output_dicts


def point_sample(input_tensor, point_coords, **kwargs):
    """A wrapper around torch.nn.functional.grid_sample for 3D point_coords tensors.

    Args:
        input_tensor: A tensor of shape (N, C, H, W).
        point_coords: A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) in [0, 1] x [0, 1].

    Returns:
        output: A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid).
    """
    add_dim = False
    if point_coords.dim() == 3:  # noqa: PLR2004
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = f_nn.grid_sample(input_tensor, 2.0 * point_coords - 1.0, padding_mode="border", **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio=3, importance_sample_ratio=0.75
):
    """Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty."""
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """Estimate uncertainty as L1 distance between 0.0 and the logit prediction."""
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))
