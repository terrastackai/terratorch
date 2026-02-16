# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Convolutional heads for mask prediction, and associated losses.

Adapted from the DETR reference implementation for TerraTorch integration.
Original: https://github.com/facebookresearch/detr/blob/main/models/segmentation.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as f_nn
from torch import Tensor, nn


def _expand(tensor: Tensor, length: int) -> Tensor:
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """Simple convolutional head, using group norm.

    Upsampling is done using a FPN approach.
    """

    def __init__(self, dim: int, fpn_dims: list[int], context_dim: int):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, dim)
        self.lay2 = nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = nn.GroupNorm(8, inter_dims[1])
        self.lay3 = nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = nn.GroupNorm(8, inter_dims[2])
        self.lay4 = nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = nn.GroupNorm(8, inter_dims[3])
        self.lay5 = nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(8, inter_dims[4])
        self.out_lay = nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: list[Tensor]) -> Tensor:
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = f_nn.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = f_nn.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + f_nn.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = f_nn.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + f_nn.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = f_nn.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + f_nn.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = f_nn.relu(x)

        return self.out_lay(x)


class MHAttentionMap(nn.Module):
    """2D attention module that returns only the attention softmax (no multiplication by value)."""

    def __init__(
        self,
        query_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,  # noqa: FBT001, FBT002
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q: Tensor, k: Tensor, mask: Tensor | None = None) -> Tensor:
        q = self.q_linear(q)
        k = f_nn.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = f_nn.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


def dice_loss(inputs: Tensor, targets: Tensor, num_boxes: float) -> Tensor:
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs: Predictions (arbitrary shape, before sigmoid).
        targets: Binary classification labels (same shape as inputs after flatten).
        num_boxes: Number of boxes for normalization.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(
    inputs: Tensor, targets: Tensor, num_boxes: float, alpha: float = 0.25, gamma: float = 2
) -> Tensor:
    """Focal loss from RetinaNet (https://arxiv.org/abs/1708.02002).

    Args:
        inputs: Predictions (arbitrary shape).
        targets: Binary labels (same shape as inputs).
        num_boxes: Number of boxes for normalization.
        alpha: Weighting factor for positive vs negative examples.
        gamma: Exponent of the modulating factor.
    """
    prob = inputs.sigmoid()
    ce_loss = f_nn.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):
    """Post-process segmentation masks for instance segmentation."""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results: list[dict], outputs: dict, target_sizes: Tensor) -> list[dict]:
        """Add interpolated binary masks to detection results.

        Args:
            results: List of per-image detection dicts (from PostProcess).
            outputs: Model outputs containing ``pred_masks`` [B, Q, H_mask, W_mask].
            target_sizes: [B, 2] tensor with (H, W) for each image.
        """
        outputs_masks = outputs["pred_masks"]
        # All images same size in TerraTorch batches
        target_h, target_w = target_sizes[0].tolist()
        outputs_masks = f_nn.interpolate(
            outputs_masks, size=(int(target_h), int(target_w)), mode="bilinear", align_corners=False
        )
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, cur_mask in enumerate(outputs_masks):
            results[i]["masks"] = cur_mask.to(torch.uint8)

        return results
