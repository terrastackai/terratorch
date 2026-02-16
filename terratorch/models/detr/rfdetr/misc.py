# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Adapted for TerraTorch: stripped to only accuracy() and NestedTensor.
# Distributed helpers reused from terratorch.models.detr.dist_utils.

from __future__ import annotations

import torch
from torch import Tensor


class NestedTensor:
    """Utility class for variable-size images with padding masks.

    Kept for reference parity tests only.
    """

    def __init__(self, tensors: Tensor, mask: Tensor | None) -> None:
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device) -> NestedTensor:
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self) -> tuple[Tensor, Tensor | None]:
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    """Computes the precision@k for the specified values of k."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
