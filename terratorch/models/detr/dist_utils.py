# Copyright contributors to the Terratorch project

"""Distributed helpers for DETR loss normalization."""

import torch


def is_dist_avail_and_initialized() -> bool:
    """Check if torch.distributed is available and initialized."""
    if not torch.distributed.is_available():
        return False
    return torch.distributed.is_initialized()


def get_world_size() -> int:
    """Return the world size, or 1 if not in a distributed setting."""
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()
