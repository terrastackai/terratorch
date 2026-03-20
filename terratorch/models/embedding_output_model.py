# Copyright contributors to the Terratorch project
import logging
from typing import List

import torch
import torch.nn.functional as F  # noqa: N812
from segmentation_models_pytorch.base import SegmentationModel
from torch import nn

from terratorch.models.model import Model
from terratorch.models.utils import pad_images


class EmbeddingOutputModel(Model, SegmentationModel):
    """Model that wraps an encoder and potential neck
    intended for embedding generation workflows.
    """

    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int = None,
        padding: str = None,
        neck: nn.Module | None = None,
    ) -> None:
        """Constructor

        Args:
            encoder (nn.Module): Encoder to be used
            patch_size (int, optional): Patch size to be used during inference.
            padding (str, optional): Padding to be used during inference.
            neck (nn.Module | None): Neck module applied after backbone. Defaults to None, which applies the identity.
        """
        super().__init__()

        self.encoder = encoder

        if neck is not None:
            self.neck = neck

        else:
            self.neck = lambda x, image_size: x

        self.patch_size = patch_size
        self.padding = padding
        self.freeze_encoder()

    def freeze_encoder(self):
        if hasattr(self.encoder, "freeze"):
            self.encoder.freeze()
        else:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

    def freeze_decoder(self):
        return []

    def forward(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        """Sequentially pass `x` through model`s encoder and neck"""

        if self.patch_size and self.padding is not None:
            x = pad_images(x, self.patch_size, self.padding)

        features = self.encoder(x, **kwargs)
        features = self.neck(features)

        return features
