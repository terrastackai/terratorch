# Copyright contributors to the Terratorch project
import logging
from typing import List

import torch
import torch.nn.functional as F  # noqa: N812
from segmentation_models_pytorch.base import SegmentationModel
from torch import nn

from terratorch.models.heads import RegressionHead, SegmentationHead
from terratorch.models.model import AuxiliaryHeadWithDecoderWithoutInstantiatedHead, Model, ModelOutput
from terratorch.models.utils import center_crop, get_image_size, pad_images


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(False)


class PixelWiseModel(Model, SegmentationModel):
    """Model that encapsulates encoder and decoder and heads
    Expects decoder to have a "forward_features" method, an embed_dims property
    and optionally a "prepare_features_for_image_model" method.
    """

    def __init__(
        self,
        task: str,
        encoder: nn.Module,
        decoder: nn.Module,
        head_kwargs: dict,
        patch_size: int = None,
        padding: str = None,
        decoder_includes_head: bool = False,
        auxiliary_heads: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] | None = None,
        neck: nn.Module | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001
        image_size_out: tuple[int, int] | None = None,
    ) -> None:
        """Constructor

        Args:
            task (str): Task to be performed. One of segmentation or regression.
            encoder (nn.Module): Encoder to be used
            decoder (nn.Module): Decoder to be used
            head_kwargs (dict): Arguments to be passed at instantiation of the head.
            patch_size (int, optional): Patch size to be used during inference.
            padding (str, optional): Padding to be used during inference.
            decoder_includes_head (bool): Whether the decoder already incldes a head. If true, a head will not be added. Defaults to False.
            auxiliary_heads (list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] | None, optional): List of
                AuxiliaryHeads with heads to be instantiated. Defaults to None.
            neck (nn.Module | None): Module applied between backbone and decoder.
                Defaults to None, which applies the identity.
            rescale (bool, optional): Rescale the output of the model if it has a different size than the ground truth.
                Uses bilinear interpolation. Defaults to True.
            image_size_out (tuple[int, int] | None): The desired (Height, Width) size of the output image or mask. This is used to ensure the model produces the correct output shape.
                If set to **None** (default), the size is dynamically determined: either set by the 'image_size' property during the forward pass, or inferred directly from the size of the input image.
        """
        super().__init__()

        self.task = task
        self.encoder = encoder
        self.decoder = decoder
        self.head = (
            self._get_head(task, decoder.out_channels, head_kwargs) if not decoder_includes_head else nn.Identity()
        )

        if auxiliary_heads is not None:
            aux_heads = {}
            for aux_head_to_be_instantiated in auxiliary_heads:
                aux_head: nn.Module = (
                    self._get_head(task, aux_head_to_be_instantiated.decoder.out_channels, head_kwargs)
                    if not aux_head_to_be_instantiated.decoder_includes_head
                    else nn.Identity()
                )
                aux_head = nn.Sequential(aux_head_to_be_instantiated.decoder, aux_head)
                aux_heads[aux_head_to_be_instantiated.name] = aux_head
        else:
            aux_heads = {}
        self.aux_heads = nn.ModuleDict(aux_heads)

        if neck is not None:
            self.neck = neck
        elif hasattr(self.encoder, "prepare_features_for_image_model"):
            # only for backwards compatibility with pre-neck times.
            def model_defined_neck(x, **kwargs):
                return self.encoder.prepare_features_for_image_model(x)  # Drop kwargs

            self.neck = model_defined_neck
        else:
            self.neck = lambda x, image_size: x

        self.rescale = rescale
        self.image_size_out = image_size_out
        self.patch_size = patch_size
        self.padding = padding

    def freeze_encoder(self):
        if hasattr(self.encoder, "freeze"):
            self.encoder.freeze()
        else:
            freeze_module(self.encoder)

    def freeze_decoder(self):
        freeze_module(self.decoder)

    def freeze_head(self):
        freeze_module(self.head)

    # TODO: do this properly
    def check_input_shape(self, x: torch.Tensor) -> bool:  # noqa: ARG002
        return True

    @staticmethod
    def _check_for_single_channel_and_squeeze(x):
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        """Sequentially pass `x` through model`s encoder, decoder and heads"""

        image_size_out = self.image_size_out or kwargs.get("image_size", None) or get_image_size(x)

        if self.patch_size and self.padding is not None:
            x = pad_images(x, self.patch_size, self.padding)
        input_size = get_image_size(x)

        features = self.encoder(x, **kwargs)
        features = self.neck(features, image_size=input_size)

        decoder_output = self.decoder([f.clone() for f in features])
        mask = self.head(decoder_output)

        if self.rescale and mask.shape[-2:] != input_size:
            mask = F.interpolate(mask, size=input_size, mode="bilinear")

        mask = self._check_for_single_channel_and_squeeze(mask)
        mask = center_crop(mask, image_size_out)

        aux_outputs = {}
        for name, decoder in self.aux_heads.items():
            aux_output = decoder([f.clone() for f in features])
            if self.rescale and aux_output.shape[-2:] != input_size:
                aux_output = F.interpolate(aux_output, size=input_size, mode="bilinear")
            aux_output = self._check_for_single_channel_and_squeeze(aux_output)
            aux_output = center_crop(aux_output, image_size_out)
            aux_outputs[name] = aux_output

        return ModelOutput(output=mask, auxiliary_heads=aux_outputs)

    def _get_head(self, task: str, input_embed_dim: int, head_kwargs):
        if task == "segmentation":
            if "num_classes" not in head_kwargs:
                msg = "num_classes must be defined for segmentation task"
                raise Exception(msg)
            return SegmentationHead(input_embed_dim, **head_kwargs)
        if task == "regression":
            return RegressionHead(input_embed_dim, **head_kwargs)
        msg = "Task must be one of segmentation or regression."
        raise Exception(msg)
