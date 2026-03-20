# Copyright contributors to the Terratorch project

import warnings
from collections.abc import Callable

from torch import nn

from terratorch.datasets import HLSBands
from terratorch.models import EncoderDecoderFactory
from terratorch.models.model import (
    AuxiliaryHead,
    Model,
    ModelFactory,
)
from terratorch.registry import MODEL_FACTORY_REGISTRY

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification", "scalar_regression"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS


@MODEL_FACTORY_REGISTRY.register
class PrithviModelFactory(ModelFactory):
    def __init__(self) -> None:
        self._factory: EncoderDecoderFactory = EncoderDecoderFactory()

    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        bands: list[HLSBands | int],
        in_channels: int
        | None = None,  # this should be removed, can be derived from bands. But it is a breaking change
        num_classes: int | None = None,
        pretrained: bool = True,  # noqa: FBT001, FBT002
        num_frames: int = 1,
        prepare_features_for_image_model: Callable | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001
        **kwargs,
    ) -> Model:
        """Model factory for prithvi models.

        Further arguments to be passed to the backbone, decoder or head. They should be prefixed with
        `backbone_`, `decoder_` and `head_` respectively.

        Args:
            task (str): Task to be performed. Currently supports "segmentation" and "regression".
            backbone (str, nn.Module): Backbone to be used. If string, should be able to be parsed
                by the specified factory. Defaults to "prithvi_100".
            decoder (Union[str, nn.Module], optional): Decoder to be used for the segmentation model.
                    If a string, it will be created from a class exposed in decoder.__init__.py with the same name.
                    If an nn.Module, we expect it to expose a property `decoder.out_channels`.
                    Will be concatenated with a Conv2d for the final convolution. Defaults to "FCNDecoder".
            in_channels (int, optional): Number of input channels. Defaults to 3.
            bands (list[terratorch.datasets.HLSBands], optional): Bands the model will be trained on.
                    Should be a list of terratorch.datasets.HLSBands.
                    Defaults to [HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE].
            num_classes (int, optional): Number of classes. None for regression tasks.
            pretrained (Union[bool, Path], optional): Whether to load pretrained weights for the backbone, if available.
                Defaults to True.
            num_frames (int, optional): Number of timesteps for the model to handle. Defaults to 1.
            prepare_features_for_image_model (Callable | None): Function to be called on encoder features
                before passing them to the decoder. Defaults to None, which applies the identity function.
            aux_decoders (list[AuxiliaryHead] | None): List of AuxiliaryHead deciders to be added to the model.
                These decoders take the input from the encoder as well.
            rescale (bool): Whether to apply bilinear interpolation to rescale the model output if its size
                is different from the ground truth. Only applicable to pixel wise models
                (e.g. segmentation, pixel wise regression). Defaults to True.


        Returns:
            nn.Module: Full model with encoder, decoder and head.
        """
        warnings.warn("PrithviModelFactory is deprecated. Please switch to EncoderDecoderFactory.", stacklevel=1)
        if in_channels is None:
            in_channels = len(bands)
        # TODO: support auxiliary heads
        kwargs["backbone_bands"] = bands
        kwargs["backbone_in_chans"] = in_channels
        kwargs["backbone_pretrained"] = pretrained
        kwargs["backbone_num_frames"] = num_frames
        if prepare_features_for_image_model:
            msg = "This functionality is no longer supported. Please migrate to EncoderDecoderFactory\
                         and use necks."
            raise RuntimeError(msg)

        if not isinstance(backbone, nn.Module):
            if not backbone.startswith("prithvi_"):
                msg = "This class only handles models for `prithvi` encoders"
                raise NotImplementedError(msg)

        return self._factory.build_model(
            task,
            backbone,
            decoder,
            num_classes=num_classes,
            necks=None,
            aux_decoders=aux_decoders,
            rescale=rescale,
            **kwargs,
        )
