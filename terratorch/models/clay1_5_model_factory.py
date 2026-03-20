import importlib
import logging
import sys
from collections.abc import Callable

import timm
import torch
from box import Box
from torch import nn

import terratorch.models.decoders as decoder_registry
from terratorch.models.backbones.clay_v1.embedder import Embedder
from terratorch.models.backbones.clay_v15.model import ClayMAE
from terratorch.models.model import (
    AuxiliaryHead,
    Model,
    ModelFactory,
)
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.scalar_output_model import ScalarOutputModel
from terratorch.models.utils import DecoderNotFoundError, extract_prefix_keys
from terratorch.registry import MODEL_FACTORY_REGISTRY

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification", "scalar_regression"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS


class DecoderNotFoundError(Exception):
    pass


class ModelWrapper(nn.Module):
    def __init__(self, batch_size, bands, platform, model: nn.Module = None) -> None:

        super().__init__()
        self.batch_size = batch_size
        self.bands = bands
        self.platform = platform
        self.model = model

        # self.embedding_shape = self.model.state_dict()['decoder.embed_to_pixels.dem.bias'].shape[0]

    def channels(self):
        return (1, self.embedding_shape)

    @property
    def parameters(self):
        return self.model.parameters

    def forward(self, args, **kwargs):
        datacube = {}
        datacube["pixels"] = args
        datacube["time"] = torch.zeros(self.batch_size, 4)
        datacube["platform"] = self.platform
        datacube["latlon"] = torch.zeros(self.batch_size, 4)
        datacube["waves"] = torch.zeros(4)
        datacube["gsd"] = torch.tensor(10)
        return self.model.forward(datacube, **kwargs)


@MODEL_FACTORY_REGISTRY.register
class Clay1_5ModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        in_channels: int,
        bands: list[int] = [],
        num_classes: int | None = None,
        pretrained: bool = True,  # noqa: FBT001, FBT002
        num_frames: int = 1,
        prepare_features_for_image_model: Callable | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001
        checkpoint_path: str = None,
        **kwargs,
    ) -> Model:
        # try:
        #     from claymodel.model import ClayMAE
        # except ImportError:
        #     message = "clay v1.5 not installed, please use pip install claimodel"
        #     logging.getLogger("terratorch").debug(message)
        #    raise Exception(message)
        batch_size = kwargs.get("batch_size")
        platform = kwargs.get("platform")
        kwargs["metadata"] = Box(kwargs["metadata"])
        return ModelWrapper(batch_size, bands, platform, ClayMAE(**kwargs))
