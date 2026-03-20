# Copyright contributors to the Terratorch project

import importlib
import sys
from collections.abc import Callable

import numpy as np
import timm
import torch
from torch import nn

import terratorch.models.decoders as decoder_registry
from terratorch.datasets import HLSBands
from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
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


def check_the_kind_of_vit(name: str = None):
    if "mae" in name.lower() or name == "MaskedAutoencoderViT":
        return "vit-mae"
    else:
        return "vit"


def filter_cefficients_when_necessary(model_state_dict: dict = None, kind: str = None):

    # Head and backbone are not correctly separated in the original SatMAE source code
    if kind == "vit":
        ban_list = [
            "patch_embed",
            "decoder_blocks",
            "decoder_pred",
            "channel_embed",
            "mask_token",
            "decoder_embed",
            "pos_embed",
        ]  # ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']
    else:
        ban_list = list()

    try:
        for item in ban_list:
            model_state_dict["model"].pop(item)
    except Exception:
        print("No weight was removed.")

    return model_state_dict


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module = None, kind: str = None) -> None:

        super().__init__()

        self.model = model
        self.kind = kind
        self.embedding_shape = self.model.state_dict()["norm.bias"].shape[0]

        if self.kind == "vit":
            self.inner_forward = self.model.forward_features
        else:
            self.inner_forward = self.model.forward_encoder

        if hasattr(self.model, "num_patches"):
            self.num_patches = self.model.num_patches

        if self.kind == "vit":
            self.forward = self._forward_vit
        else:
            self.forward = self._forward_vit_mae

    def channels(self):
        return (1, self.embedding_shape)

    @property
    def parameters(self):
        return self.model.parameters

    def _forward_vit(self, x, **kwargs):

        x = self.inner_forward(x)

        return x

    def _forward_vit_mae(self, x, mask_ratio=0.75):

        x, _, ids_restore = self.inner_forward(x, mask_ratio)

        return x, ids_restore

    def summary(self):
        print(self)


@MODEL_FACTORY_REGISTRY.register
class SatMAEModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        in_channels: int,
        bands: list[HLSBands | int],
        num_classes: int | None = None,
        pretrained: bool = True,  # noqa: FBT001, FBT002
        num_frames: int = 1,
        prepare_features_for_image_model: Callable | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001
        checkpoint_path: str = None,
        **kwargs,
    ) -> Model:
        """Model factory for SatMAE  models.

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
                is different from the ground truth. Only applicable to pixel wise models (e.g. segmentation, pixel wise regression). Defaults to True.

        Raises:
            NotImplementedError: _description_
            DecoderNotFoundException: _description_

        Returns:
            nn.Module: _description_
        """

        self.possible_modules = None

        if not torch.cuda.is_available():
            self.CPU_ONLY = True
        else:
            self.CPU_ONLY = False

        # Path for accessing the model source code.
        self.syspath_kwarg = "model_sys_path"

        bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in bands]

        # TODO: support auxiliary heads
        if not isinstance(backbone, nn.Module):
            if "SatMAE" not in kwargs[self.syspath_kwarg]:
                msg = "This class only handles models for `SatMAE` encoders"
                raise NotImplementedError(msg)

            task = task.lower()
            if task not in SUPPORTED_TASKS:
                msg = f"Task {task} not supported. Please choose one of {SUPPORTED_TASKS}"
                raise NotImplementedError(msg)

            backbone_kwargs, kwargs = extract_prefix_keys(kwargs, "backbone_")
            backbone_name = backbone

            # Trying to find the model on HuggingFace.
            try:
                backbone: nn.Module = timm.create_model(
                    backbone,
                    pretrained=pretrained,
                    in_chans=in_channels,
                    num_frames=num_frames,
                    bands=bands,
                    features_only=True,
                    **backbone_kwargs,
                )
            except Exception:
                # When the model is not on HG, it needs be restored locally.
                print("This model is not available on HuggingFace. Trying to instantiate locally ...")

                assert checkpoint_path, "A checkpoint must be provided to restore the model."

                # The SatMAE source code must be installed or available via PYTHONPATH.
                try:
                    if self.syspath_kwarg in kwargs:
                        syspath_value = kwargs.get(self.syspath_kwarg)

                    else:
                        Exception(
                            f"It is necessary to define the variable {self.syspath_kwarg} on yaml"
                            "config for restoring local model."
                        )

                    sys.path.insert(0, syspath_value)

                    # There are dozens of classes in the SatMAE repo, but it seems to be the right open_generic_torch_model
                    backbone_template = None

                    self.possible_modules = [importlib.import_module(mod) for mod in ["models_mae", "models_vit"]]

                    for backbone_module in self.possible_modules:
                        backbone_template_ = getattr(backbone_module, backbone_name, None)
                        if not backbone_template_:
                            pass
                        else:
                            backbone_template = backbone_template_

                except ModuleNotFoundError:
                    print(f"It is better to review the field {self.syspath_kwarg} in the yaml file.")

                # Is it a ViT or a ViT-MAE ?
                backbone_kind = check_the_kind_of_vit(name=backbone_name)

                backbone: nn.Module = ModelWrapper(model=backbone_template(**backbone_kwargs), kind=backbone_kind)

                if self.CPU_ONLY:
                    model_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                else:
                    model_dict = torch.load(checkpoint_path, weights_only=True)

                # Filtering parameters from the model state_dict (when necessary)
                model_dict = filter_cefficients_when_necessary(model_state_dict=model_dict, kind=backbone_kind)

                if backbone_kind == "vit":
                    backbone.model.fc_norm = nn.Identity()
                    backbone.model.head_drop = nn.Identity()
                    backbone.model.head = nn.Identity()
                    backbone.model.pos_embed = None  # TODO It needs be corrected from source

                # Load saved model when it exists
                if pretrained:
                    backbone.model.load_state_dict(model_dict["model"], strict=False)

                # Print the general architecture
                backbone.summary()

                print("Model SatMAE was successfully restored.")

        # allow decoder to be a module passed directly
        decoder_cls = _get_decoder(decoder)

        decoder_kwargs, kwargs = extract_prefix_keys(kwargs, "decoder_")

        # If backabone is a ViT-MAE, the attribute "num_patches" will be necessary
        if hasattr(backbone, "num_patches"):
            decoder_kwargs["num_patches"] = backbone.num_patches

        # TODO: remove this
        if "SatMAEHead" in decoder:
            decoder: nn.Module = decoder_cls(**decoder_kwargs)
        else:
            decoder: nn.Module = decoder_cls(backbone.channels(), **decoder_kwargs)

        head_kwargs, kwargs = extract_prefix_keys(kwargs, "head_")
        if num_classes:
            head_kwargs["num_classes"] = num_classes
        if aux_decoders is None:
            return _build_appropriate_model(
                task, backbone, decoder, head_kwargs, prepare_features_for_image_model, rescale=rescale
            )

        to_be_aux_decoders: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] = []
        for aux_decoder in aux_decoders:
            args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
            aux_decoder_cls: nn.Module = _get_decoder(aux_decoder.decoder)
            aux_decoder_kwargs, kwargs = extract_prefix_keys(args, "decoder_")
            aux_decoder_instance = aux_decoder_cls(backbone.feature_info.channels(), **aux_decoder_kwargs)
            # aux_decoder_instance = aux_decoder_cls([128, 256, 512, 1024], **decoder_kwargs)

            aux_head_kwargs, kwargs = extract_prefix_keys(args, "head_")
            if num_classes:
                aux_head_kwargs["num_classes"] = num_classes
            # aux_head: nn.Module = _get_head(task, aux_decoder_instance, num_classes=num_classes, **head_kwargs)
            # aux_decoder.decoder = nn.Sequential(aux_decoder_instance, aux_head)
            to_be_aux_decoders.append(
                AuxiliaryHeadWithDecoderWithoutInstantiatedHead(aux_decoder.name, aux_decoder_instance, aux_head_kwargs)
            )

        return _build_appropriate_model(
            task,
            backbone,
            decoder,
            head_kwargs,
            prepare_features_for_image_model,
            rescale=rescale,
            auxiliary_heads=to_be_aux_decoders,
        )


def _build_appropriate_model(
    task: str,
    backbone: nn.Module,
    decoder: nn.Module,
    head_kwargs: dict,
    prepare_features_for_image_model: Callable,
    rescale: bool = True,  # noqa: FBT001, FBT002
    auxiliary_heads: dict | None = None,
):
    if task in PIXEL_WISE_TASKS:
        return PixelWiseModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            prepare_features_for_image_model=prepare_features_for_image_model,
            rescale=rescale,
            auxiliary_heads=auxiliary_heads,
        )
    elif task in SCALAR_TASKS:
        return ScalarOutputModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            prepare_features_for_image_model=prepare_features_for_image_model,
            auxiliary_heads=auxiliary_heads,
        )


def _get_decoder(decoder: str | nn.Module) -> nn.Module:
    if isinstance(decoder, nn.Module):
        return decoder
    if isinstance(decoder, str):
        try:
            decoder = getattr(decoder_registry, decoder)
            return decoder
        except AttributeError as decoder_not_found_exception:
            msg = f"Decoder {decoder} was not found in the registry."
            raise DecoderNotFoundError(msg) from decoder_not_found_exception
    msg = "Decoder must be str or nn.Module"
    raise Exception(msg)
