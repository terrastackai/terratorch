# Copyright contributors to the Terratorch project

import logging
import warnings
from typing import List

import torch
from torch import nn

from terratorch.models.embedding_output_model import EmbeddingOutputModel
from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
)
from terratorch.models.necks import Neck, NeckSequential, build_neck_list
from terratorch.models.peft_utils import get_peft_backbone
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.scalar_output_model import ScalarOutputModel
from terratorch.models.utils import TemporalWrapper, extract_prefix_keys
from terratorch.registry import BACKBONE_REGISTRY, DECODER_REGISTRY, MODEL_FACTORY_REGISTRY

from .utils import _get_backbone

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification", "scalar_regression"]
EMBEDDING_GENERATION = ["embedding_generation"]

SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS + EMBEDDING_GENERATION


def _get_decoder_and_head_kwargs(
    decoder: str | nn.Module,
    channel_list: list[int],
    decoder_kwargs: dict,
    head_kwargs: dict,
    num_outputs: int | None = None,
    num_classes: int | None = None,
) -> tuple[nn.Module, dict, bool]:

    if num_outputs is not None and num_classes is not None:
        raise ValueError("Only one of `num_outputs` or `num_classes` should be provided.")

    # if its already an nn Module, check if it includes a head. if it doesnt, pass num classes/num outputs to head kwargs
    if isinstance(decoder, nn.Module):
        includes_head = getattr(decoder, "includes_head", False)

        if includes_head and head_kwargs:
            msg = "Decoder already includes a head, but `head_` arguments were specified. These should be removed."
            raise ValueError(msg)

        if not includes_head:
            if num_outputs is not None:
                head_kwargs["num_outputs"] = num_outputs
            elif num_classes is not None:
                head_kwargs["num_classes"] = num_classes

        return decoder, head_kwargs, False

    # if its not an nn module, check if the class includes a head
    # depending on that, pass num classes/num outputs to either head kwrags or decoder
    if hasattr(DECODER_REGISTRY.find_class(decoder), "includes_head"):
        includes_head = DECODER_REGISTRY.find_registry(decoder).includes_head

    else:
        includes_head = False
        msg = (
            f"Decoder {decoder} does not have an `includes_head` attribute. Falling back to the value of the registry."
        )
        logging.debug(msg)

    key = "num_outputs" if num_outputs is not None else "num_classes"
    num_outputs = num_outputs or num_classes
    if num_outputs is not None:
        if includes_head:
            decoder_kwargs["num_classes"] = num_outputs
            if head_kwargs:
                msg = "Decoder already includes a head, but `head_` arguments were specified. These should be removed."
                raise ValueError(msg)
        else:
            head_kwargs[key] = num_outputs

    return DECODER_REGISTRY.build(decoder, channel_list, **decoder_kwargs), head_kwargs, includes_head


def _check_all_args_used(kwargs):
    if kwargs:
        msg = f"arguments {kwargs} were passed but not used."
        raise ValueError(msg)


def _get_argument_from_instance(model, name):
    return getattr(model._timm_module.patch_embed, name)[-1]


@MODEL_FACTORY_REGISTRY.register
class EncoderDecoderFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module | None = None,
        backbone_kwargs: dict | None = None,
        decoder_kwargs: dict | None = None,
        head_kwargs: dict | None = None,
        num_classes: int | None = None,
        num_outputs: int | None = None,
        necks: list[dict] | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001,
        image_size_out: tuple[int, int] | None = None,
        peft_config: dict | None = None,
        **kwargs,
    ) -> Model:
        """Generic model factory that combines an encoder and decoder, together with a head, for a specific task.

        Further arguments to be passed to the backbone, decoder or head. They should be prefixed with
        `backbone_`, `decoder_` and `head_` respectively.

        Args:
            task (str): Task to be performed. Currently supports "segmentation", "regression" and "classification".
            backbone (str, nn.Module): Backbone to be used. If a string, will look for such models in the different
                registries supported (internal terratorch registry, timm, ...). If a torch nn.Module, will use it
                directly. The backbone should have and `out_channels` attribute and its `forward` should return a list[Tensor].
            decoder (Union[str, nn.Module], optional): Decoder to be used for the segmentation model.
                    If a string, will look for such decoders in the different
                    registries supported (internal terratorch registry, smp, ...).
                    If an nn.Module, we expect it to expose a property `decoder.out_channels`.
                    Pixel wise tasks will be concatenated with a Conv2d for the final convolution.
                    Defaults to "FCNDecoder". Defaults to 'None' for embedding generation tasks.
            backbone_kwargs (dict, optional) : Arguments to be passed to instantiate the backbone.
            decoder_kwargs (dict, optional) : Arguments to be passed to instantiate the decoder.
            head_kwargs (dict, optional) : Arguments to be passed to the head network.
            num_classes (int, optional): Number of classes for segmentation and classification tasks.
            num_outputs (int, optional):  Number of variables to predict if task is regression.
            necks (list[dict]): nn.Modules to be called in succession on encoder features
                before passing them to the decoder. Should be registered in the NECKS_REGISTRY registry.
                Expects each one to have a key "name" and subsequent keys for arguments, if any.
                Defaults to None, which applies the identity function.
            aux_decoders (list[AuxiliaryHead] | None): List of AuxiliaryHead decoders to be added to the model.
                These decoders take the input from the encoder as well.
            rescale (bool): Whether to apply bilinear interpolation to rescale the model output if its size
                is different from the ground truth. Only applicable to pixel wise models
                (e.g. segmentation, pixel wise regression). Defaults to True.
            image_size_out (tuple[int, int] | None): The desired (Height, Width) size of the output image or mask for pixelwise tasks (e.g. segmentation, pixel wise regression).
                This is used to ensure the model produces the correct output shape. If set to **None** (default), the size is dynamically determined: either
                set by the 'image_size' property during the forward pass, or inferred directly from the size of the input image.
            peft_config (dict): Configuration options for using [PEFT](https://huggingface.co/docs/peft/index).
                The dictionary should have the following keys:

                - "method": Which PEFT method to use. Should be one implemented in PEFT, a list is available [here](https://huggingface.co/docs/peft/package_reference/peft_types#peft.PeftType).
                - "replace_qkv": String containing a substring of the name of the submodules to replace with QKVSep.
                  This should be used when the qkv matrices are merged together in a single linear layer and the PEFT
                  method should be applied separately to query, key and value matrices (e.g. if LoRA is only desired in
                  Q and V matrices). e.g. If using Prithvi this should be "qkv"
                - "peft_config_kwargs": Dictionary containing keyword arguments which will be passed to [PeftConfig](https://huggingface.co/docs/peft/package_reference/config#peft.PeftConfig)


        Returns:
            nn.Module: Full model with encoder, decoder and head.
        """
        task = task.lower()
        if task not in SUPPORTED_TASKS:
            msg = f"Task {task} not supported. Please choose one of {SUPPORTED_TASKS}"
            raise NotImplementedError(msg)

        if not backbone_kwargs:
            backbone_kwargs, kwargs = extract_prefix_keys(kwargs, "backbone_")

        backbone = _get_backbone(backbone, **backbone_kwargs)

        # If patch size is not provided in the config or by the model, it might lead to errors due to irregular images.
        patch_size = backbone_kwargs.get("patch_size", None)

        if patch_size is None:
            # Infer patch size from model by checking all backbone modules
            for module in backbone.modules():
                if hasattr(module, "patch_size"):
                    patch_size = module.patch_size
                    break
        padding = backbone_kwargs.get("padding", "reflect")

        if peft_config is not None:
            if not backbone_kwargs.get("pretrained", False):
                msg = (
                    "You are using PEFT without a pretrained backbone. If you are loading a checkpoint afterwards "
                    "this is probably fine, but if you are training a model check the backbone_pretrained parameter."
                )
                warnings.warn(msg, stacklevel=1)

            backbone = get_peft_backbone(peft_config, backbone)

        try:
            out_channels = backbone.out_channels
        except AttributeError as e:
            msg = "backbone must have out_channels attribute"
            raise AttributeError(msg) from e

        if necks is None:
            necks = []
        neck_list, channel_list = build_neck_list(necks, out_channels)

        # some decoders already include a head
        # for these, we pass the num_outputs to them
        # others dont include a head
        # for those, we dont pass num_outputs
        if decoder:
            if not decoder_kwargs:
                decoder_kwargs, kwargs = extract_prefix_keys(kwargs, "decoder_")

            if not head_kwargs:
                head_kwargs, kwargs = extract_prefix_keys(kwargs, "head_")

            decoder, head_kwargs, decoder_includes_head = _get_decoder_and_head_kwargs(
                decoder, channel_list, decoder_kwargs, head_kwargs, num_outputs=num_outputs, num_classes=num_classes
            )

            if aux_decoders is None:
                _check_all_args_used(kwargs)
                return _build_appropriate_model(
                    task,
                    backbone,
                    decoder,
                    head_kwargs,
                    patch_size=patch_size,
                    padding=padding,
                    necks=neck_list,
                    decoder_includes_head=decoder_includes_head,
                    rescale=rescale,
                    image_size_out=image_size_out,
                )

            to_be_aux_decoders: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] = []
            for aux_decoder in aux_decoders:
                args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
                aux_decoder_kwargs, args = extract_prefix_keys(args, "decoder_")
                aux_head_kwargs, args = extract_prefix_keys(args, "head_")
                aux_decoder_instance, aux_head_kwargs, aux_decoder_includes_head = _get_decoder_and_head_kwargs(
                    aux_decoder.decoder,
                    channel_list,
                    aux_decoder_kwargs,
                    aux_head_kwargs,
                    num_outputs=num_outputs,
                    num_classes=num_classes,
                )
                to_be_aux_decoders.append(
                    AuxiliaryHeadWithDecoderWithoutInstantiatedHead(
                        aux_decoder.name, aux_decoder_instance, aux_head_kwargs
                    )
                )
                _check_all_args_used(args)

            _check_all_args_used(kwargs)

            return _build_appropriate_model(
                task,
                backbone,
                decoder,
                head_kwargs,
                patch_size=patch_size,
                padding=padding,
                necks=neck_list,
                decoder_includes_head=decoder_includes_head,
                rescale=rescale,
                image_size_out=image_size_out,
                auxiliary_heads=to_be_aux_decoders,
            )
        else:
            if task not in EMBEDDING_GENERATION and decoder is None:
                raise ValueError(f"A decoder must be provided for task '{task}'.")

            return _build_appropriate_model(
                task,
                backbone,
                decoder,
                head_kwargs={},
                patch_size=patch_size,
                padding=padding,
                necks=neck_list,
            )


def _build_appropriate_model(
    task: str,
    backbone: nn.Module,
    decoder: nn.Module | None,
    head_kwargs: dict,
    patch_size: int | list | None,
    padding: str,
    decoder_includes_head: bool = False,
    necks: list[Neck] | None = None,
    rescale: bool = True,  # noqa: FBT001, FBT002
    image_size_out: tuple[int, int] | None = None,
    auxiliary_heads: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] | None = None,
):
    if necks:
        neck_module: nn.Module = NeckSequential(*necks)
    else:
        neck_module = None
    if task in PIXEL_WISE_TASKS:
        return PixelWiseModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            patch_size=patch_size,
            padding=padding,
            decoder_includes_head=decoder_includes_head,
            neck=neck_module,
            rescale=rescale,
            image_size_out=image_size_out,
            auxiliary_heads=auxiliary_heads,
        )
    elif task in SCALAR_TASKS:
        return ScalarOutputModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            patch_size=patch_size,
            padding=padding,
            decoder_includes_head=decoder_includes_head,
            neck=neck_module,
            auxiliary_heads=auxiliary_heads,
        )

    elif task in EMBEDDING_GENERATION:
        return EmbeddingOutputModel(
            backbone,
            patch_size=patch_size,
            padding=padding,
            neck=neck_module,
        )
