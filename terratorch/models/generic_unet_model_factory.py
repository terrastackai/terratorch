# Copyright contributors to the Terratorch project

"""
This is just an example of a possible structure to include SMP models
Right now it always returns a UNET, but could easily be extended to many of the models provided by SMP.
"""

import importlib

import torch
from torch import nn

from terratorch.models.model import Model, ModelFactory, ModelOutput
from terratorch.models.utils import extract_prefix_keys
from terratorch.registry import MODEL_FACTORY_REGISTRY


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(False)


@MODEL_FACTORY_REGISTRY.register
class GenericUnetModelFactory(ModelFactory):
    def _check_model_availability(self, model, builtin_engine, engine, **model_kwargs):

        try:
            print(f"Using module {model} from terratorch.")
            if builtin_engine:
                model_class = getattr(builtin_engine, model)
            else:
                model_class = None
        except:
            if _has_mmseg:
                print("Module not available on terratorch.")
                print(f"Using module {model} from mmseg.")
                if engine:
                    model_class = getattr(engine, model)
                else:
                    model_class = None
            else:
                raise Exception("mmseg is not installed.")

        if model_class:
            model = model_class(
                **model_kwargs,
            )
        else:
            model = None

        return model

    def build_model(
        self,
        task: str = "segmentation",
        backbone: str | None = None,
        decoder: str | None = None,
        dilations: tuple[int] = (1, 6, 12, 18),
        in_channels: int = 6,
        pretrained: str | bool | None = True,
        regression_relu: bool = False,
        **kwargs,
    ) -> Model:
        """Factory to create model based on mmseg.

        Args:
            task (str): Must be "segmentation".
            model (str): Decoder architecture. Currently only supports "unet".
            in_channels (int): Number of input channels.
            pretrained(str | bool): Which weights to use for the backbone. If true, will use "imagenet". If false or None, random weights. Defaults to True.
            regression_relu (bool). Whether to apply a ReLU if task is regression. Defaults to False.

        Returns:
            Model: UNet model.
        """
        if task not in ["segmentation", "regression"]:
            msg = f"This model can only perform pixel wise tasks, but got task {task}"
            raise Exception(msg)

        builtin_engine_decoders = importlib.import_module("terratorch.models.decoders")
        builtin_engine_encoders = importlib.import_module("terratorch.models.backbones")

        # Default values
        backbone_builtin_engine = None
        decoder_builtin_engine = None
        backbone_engine = None
        decoder_engine = None
        backbone_model_kwargs = {}
        decoder_model_kwargs = {}

        try:
            engine_decoders = importlib.import_module("mmseg.models.decode_heads")
            engine_encoders = importlib.import_module("mmseg.models.backbones")
            _has_mmseg = True
        except:
            engine_decoders = None
            engine_encoders = None
            _has_mmseg = False
            print("mmseg is not installed.")

        if backbone:
            backbone_kwargs = _extract_prefix_keys(kwargs, "backbone_")
            backbone_model_kwargs = backbone_kwargs
            backbone_engine = engine_encoders
            backbone_builtin_engine = builtin_engine_encoders
        else:
            backbone = None

        if decoder:
            decoder_kwargs = _extract_prefix_keys(kwargs, "decoder_")
            decoder_model_kwargs = decoder_kwargs
            decoder_engine = engine_decoders
            decoder_builtin_engine = builtin_engine_decoders
        else:
            decoder = None

        if not backbone and not decoder:
            print("It is necessary to define a backbone and/or a decoder.")

        # Instantianting backbone and decoder
        backbone = self._check_model_availability(
            backbone, backbone_builtin_engine, backbone_engine, **backbone_model_kwargs
        )
        decoder = self._check_model_availability(
            decoder, decoder_builtin_engine, decoder_engine, **decoder_model_kwargs
        )

        return GenericUnetModelWrapper(
            backbone,
            decoder=decoder,
            relu=task == "regression" and regression_relu,
            squeeze_single_class=task == "regression",
        )


class GenericUnetModelWrapper(Model, nn.Module):
    def __init__(
        self, unet_model, decoder=None, relu=False, squeeze_single_class=False, intermediary_outputs_in_decoder=False
    ) -> None:
        super().__init__()
        self.unet_model = unet_model
        self.decoder = decoder
        self.final_act = nn.ReLU() if relu else nn.Identity()
        self.squeeze_single_class = squeeze_single_class
        self.intermediary_outputs_in_decoder = intermediary_outputs_in_decoder

        if decoder:
            self.decode = self._with_decoder
        else:
            self.decode = self._no_decoder

        if self.intermediary_outputs_in_decoder:
            self.catch_unet_outputs = lambda x: x
        else:
            self.catch_unet_outputs = lambda x: x[-1]

    def _no_decoder(self, x):
        return x

    def _with_decoder(self, x):

        return self.decoder(x)

    def forward(self, *args, **kwargs):

        # It supposes the input has dimension (B, C, H, W)
        input_data = [args[0]]  # It adapts the input to became a list of time 'snapshots'
        args = input_data

        unet_output = self.unet_model(*args, **kwargs)
        unet_output = self.final_act(unet_output)

        unet_output = self.catch_unet_outputs(unet_output)
        # Decoding is optional
        unet_output_decoded = self.decode(unet_output)

        if unet_output_decoded.shape[1] == 1 and self.squeeze_single_class:
            unet_output_decoded = unet_output_decoded.squeeze(1)

        model_output = ModelOutput(unet_output_decoded)

        return model_output

    def freeze_encoder(self):
        raise freeze_module(self.unet_model)

    def freeze_decoder(self):
        raise freeze_module(self.decoder)


def _extract_prefix_keys(d: dict, prefix: str) -> dict:
    extracted_dict = {}
    keys_to_del = []
    for k, v in d.items():
        if k.startswith(prefix):
            extracted_dict[k.split(prefix)[1]] = v
            keys_to_del.append(k)

    for k in keys_to_del:
        del d[k]

    return extracted_dict
