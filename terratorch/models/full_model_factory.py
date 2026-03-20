# Copyright contributors to the Terratorch project

import warnings

from torch import nn

from terratorch.models.model import ModelFactory
from terratorch.models.peft_utils import get_peft_backbone
from terratorch.registry import FULL_MODEL_REGISTRY, MODEL_FACTORY_REGISTRY


def _get_model(model: str | nn.Module, **model_kwargs) -> nn.Module:
    if isinstance(model, nn.Module):
        return model
    return FULL_MODEL_REGISTRY.build(model, **model_kwargs)


def _check_all_args_used(kwargs):
    if kwargs:
        msg = f"arguments {kwargs} were passed but not used."
        raise ValueError(msg)


@MODEL_FACTORY_REGISTRY.register
class FullModelFactory(ModelFactory):
    def build_model(
        self,
        model: str | nn.Module,
        rescale: bool = True,  # noqa: FBT002, FBT001
        padding: str = "reflect",
        peft_config: dict | None = None,
        **kwargs,
    ) -> nn.Module:
        """Generic model factory that wraps any model.

        All kwargs are passed to the model.

        Args:
            task (str): Task to be performed. Currently supports "segmentation" and "regression".
            model (str, nn.Module): Model to be used. If a string, will look for such models in the different
                registries supported (internal terratorch registry, ...). If a torch nn.Module, will use it
                directly.
            rescale (bool): Whether to apply bilinear interpolation to rescale the model output if its size
                is different from the ground truth. Only applicable to pixel wise models
                (e.g. segmentation, pixel wise regression, reconstruction). Defaults to True.
            padding (str): Padding method used if images are not divisible by the patch size. Defaults to "reflect".
            peft_config (dict): Configuration options for using [PEFT](https://huggingface.co/docs/peft/index).
                The dictionary should have the following keys:
                - "method": Which PEFT method to use. Should be one implemented in PEFT, a list is available [here](https://huggingface.co/docs/peft/package_reference/peft_types#peft.PeftType).
                - "replace_qkv": String containing a substring of the name of the submodules to replace with QKVSep.
                  This should be used when the qkv matrices are merged together in a single linear layer and the PEFT
                  method should be applied separately to query, key and value matrices (e.g. if LoRA is only desired in
                  Q and V matrices). e.g. If using Prithvi this should be "qkv"
                - "peft_config_kwargs": Dictionary containing keyword arguments which will be passed to [PeftConfig](https://huggingface.co/docs/peft/package_reference/config#peft.PeftConfig)


        Returns:
            nn.Module: Full model.
        """

        model = _get_model(model, **kwargs)

        # If patch size is not provided in the config or by the model, it might lead to errors due to irregular images.
        patch_size = kwargs.get("patch_size", None)

        if patch_size is None:
            # Infer patch size from model by checking all backbone modules
            for module in model.modules():
                if hasattr(module, "patch_size"):
                    patch_size = module.patch_size
                    break

        if peft_config is not None:
            if not kwargs.get("pretrained", False):
                msg = (
                    "You are using PEFT without a pretrained backbone. If you are loading a checkpoint afterwards "
                    "this is probably fine, but if you are training a model check the backbone_pretrained parameter."
                )
                warnings.warn(msg, stacklevel=1)

            model = get_peft_backbone(peft_config, model)

        return model
