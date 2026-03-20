import warnings
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

try:
    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
    from peft.mapping_func import get_peft_model

    _has_peft = True
except ModuleNotFoundError:
    _has_peft = False

TESTED_PEFT_METHODS = ["LORA"]


def _get_submodules(model: nn.Module, key: str) -> tuple[nn.Module, nn.Module, str]:
    # adapted from PEFT
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.rsplit(".", maxsplit=1)[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


@dataclass(frozen=True)
class TerratorchPEFTConfig:
    method: str
    replace_qkv: str | None
    peft_config_kwargs: dict[str, Any]


def _validate_terratorch_peft_config(peft_config: dict[str, Any]) -> TerratorchPEFTConfig:
    terratorch_peft_config = TerratorchPEFTConfig(
        method=peft_config["method"],
        replace_qkv=peft_config.get("replace_qkv", None),
        peft_config_kwargs=peft_config.get("peft_config_kwargs", {}),
    )
    if terratorch_peft_config.method not in TESTED_PEFT_METHODS:
        msg = f"PEFT method {terratorch_peft_config.method} has not been tested. Use at your own risk."
        warnings.warn(msg, stacklevel=1)
    return terratorch_peft_config


def get_peft_backbone(peft_config: dict[str, Any], backbone: nn.Module) -> nn.Module:
    terratorch_peft_config = _validate_terratorch_peft_config(peft_config)
    if not _has_peft:
        msg = "You need to install terratorch with peft dependency to use peft_config. Use pip install terratorch[peft]"
        raise ImportError(msg)
    peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[terratorch_peft_config.method]
    peft_config_peft = peft_config_cls(**terratorch_peft_config.peft_config_kwargs)
    if terratorch_peft_config.replace_qkv is not None:
        replace_qkv(backbone, terratorch_peft_config.replace_qkv)  # modifies inplace
    backbone = get_peft_model(backbone, peft_config_peft)
    return backbone


class QKVSep(nn.Module):
    def __init__(self, original_qkv: nn.Linear):
        super().__init__()
        if original_qkv.out_features != original_qkv.in_features * 3:
            msg = "The output features must be 3 times the input features for Q, K, V separation"
            raise ValueError(msg)

        self.in_features = original_qkv.in_features
        self.out_features = original_qkv.out_features

        # Create nn.Linear layers for Q, K, V using slices of the original weights and biases
        self.q_linear = nn.Linear(self.in_features, self.in_features)
        self.k_linear = nn.Linear(self.in_features, self.in_features)
        self.v_linear = nn.Linear(self.in_features, self.in_features)

        # Assign weights and biases from the original layer
        with torch.no_grad():
            self.q_linear.weight = nn.Parameter(original_qkv.weight[: self.in_features, :])
            self.k_linear.weight = nn.Parameter(original_qkv.weight[self.in_features : 2 * self.in_features, :])
            self.v_linear.weight = nn.Parameter(original_qkv.weight[2 * self.in_features :, :])

            if original_qkv.bias is not None:
                self.q_linear.bias = nn.Parameter(original_qkv.bias[: self.in_features])
                self.k_linear.bias = nn.Parameter(original_qkv.bias[self.in_features : 2 * self.in_features])
                self.v_linear.bias = nn.Parameter(original_qkv.bias[2 * self.in_features :])
            else:
                self.q_linear.bias = None
                self.k_linear.bias = None
                self.v_linear.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        return torch.cat((q, k, v), dim=-1)


def replace_qkv(model: nn.Module, qkv_suffix: str):
    # This is needed for ViTEncoderDecoder because the qkv matrices are together,
    # and it would not work with LoRA (and probably other adapters)
    replaced = False
    for key, _ in model.named_modules():
        if key.endswith(f".{qkv_suffix}"):
            replaced = True
            parent, target, target_name = _get_submodules(model, key)
            if not isinstance(target, nn.Linear):
                msg = "Only a qkv nn.Linear can be replaced."
                raise ValueError(msg)
            new_module = QKVSep(target)
            setattr(parent, target_name, new_module)
    if not replaced:
        warnings.warn("replace_qkv was not None but no module was found ending with that pattern.", stacklevel=1)
