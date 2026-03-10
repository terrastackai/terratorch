# Copyright contributors to the Terratorch project
import gc
import os
import sys
import warnings

import pytest
import timm
import torch

from terratorch.cli_tools import build_lightning_cli
from terratorch.models.backbones import scalemae, torchgeo_vit
from terratorch.registry import BACKBONE_REGISTRY

NUM_CHANNELS = 6
NUM_FRAMES = 4


@pytest.fixture
def input_224():
    return torch.ones((1, NUM_CHANNELS, 224, 224))


def test_custom_module(input_224):
    sys.path.append("examples/utils/custom_modules")

    from alexnet import alexnet_encoder

    model = BACKBONE_REGISTRY.build("alexnet_encoder", num_channels=6)
    output = model(input_224)


@pytest.mark.parametrize("case", ["fit", "test", "validate"])
def test_custom_module_yaml(case):
    command_list = [case, "-c", "examples/utils/alexnet_custom_model_config.yaml"]
    _ = build_lightning_cli(command_list)

    gc.collect()


@pytest.mark.parametrize("case", ["fit", "test", "validate"])
def test_custom_module_env_var(case):
    """Test that the TERRATORCH_CUSTOM_MODULE_PATH environment variable can be used to specify the path to a custom module."""
    command_list = [
        case,
        "-c",
        "examples/utils/alexnet_custom_model_config_env_vars.yaml",
    ]  # This config does not contain the 'custom_module_path'.
    os.environ["TERRATORCH_CUSTOM_MODULE_PATH"] = "examples/utils/custom_modules"
    _ = build_lightning_cli(command_list)

    gc.collect()
