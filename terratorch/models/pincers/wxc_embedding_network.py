import numpy as np
import torch
from granitewxc.decoders.downscaling import ConvEncoderDecoder
from granitewxc.models.finetune_model import PatchEmbed
from granitewxc.utils.config import ExperimentConfig
from torch import nn


def get_embedding_network(config: ExperimentConfig) -> torch.nn.Module:
    if "encoder_decoder_kernel_size_per_stage" not in config.model.__dict__:
        config.model.encoder_decoder_kernel_size_per_stage = [
            [3] * len(inner) for inner in config.model.encoder_decoder_scale_per_stage
        ]

    n_output_parameters = len(config.data.output_vars)
    if config.model.__dict__.get("loss_type", "patch_rmse_loss") == "cross_entropy":
        if config.model.__dict__.get("cross_entropy_bin_width_type", "uniform") == "uniform":
            n_output_parameters = config.model.__dict__.get("cross_entropy_n_bins", 512)
        else:
            n_output_parameters = len(np.load(config.model.cross_entropy_bin_boundaries_file)) + 1

    n_parameters = len(config.data.input_surface_vars) + len(config.data.input_levels) * len(
        config.data.input_vertical_vars
    )

    embedding = PatchEmbed(
        patch_size=config.model.downscaling_patch_size,
        channels=n_parameters * config.data.n_input_timestamps,
        embed_dim=config.model.downscaling_embed_dim,
    )

    n_static_parameters = config.model.num_static_channels + len(config.data.input_static_surface_vars)
    if config.model.residual == "climate":
        n_static_parameters += n_parameters

    embedding_static = PatchEmbed(
        patch_size=config.model.downscaling_patch_size,
        channels=n_static_parameters,
        embed_dim=config.model.downscaling_embed_dim,
    )

    upscale = ConvEncoderDecoder(
        in_channels=config.model.downscaling_embed_dim,
        channels=config.model.encoder_decoder_conv_channels,
        out_channels=config.model.embed_dim,
        kernel_size=config.model.encoder_decoder_kernel_size_per_stage[0],
        scale=config.model.encoder_decoder_scale_per_stage[0],
        upsampling_mode=config.model.encoder_decoder_upsampling_mode,
    )
    return embedding, embedding_static, upscale
