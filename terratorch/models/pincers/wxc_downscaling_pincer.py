import numpy as np
from granitewxc.utils.config import ExperimentConfig
from granitewxc.utils.downscaling_model import ClimateDownscaleFinetuneModel, get_scalers
from torch import nn

from terratorch.models.pincers.wxc_embedding_network import get_embedding_network
from terratorch.models.pincers.wxc_upscaler import get_upscaler


def get_downscaling_pincer(config: ExperimentConfig, backbone: nn.Module):

    n_output_parameters = len(config.data.output_vars)
    if config.model.__dict__.get("loss_type", "patch_rmse_loss") == "cross_entropy":
        if config.model.__dict__.get("cross_entropy_bin_width_type", "uniform") == "uniform":
            n_output_parameters = config.model.__dict__.get("cross_entropy_n_bins", 512)
        else:
            n_output_parameters = len(np.load(config.model.cross_entropy_bin_boundaries_file)) + 1

    embedding, embedding_static, upscale = get_embedding_network(config)
    head = get_upscaler(config, n_output_parameters)
    scalers = get_scalers(config)

    model = ClimateDownscaleFinetuneModel(
        embedding=embedding,
        embedding_static=embedding_static,
        upscale=upscale,
        backbone=backbone,
        head=head,
        input_scalers_mu=scalers["input_mu"],
        input_scalers_sigma=scalers["input_sigma"],
        input_scalers_epsilon=1e-6,
        static_input_scalers_mu=scalers["input_static_mu"],
        static_input_scalers_sigma=scalers["input_static_sigma"],
        static_input_scalers_epsilon=1e-6,
        output_scalers_mu=scalers["target_mu"],
        output_scalers_sigma=scalers["target_sigma"],
        n_input_timestamps=config.data.n_input_timestamps,
        embed_dim_backbone=config.model.embed_dim,
        n_lats_px_backbone=int(config.data.input_size_lat * np.prod(config.model.encoder_decoder_scale_per_stage[0])),
        n_lons_px_backbone=int(config.data.input_size_lon * np.prod(config.model.encoder_decoder_scale_per_stage[0])),
        patch_size_px_backbone=config.model.token_size,
        mask_unit_size_px_backbone=config.mask_unit_size,
        n_bins=n_output_parameters,
        return_logits=config.model.__dict__.get("loss_type") == "cross_entropy",
        residual=config.model.__dict__.get("residual", None),
        residual_connection=config.model.__dict__.get("residual_connection", False),
        config=config,
    )
    return model
