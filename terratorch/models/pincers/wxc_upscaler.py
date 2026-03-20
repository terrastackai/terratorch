from granitewxc.decoders.downscaling import ConvEncoderDecoder
from granitewxc.utils.config import ExperimentConfig


def get_upscaler(config: ExperimentConfig, n_output_parameters: int) -> ConvEncoderDecoder:
    head = ConvEncoderDecoder(
        in_channels=config.model.embed_dim,
        channels=config.model.encoder_decoder_conv_channels,
        out_channels=n_output_parameters,
        kernel_size=config.model.encoder_decoder_kernel_size_per_stage[1],
        scale=config.model.encoder_decoder_scale_per_stage[1],
        upsampling_mode=config.model.encoder_decoder_upsampling_mode,
    )
    return head
