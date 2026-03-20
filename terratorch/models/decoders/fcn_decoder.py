# Copyright contributors to the Terratorch project

from torch import Tensor, nn

from terratorch.registry import TERRATORCH_DECODER_REGISTRY


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: Tensor):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def _conv_upscale_block(input_channels, output_channels, kernel_size, stride, dilation, padding, output_padding):
    return nn.Sequential(
        nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            output_padding=output_padding,
        ),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=True),
        Norm2d(output_channels),
        nn.GELU(),
    )


@TERRATORCH_DECODER_REGISTRY.register
class FCNDecoder(nn.Module):
    """Fully Convolutional Decoder"""

    def __init__(self, embed_dim: int, channels: int = 256, num_convs: int = 4, in_index: int = -1) -> None:
        """Constructor

        Args:
            embed_dim (_type_): Input embedding dimension
            channels (int, optional): Number of channels for each conv. Defaults to 256.
            num_convs (int, optional): Number of convs. Defaults to 4.
            in_index (int, optional): Index of the input list to take. Defaults to -1.
        """
        super().__init__()
        kernel_size = 2
        stride = 2
        dilation = 1
        padding = 0
        output_padding = 0
        self.channels = channels
        self.num_convs = num_convs
        self.in_index = in_index
        self.embed_dim = embed_dim[in_index]
        if num_convs < 1:
            msg = "num_convs must be >= 1"
            raise Exception(msg)

        convs = []

        for i in range(num_convs):
            in_channels = self.embed_dim if i == 0 else self.channels
            convs.append(
                _conv_upscale_block(in_channels, self.channels, kernel_size, stride, dilation, padding, output_padding)
            )

        self.convs = nn.Sequential(*convs)

    @property
    def out_channels(self):
        return self.channels

    def forward(self, x: list[Tensor]):
        x = x[self.in_index]
        decoded = self.convs(x)
        return decoded
