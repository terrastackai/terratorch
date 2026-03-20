# Copyright contributors to the Terratorch project

from torch import Tensor, nn

from terratorch.registry import TERRATORCH_DECODER_REGISTRY


@TERRATORCH_DECODER_REGISTRY.register
class LinearDecoder(nn.Module):
    """
    A linear decoder using a transposed convolution layer for upsampling.
    """

    includes_head: bool = True

    def __init__(self, embed_dim: list[int], num_classes: int, upsampling_size: int, in_index: int = -1) -> None:
        """Constructor

        Args:
            embed_dim (list[int]): A list of embedding dimensions for different feature maps.
            num_classes (int): Number of output classes.
            upsampling_size (int): Kernel and stride size for transposed convolution.
            in_index (int, optional): Index of the input feature map to use. Defaults to -1."
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_index = in_index
        self.embed_dim = embed_dim[in_index]

        self.conv = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.num_classes,
            kernel_size=upsampling_size,
            stride=upsampling_size,
            padding=0,
            output_padding=0,
        )

    @property
    def out_channels(self) -> int:
        return self.num_classes

    def forward(self, x: list[Tensor]) -> Tensor:
        return self.conv(x[self.in_index])
