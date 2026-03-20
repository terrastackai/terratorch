# Copyright contributors to the Terratorch project

"""Pass the features straight through"""

from torch import Tensor, nn

from terratorch.registry import TERRATORCH_DECODER_REGISTRY


@TERRATORCH_DECODER_REGISTRY.register
class IdentityDecoder(nn.Module):
    """Identity decoder. Useful to pass the feature straight to the head."""

    def __init__(self, embed_dim: int, out_index=-1) -> None:
        """Constructor

        Args:
            embed_dim (int): Input embedding dimension
            out_index (int, optional): Index of the input list to take.. Defaults to -1.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.dim = out_index

    @property
    def out_channels(self):
        return self.embed_dim[self.dim]

    def forward(self, x: list[Tensor]):
        return x[self.dim]
