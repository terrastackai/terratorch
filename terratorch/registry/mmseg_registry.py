import importlib
import inspect
import logging
import warnings
from collections.abc import Callable, Set

from torch import nn

from terratorch.registry import DECODER_REGISTRY


class MMsegDecoderWrapper(nn.Module):
    """
    A wrapper for decoders designed to handle single or multiple embeddings with specified indices.

    Attributes:
        decoder (nn.Module): The decoder module being wrapped.
        channels (int): The number of output channels of the decoder.
    Methods:
        forward(x: List[torch.Tensor]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    includes_head: bool = True

    def __init__(self, decoder) -> None:
        """
        Args:
            decoder (nn.Module): The decoder module to be wrapped.
            out_channels (int): Output channels of the decoder, needed for terratorch head.
                Necessary since not all smp decoders have the `out_channels` attribute.
            Defaults to -1.
        """
        super().__init__()
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(x)


class MMSegRegistry(Set):
    """Registry wrapper for mmseg"""

    includes_head: bool = True

    def __init__(self):
        if not importlib.util.find_spec("mmseg"):
            msg = "mmsegmentation must be installed to instantiate an MMSegRegistry"
            raise ImportError(msg)
        self.mmseg_reg = importlib.import_module("mmseg.models.decode_heads")
        self.mmseg_decoder_names = {x for x, _ in inspect.getmembers(self.mmseg_reg, inspect.isclass)}

    def register(self, constructor: Callable | type) -> Callable:
        raise NotImplementedError()

    def build(self, name: str, in_channels, *constructor_args, **constructor_kwargs) -> nn.Module:
        """Build and return the component.
        Use prefixes ending with _ to forward to a specific source
        """
        try:
            decoder = getattr(self.mmseg_reg, name)
        except AttributeError as e:
            msg = f"Decoder {name} not found"
            raise KeyError(msg) from e

        if len(in_channels) == 1:
            in_channels = in_channels[0]
        if "num_classes" not in constructor_kwargs:
            msg = "num_classes is a required argument for mmseg decoders. If you are using an mmseg decoder for a regression task please include num_classes=1."
            raise ValueError(msg)

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=UserWarning
            )  # dont worry about warnings related to loss. we dont use that component.
            decoder = decoder(*constructor_args, in_channels=in_channels, **constructor_kwargs)
        return MMsegDecoderWrapper(decoder)

    def __iter__(self):
        return iter(self.mmseg_decoder_names)

    def __len__(self):
        return len(self.mmseg_decoder_names)

    def __contains__(self, key):
        return key in self.mmseg_decoder_names

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"Mmseg registry with {len(self)} registered backbones"

    def __getitem__(self, name):
        try:
            return getattr(self.mmseg_reg, name)
        except AttributeError as e:
            msg = f"Decoder {name} not found"
            raise KeyError(msg) from e


if importlib.util.find_spec("mmseg"):
    MMSEG_DECODER_REGISTRY = MMSegRegistry()
    DECODER_REGISTRY.register_source("mmseg", MMSEG_DECODER_REGISTRY)
else:
    logging.getLogger("terratorch").debug("mmseg not installed, so MmsegDecoderRegistry not created")
