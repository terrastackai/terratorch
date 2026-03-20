import warnings

from terratorch.registry import TERRATORCH_FULL_MODEL_REGISTRY

try:
    from terracodec import (
        flextec_v1_s2l2a as _flextec_v1_s2l2a,
    )
    from terracodec import (
        terracodec_v1_elic_s2l2a as _terracodec_v1_elic_s2l2a,
    )
    from terracodec import (
        terracodec_v1_fp_s2l2a as _terracodec_v1_fp_s2l2a,
    )
    from terracodec import (
        terracodec_v1_tt_s2l1c as _terracodec_v1_tt_s2l1c,
    )
    from terracodec import (
        terracodec_v1_tt_s2l2a as _terracodec_v1_tt_s2l2a,
    )

    terracodec_available = True
    import_error = None

except Exception as e:
    terracodec_available = False
    import_error = e

__all__ = [
    "flextec_v1_s2l2a",
    "terracodec_v1_elic_s2l2a",
    "terracodec_v1_fp_s2l2a",
    "terracodec_v1_tt_s2l1c",
    "terracodec_v1_tt_s2l2a",
]


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terracodec_v1_fp_s2l2a(
    compression: str | float | int = "lambda-10",
    image_size: int = 256,
    mode="eval",
    **kwargs,
):
    """TerraCodec 1.0 FactorizedPrior model for Sentinel-2 L2A data."""
    if not terracodec_available:
        warnings.warn("Cannot import TerraCodec model. \nMake sure to install `pip install terracodec` first.")
        raise import_error
    return _terracodec_v1_fp_s2l2a(compression=compression, image_size=image_size, mode=mode, **kwargs)


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terracodec_v1_elic_s2l2a(
    compression: str | float | int = "lambda-10",
    image_size: int = 256,
    mode="eval",
    **kwargs,
):
    """TerraCodec 1.0 ELIC model for Sentinel-2 L2A data."""
    if not terracodec_available:
        warnings.warn("Cannot import TerraCodec model. \nMake sure to install `pip install terracodec` first.")
        raise import_error
    return _terracodec_v1_elic_s2l2a(compression=compression, image_size=image_size, mode=mode, **kwargs)


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terracodec_v1_tt_s2l2a(
    compression: str | float | int = "lambda-5",
    image_size: int = 256,
    mode="eval",
    **kwargs,
):
    """TerraCodec 1.0 Temporal Transformer model for Sentinel-2 L2A data."""
    if not terracodec_available:
        warnings.warn("Cannot import TerraCodec model. \nMake sure to install `pip install terracodec` first.")
        raise import_error
    return _terracodec_v1_tt_s2l2a(compression=compression, image_size=image_size, mode=mode, **kwargs)


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terracodec_v1_tt_s2l1c(
    compression: str | float | int = "lambda-20",
    image_size: int = 256,
    mode="eval",
    **kwargs,
):
    """TerraCodec 1.0 Temporal Transformer model for Sentinel-2 L1C data."""
    if not terracodec_available:
        warnings.warn("Cannot import TerraCodec model. \nMake sure to install `pip install terracodec` first.")
        raise import_error
    return _terracodec_v1_tt_s2l1c(compression=compression, image_size=image_size, mode=mode, **kwargs)


@TERRATORCH_FULL_MODEL_REGISTRY.register
def flextec_v1_s2l2a(
    compression: str | float | int = "lambda-800",
    image_size: int = 256,
    mode="eval",
    lr_only: bool = False,
    **kwargs,
):
    """TerraCodec 1.0 FlexTEC model for Sentinel-2 L2A data."""
    if not terracodec_available:
        warnings.warn("Cannot import TerraCodec model. \nMake sure to install `pip install terracodec` first.")
        raise import_error
    return _flextec_v1_s2l2a(
        compression=compression,
        image_size=image_size,
        mode=mode,
        lr_only=lr_only,
        **kwargs,
    )
