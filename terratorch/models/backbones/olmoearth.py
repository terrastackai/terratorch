# Copyright contributors to the Terratorch project
# Licensed under the Apache License 2.0

"""OlmoEarth backbone integration for TerraTorch.

This module registers OlmoEarth (Allen AI) foundation models as backbones
in the TerraTorch backbone registry. OlmoEarth is a spatio-temporal, multimodal
foundation model for Earth observations supporting Sentinel-1, Sentinel-2, and Landsat.

Requires the `olmoearth-pretrain` package:
    pip install terratorch[olmoearth]

References:
    - GitHub: https://github.com/allenai/olmoearth_pretrain
    - Paper: https://arxiv.org/abs/2511.13655
"""

import logging
from collections.abc import Callable

import torch
from torch import Tensor, nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

logger = logging.getLogger(__name__)


# OlmoEarth model configurations
# Maps terratorch model names -> (ModelID enum value, encoder embedding size)
OLMOEARTH_CONFIGS: dict[str, dict] = {
    "olmoearth_v1_nano": {
        "model_id": "OlmoEarth-v1-Nano",
        "embed_dim": 64,
    },
    "olmoearth_v1_tiny": {
        "model_id": "OlmoEarth-v1-Tiny",
        "embed_dim": 128,
    },
    "olmoearth_v1_base": {
        "model_id": "OlmoEarth-v1-Base",
        "embed_dim": 768,
    },
    "olmoearth_v1_large": {
        "model_id": "OlmoEarth-v1-Large",
        "embed_dim": 1024,
    },
    "olmoearth_v1_1_nano": {
        "model_id": "OlmoEarth-v1_1-Nano",
        "embed_dim": 128,
    },
    "olmoearth_v1_1_tiny": {
        "model_id": "OlmoEarth-v1_1-Tiny",
        "embed_dim": 192,
    },
    "olmoearth_v1_1_base": {
        "model_id": "OlmoEarth-v1_1-Base",
        "embed_dim": 768,
    },
    "olmoearth_v1_2_nano": {
        "model_id": "OlmoEarth-v1_2-Nano",
        "embed_dim": 128,
    },
    "olmoearth_v1_2_tiny": {
        "model_id": "OlmoEarth-v1_2-Tiny",
        "embed_dim": 192,
    },
    "olmoearth_v1_2_small": {
        "model_id": "OlmoEarth-v1_2-Small",
        "embed_dim": 384,
    },
    "olmoearth_v1_2_base": {
        "model_id": "OlmoEarth-v1_2-Base",
        "embed_dim": 768,
    },
}


class OlmoEarthBackbone(nn.Module):
    """Wrapper around OlmoEarth encoder for use as a TerraTorch backbone.

    This wrapper loads an OlmoEarth model (encoder only) and adapts its
    forward pass to produce a list of feature maps compatible with TerraTorch
    decoders. The encoder's output embeddings are reshaped from (B, N, D) token
    sequences to (B, D, H, W) spatial feature maps.

    OlmoEarth's native interface expects a MaskedOlmoEarthSample input with
    multi-modal data. This wrapper provides a simplified interface that accepts
    a standard image tensor (B, C, H, W) and wraps it as Sentinel-2 input.
    """

    def __init__(
        self,
        model_id: str,
        embed_dim: int,
        pretrained: bool = True,
        patch_size: int = 8,
        input_res: int = 10,
        modality: str = "sentinel2",
        ckpt_path: str | None = None,
    ):
        """Initialize OlmoEarth backbone.

        Args:
            model_id: OlmoEarth model identifier (e.g. "OlmoEarth-v1-Base")
            embed_dim: Embedding dimension of the encoder
            pretrained: Whether to load pretrained weights from HuggingFace
            patch_size: Patch size for tokenization (default 8)
            input_res: Input resolution in meters (default 10m for Sentinel-2)
            modality: Input modality name (default "sentinel2")
            ckpt_path: Optional path to a local checkpoint file
        """
        super().__init__()

        from olmoearth_pretrain.model_loader import ModelID, load_model_from_id, load_model_from_path

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.input_res = input_res
        self.modality = modality

        # Load the full model (encoder + decoder)
        if ckpt_path is not None:
            full_model = load_model_from_path(ckpt_path, load_weights=pretrained)
        else:
            model_id_enum = ModelID(model_id)
            full_model = load_model_from_id(model_id_enum, load_weights=pretrained)

        # Extract only the encoder
        self.encoder = full_model.encoder

        # Expose embedding dimension for downstream components
        self.num_features = embed_dim

    @property
    def out_channels(self) -> list[int]:
        """Return the output channel dimensions for each feature level.

        OlmoEarth encoder produces a single feature level.
        """
        return [self.embed_dim]

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass through the OlmoEarth encoder.

        Takes a standard (B, C, H, W) image tensor and returns a list of
        spatial feature maps [(B, D, H', W')] suitable for TerraTorch decoders.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List containing a single feature map of shape (B, embed_dim, H', W')
            where H' = H // patch_size and W' = W // patch_size.
        """
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

        B, C, H, W = x.shape

        # Compute spatial grid dimensions after patchification
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # Create a MaskedOlmoEarthSample wrapping the input as the specified modality.
        # OlmoEarth expects input shape: (B, H_patches, W_patches, T, BandSets, C_per_bandset, patch_size, patch_size)
        # For single-timestep, single-band-set usage:
        # Reshape input: (B, C, H, W) -> (B, h_patches, w_patches, 1, 1, C, patch_size, patch_size)
        x_patchified = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        # x_patchified: (B, C, h_patches, w_patches, patch_size, patch_size)
        x_patchified = x_patchified.permute(0, 2, 3, 1, 4, 5)
        # x_patchified: (B, h_patches, w_patches, C, patch_size, patch_size)
        x_patchified = x_patchified.unsqueeze(3).unsqueeze(4)
        # x_patchified: (B, h_patches, w_patches, 1, 1, C, patch_size, patch_size)

        # Create mask: all tokens are visible (ONLINE_ENCODER value)
        mask = torch.full(
            (B, h_patches, w_patches, 1, 1),
            fill_value=MaskValue.ONLINE_ENCODER.value,
            device=x.device,
            dtype=torch.long,
        )

        # Construct timestamps (single timestep, zeros)
        timestamps = torch.zeros(B, 1, device=x.device, dtype=torch.float32)

        # Build the sample
        sample_dict = {
            self.modality: x_patchified,
            f"{self.modality}_mask": mask,
        }
        sample = MaskedOlmoEarthSample(
            timestamps=timestamps,
            **sample_dict,
        )

        # Run encoder
        encoder_output = self.encoder(
            sample,
            patch_size=self.patch_size,
            input_res=self.input_res,
        )

        # Extract the projected/aggregated embeddings
        # The encoder returns a dict with 'project_aggregated' containing per-patch embeddings
        # Shape: (B, h_patches, w_patches, embed_dim)
        proj_agg = encoder_output["project_aggregated"]

        # proj_agg is a Tensor of shape (B, h_patches * w_patches, embed_dim)
        # or it may come from TokensAndMasks - extract the modality tokens
        if hasattr(proj_agg, "as_tensor"):
            features = proj_agg.as_tensor()
        elif isinstance(proj_agg, dict):
            # Get the first available modality
            features = next(iter(proj_agg.values()))
            if features.ndim > 3:
                features = features.flatten(1, -2)
        elif isinstance(proj_agg, Tensor):
            features = proj_agg
        else:
            # Fallback: try to get tokens from tokens_and_masks
            tokens_and_masks = encoder_output["tokens_and_masks"]
            modality_tokens = getattr(tokens_and_masks, self.modality, None)
            if modality_tokens is not None:
                # Shape: (B, h, w, t, bs, D) -> flatten spatial
                features = modality_tokens.flatten(1, -2)
            else:
                msg = f"Could not extract features from OlmoEarth encoder output: {type(proj_agg)}"
                raise RuntimeError(msg)

        # Reshape from (B, N, D) to (B, D, H', W')
        if features.ndim == 3:
            features = features[:, : h_patches * w_patches, :]
            features = features.permute(0, 2, 1).reshape(B, self.embed_dim, h_patches, w_patches)
        elif features.ndim == 4:
            # Already (B, H', W', D) format
            features = features.permute(0, 3, 1, 2)

        return [features]


def _create_olmoearth_backbone(
    variant: str,
    pretrained: bool = True,
    patch_size: int = 8,
    input_res: int = 10,
    modality: str = "sentinel2",
    ckpt_path: str | None = None,
    **kwargs,
) -> OlmoEarthBackbone:
    """Create an OlmoEarth backbone model.

    Args:
        variant: Model variant name (e.g. "olmoearth_v1_base")
        pretrained: Whether to load pretrained weights
        patch_size: Patch size for tokenization
        input_res: Input resolution in meters
        modality: Input modality name
        ckpt_path: Optional path to local checkpoint
        **kwargs: Additional keyword arguments (unused)

    Returns:
        OlmoEarthBackbone instance
    """
    if variant not in OLMOEARTH_CONFIGS:
        msg = f"Unknown OlmoEarth variant '{variant}'. Available: {list(OLMOEARTH_CONFIGS.keys())}"
        raise ValueError(msg)

    cfg = OLMOEARTH_CONFIGS[variant]

    model = OlmoEarthBackbone(
        model_id=cfg["model_id"],
        embed_dim=cfg["embed_dim"],
        pretrained=pretrained,
        patch_size=patch_size,
        input_res=input_res,
        modality=modality,
        ckpt_path=ckpt_path,
    )

    return model


# Register all OlmoEarth variants with the TerraTorch backbone registry


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_nano(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1 Nano (1.4M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_nano", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_tiny(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1 Tiny (6.2M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_tiny", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_base(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1 Base (89M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_base", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_large(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1 Large (308M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_large", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_1_nano(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.1 Nano (5.5M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_1_nano", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_1_tiny(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.1 Tiny (12.5M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_1_tiny", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_1_base(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.1 Base (114M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_1_base", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_nano(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.2 Nano (5.5M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_2_nano", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_tiny(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.2 Tiny (12.5M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_2_tiny", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_small(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.2 Small (35.6M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_2_small", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_base(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.2 Base (114M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_2_base", pretrained=pretrained, **kwargs)
