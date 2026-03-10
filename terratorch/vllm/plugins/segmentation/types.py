# Copyright contributors to the Terratorch project

import os
from pathlib import Path
from typing import Any, Literal, Optional, Union
from typing_extensions import Self

from pydantic import BaseModel, model_validator


class PluginConfig(BaseModel):
    output_path: str = None
    """
    Default output folder path to be used when the out_data_format is set to path. 
    If omitted, the plugin will default to the current user home directory.
    """

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if not self.output_path:
            self.output_path = str(Path.home())
        elif os.path.exists(self.output_path):
            if not os.access(self.output_path, os.W_OK):
                raise ValueError(f"The path: {self.output_path} is not writable")
        else:
            raise ValueError(f"The path: {self.output_path} does not exist")

        return self


class TiledInferenceParameters(BaseModel):
    h_crop: int = 512
    h_stride: int = None
    w_crop: int = 512
    w_stride: int = None
    average_patches: bool = True
    delta: int = 8
    blend_overlaps: bool = True
    padding: str | bool = "reflect"


class RequestData(BaseModel):
    data_format: Literal["path", "url"]
    """
    Data type for the input image.
    Allowed values are: [`path`, `url`]
    """

    out_data_format: Literal["b64_json", "path"]
    """
    Data type for the output image.
    Allowed values are: [`b64_json`, `path`]
    """

    data: Any
    """
    Input image data
    """

    indices: Optional[list[int]] = None
    """
    Indices for bands to be processed in the input file
    """

    out_path: Optional[str] = None
    """
    Path to store the output image. Only used when out_data_format is set to 'path'
    """


class RequestInfo(BaseModel):
    """Base class for request cache entries, holding fields common to all plugins."""

    model_config = {"arbitrary_types_allowed": True}

    out_data_format: Literal["b64_json", "path"]
    """
    Data type for the output image.
    Allowed values are: [`b64_json`, `path`]
    """

    out_path: Optional[str] = None
    """
    Path to store the output image. Only used when out_data_format is set to 'path'.
    """

    metadata: dict
    """Rasterio metadata dict for the input image."""


class SegmentationRequestInfo(RequestInfo):
    """Request cache entry for :class:`SegmentationIOProcessor`."""

    original_h: int
    """Original (unpadded) image height in pixels."""

    original_w: int
    """Original (unpadded) image width in pixels."""

    h1: int
    """Number of tile rows."""

    w1: int
    """Number of tile columns."""


class TerramindSegmentationRequestInfo(RequestInfo):
    """Request cache entry for :class:`TerramindSegmentationIOProcessor`."""

    dataset_path: str
    """Path to the directory containing the input dataset."""

    prompt_data: list
    """List of tile data objects produced by :func:`prepare_tiled_inference_input`."""

    h_img: int
    """Full image height in pixels."""

    w_img: int
    """Full image width in pixels."""

    input_batch_size: int
    """Batch size used for tiled inference."""

    filename: str
    """Path to the primary input file (used to derive the output filename)."""

    delta: int
    """Overlap delta used during tiled inference."""


MultiModalPromptType = Union[RequestData]


class RequestOutput(BaseModel):
    data_format: Literal["b64_json", "path"]
    """
    Data type for the output image.
    Allowed values are: [`b64_json`, `path`]
    """

    data: Any
    """
    Output image data
    """

    request_id: Optional[str] = None
    """
    The vLLM request ID if applicable
    """
