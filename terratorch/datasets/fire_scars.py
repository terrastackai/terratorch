# Copyright contributors to the Terratorch project

import dataclasses
import glob
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torchgeo.datasets import NonGeoDataset, RasterDataset
from xarray import DataArray

from terratorch.datasets.utils import clip_image_percentile, default_transform, validate_bands


class FireScarsNonGeo(NonGeoDataset):
    """NonGeo dataset implementation for [fire scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars)."""

    all_band_names = (
        "BLUE",
        "GREEN",
        "RED",
        "NIR_NARROW",
        "SWIR_1",
        "SWIR_2",
    )

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    num_classes = 2
    splits = {"train": "training", "val": "validation"}  # Only train and val splits available

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        no_data_replace: float | None = 0,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,
    ) -> None:
        """Constructor

        Args:
            data_root (str): Path to the data root directory.
            bands (list[str]): Bands that should be output by the dataset. Defaults to all bands.
            transform (A.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the corresponding data module,
                should not include normalization. Defaults to None, which applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value.
                If None, does no replacement. Defaults to 0.
            no_label_replace (int | None): Replace nan values in label with this value.
                If none, does no replacement. Defaults to -1.
            use_metadata (bool): whether to return metadata info (time and location).
        """
        super().__init__()
        if split not in self.splits:
            msg = f"Incorrect split '{split}', please choose one of {self.splits}."
            raise ValueError(msg)
        split_name = self.splits[split]
        self.split = split

        validate_bands(bands, self.all_band_names)
        self.bands = bands
        self.band_indices = np.asarray([self.all_band_names.index(b) for b in bands])
        self.data_root = Path(data_root)

        input_dir = self.data_root / split_name
        self.image_files = sorted(glob.glob(os.path.join(input_dir, "*_merged.tif")))
        self.segmentation_mask_files = sorted(glob.glob(os.path.join(input_dir, "*.mask.tif")))

        self.use_metadata = use_metadata
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace

        # If no transform is given, apply only to transform to torch tensor
        self.transform = transform if transform else default_transform

    def __len__(self) -> int:
        return len(self.image_files)

    def _get_date(self, index: int) -> torch.Tensor:
        file_name = self.image_files[index]
        base_filename = os.path.basename(file_name)

        filename_regex = r"subsetted_512x512_HLS\.S30\.T[0-9A-Z]{5}\.(?P<date>[0-9]+)\.v1\.4_merged\.tif"
        match = re.match(filename_regex, base_filename)
        date_str = match.group("date")
        year = int(date_str[:4])
        julian_day = int(date_str[4:])

        return torch.tensor([[year, julian_day]], dtype=torch.float32)

    def _get_coords(self, image: DataArray) -> torch.Tensor:
        px = image.x.shape[0] // 2
        py = image.y.shape[0] // 2

        # get center point to reproject to lat/lon
        point = image.isel(band=0, x=slice(px, px + 1), y=slice(py, py + 1))
        point = point.rio.reproject("epsg:4326")

        lat_lon = np.asarray([point.y[0], point.x[0]])

        return torch.tensor(lat_lon, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = self._load_file(self.image_files[index], nan_replace=self.no_data_replace)

        location_coords, temporal_coords = None, None
        if self.use_metadata:
            location_coords = self._get_coords(image)
            temporal_coords = self._get_date(index)

        # to channels last
        image = image.to_numpy()
        image = np.moveaxis(image, 0, -1)

        # filter bands
        image = image[..., self.band_indices]

        output = {
            "image": image.astype(np.float32),
            "mask": self._load_file(self.segmentation_mask_files[index], nan_replace=self.no_label_replace).to_numpy()[
                0
            ],
        }
        if self.transform:
            output = self.transform(**output)
        output["mask"] = output["mask"].long()

        if self.use_metadata:
            output["location_coords"] = location_coords
            output["temporal_coords"] = temporal_coords

        return output

    def _load_file(self, path: Path, nan_replace: int | float | None = None) -> DataArray:
        data = rioxarray.open_rasterio(path, masked=True)
        if nan_replace is not None:
            data = data.fillna(nan_replace)
        return data

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        num_images = 4

        rgb_indices = [self.bands.index(band) for band in self.rgb_bands]
        if len(rgb_indices) != 3:
            msg = "Dataset doesn't contain some of the RGB bands"
            raise ValueError(msg)

        # RGB -> channels-last
        image = sample["image"][rgb_indices, ...].permute(1, 2, 0).numpy()
        mask = sample["mask"].numpy()

        image = clip_image_percentile(image)

        if "prediction" in sample:
            prediction = sample["prediction"]
            num_images += 1
        else:
            prediction = None

        fig, ax = plt.subplots(1, num_images, figsize=(12, 5), layout="compressed")

        ax[0].axis("off")

        norm = mpl.colors.Normalize(vmin=0, vmax=self.num_classes - 1)
        ax[1].axis("off")
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        ax[2].axis("off")
        ax[2].title.set_text("Ground Truth Mask")
        ax[2].imshow(mask, cmap="jet", norm=norm)

        ax[3].axis("off")
        ax[3].title.set_text("GT Mask on Image")
        ax[3].imshow(image)
        ax[3].imshow(mask, cmap="jet", alpha=0.3, norm=norm)

        if "prediction" in sample:
            ax[4].title.set_text("Predicted Mask")
            ax[4].imshow(prediction, cmap="jet", norm=norm)

        cmap = plt.get_cmap("jet")
        legend_data = [[i, cmap(norm(i)), str(i)] for i in range(self.num_classes)]
        handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        labels = [n for k, c, n in legend_data]
        ax[0].legend(handles, labels, loc="center")
        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class FireScarsHLS(RasterDataset):
    """RasterDataset implementation for fire scars input images."""

    filename_glob = "subsetted*_merged.tif"
    filename_regex = r"subsetted_512x512_HLS\..30\..{6}\.(?P<date>[0-9]*)\.v1.4_merged.tif"
    date_format = "%Y%j"
    is_image = True
    separate_files = False
    all_bands = dataclasses.field(default_factory=["B02", "B03", "B04", "B8A", "B11", "B12"])
    rgb_bands = dataclasses.field(default_factory=["B04", "B03", "B02"])


class FireScarsSegmentationMask(RasterDataset):
    """RasterDataset implementation for fire scars segmentation mask.
    Can be easily merged with input images using the & operator.
    """

    filename_glob = "subsetted*.mask.tif"
    filename_regex = r"subsetted_512x512_HLS\..30\..{6}\.(?P<date>[0-9]*)\.v1.4.mask.tif"
    date_format = "%Y%j"
    is_image = False
    separate_files = False
