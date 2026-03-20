import glob
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import torch
from einops import rearrange
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from xarray import DataArray

from terratorch.datasets.utils import default_transform, filter_valid_files, to_rgb, validate_bands


class MultiTemporalCropClassification(NonGeoDataset):
    """NonGeo dataset implementation for [multi-temporal crop classification](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)."""

    all_band_names = (
        "BLUE",
        "GREEN",
        "RED",
        "NIR_NARROW",
        "SWIR_1",
        "SWIR_2",
    )

    class_names = (
        "Natural Vegetation",
        "Forest",
        "Corn",
        "Soybeans",
        "Wetlands",
        "Developed / Barren",
        "Open Water",
        "Winter Wheat",
        "Alfalfa",
        "Fallow / Idle Cropland",
        "Cotton",
        "Sorghum",
        "Other",
    )

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    num_classes = 13
    time_steps = 3
    splits = {"train": "training", "val": "validation"}  # Only train and val splits available
    col_name = "chip_id"
    date_columns = ["first_img_date", "middle_img_date", "last_img_date"]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        expand_temporal_dimension: bool = True,
        reduce_zero_label: bool = True,
        use_metadata: bool = False,
        metadata_file_name: str = "chips_df.csv",
    ) -> None:
        """Constructor

        Args:
            data_root (str): Path to the data root directory.
            split (str): one of 'train' or 'val'.
            bands (list[str]): Bands that should be output by the dataset. Defaults to all bands.
            transform (A.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the corresponding data module,
                should not include normalization. Defaults to None, which applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value.
                If None, does no replacement. Defaults to None.
            no_label_replace (int | None): Replace nan values in label with this value.
                If none, does no replacement. Defaults to None.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to True.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to True.
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

        data_dir = self.data_root / f"{split_name}_chips"
        self.image_files = sorted(glob.glob(os.path.join(data_dir, "*_merged.tif")))
        self.segmentation_mask_files = sorted(glob.glob(os.path.join(data_dir, "*.mask.tif")))
        split_file = self.data_root / f"{split_name}_data.txt"

        with open(split_file) as f:
            split = f.readlines()
        valid_files = {rf"{substring.strip()}" for substring in split}
        self.image_files = filter_valid_files(
            self.image_files,
            valid_files=valid_files,
            ignore_extensions=True,
            allow_substring=True,
        )
        self.segmentation_mask_files = filter_valid_files(
            self.segmentation_mask_files,
            valid_files=valid_files,
            ignore_extensions=True,
            allow_substring=True,
        )

        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.reduce_zero_label = reduce_zero_label
        self.expand_temporal_dimension = expand_temporal_dimension
        self.use_metadata = use_metadata
        self.metadata = None
        self.metadata_file_name = metadata_file_name
        if self.use_metadata:
            metadata_file = self.data_root / self.metadata_file_name
            self.metadata = pd.read_csv(metadata_file)
            self._build_image_metadata_mapping()

        # If no transform is given, apply only to transform to torch tensor
        self.transform = transform if transform else default_transform

    def _build_image_metadata_mapping(self):
        """Build a mapping from image filenames to metadata indices."""
        self.image_to_metadata_index = dict()

        for idx, image_file in enumerate(self.image_files):
            image_filename = Path(image_file).name
            image_id = image_filename.replace("_merged.tif", "").replace(".tif", "")
            metadata_indices = self.metadata.index[self.metadata[self.col_name] == image_id].tolist()
            self.image_to_metadata_index[idx] = metadata_indices[0]

    def __len__(self) -> int:
        return len(self.image_files)

    def _get_date(self, row: pd.Series) -> torch.Tensor:
        """Extract and format temporal coordinates (T, date) from metadata."""
        temporal_coords = []
        for col in self.date_columns:
            date_str = row[col]
            date = pd.to_datetime(date_str)
            temporal_coords.append([date.year, date.dayofyear - 1])

        return torch.tensor(temporal_coords, dtype=torch.float32)

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
            metadata_idx = self.image_to_metadata_index.get(index, None)
            if metadata_idx is not None:
                row = self.metadata.iloc[metadata_idx]
                temporal_coords = self._get_date(row)

        # to channels last
        image = image.to_numpy()
        if self.expand_temporal_dimension:
            image = rearrange(image, "(time channels) h w -> channels time h w", channels=len(self.bands))
        image = np.moveaxis(image, 0, -1)

        # filter bands
        image = image[..., self.band_indices]

        output = {
            "image": image.astype(np.float32),
            "mask": self._load_file(self.segmentation_mask_files[index], nan_replace=self.no_label_replace).to_numpy()[
                0
            ],
        }

        if self.reduce_zero_label:
            output["mask"] -= 1
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
        num_images = self.time_steps + 2

        rgb_indices = [self.bands.index(band) for band in self.rgb_bands]
        if len(rgb_indices) != 3:
            msg = "Dataset doesn't contain some of the RGB bands"
            raise ValueError(msg)

        images = sample["image"]
        images = images[rgb_indices, ...]  # Shape: (T, 3, H, W)

        processed_images = []
        for t in range(self.time_steps):
            img = images[:, t]
            img = img.numpy()
            img = to_rgb(img, rgb_indices, gamma=0.9)
            processed_images.append(img)

        mask = sample["mask"].numpy()
        if "prediction" in sample:
            num_images += 1
        fig, ax = plt.subplots(1, num_images, figsize=(12, 5), layout="compressed")
        ax[0].axis("off")

        norm = mpl.colors.Normalize(vmin=0, vmax=self.num_classes - 1)
        for i, img in enumerate(processed_images):
            ax[i + 1].axis("off")
            ax[i + 1].title.set_text(f"T{i}")
            ax[i + 1].imshow(img)

        ax[self.time_steps + 1].axis("off")
        ax[self.time_steps + 1].title.set_text("Ground Truth Mask")
        ax[self.time_steps + 1].imshow(mask, cmap="jet", norm=norm)

        if "prediction" in sample:
            prediction = sample["prediction"]
            ax[self.time_steps + 2].axis("off")
            ax[self.time_steps + 2].title.set_text("Predicted Mask")
            ax[self.time_steps + 2].imshow(prediction, cmap="jet", norm=norm)

        cmap = plt.get_cmap("jet")
        legend_data = [[i, cmap(norm(i)), self.class_names[i]] for i in range(self.num_classes)]
        handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        labels = [n for k, c, n in legend_data]
        ax[0].legend(handles, labels, loc="center")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
