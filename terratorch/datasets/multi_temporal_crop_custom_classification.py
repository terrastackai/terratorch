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
import xarray as xr
import torch
from einops import rearrange
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from xarray import DataArray

import warnings
from terratorch.datasets.utils import default_transform, filter_valid_files, validate_bands, to_rgb, generate_bands_intervals


class CDLMultiTemporalCropClassification(NonGeoDataset):
    """NonGeo dataset implementation for the custom multitemporal cropland classification of my master's thesis"""

    l2a_bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    ]

    class_names = (
        "Cropland",
        "Orchards",
        "Non-food cropland",
        "Wooded vegetation",
        "Grassland/other vegetation",
        "Non-vegetation"
    )

    rgb_bands = ("B4", "B3", "B2")

    def __init__(
        self,
        data_root: Path,
        num_classes: int,
        n_timesteps: int,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        dataset_bands: list[str] = l2a_bands,
        output_bands: list[str] = l2a_bands,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        expand_temporal_dimension: bool = True,
        temporal_channel_major: bool = False,
        reduce_zero_label: bool = True
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
        """
        super().__init__()

        label_data_root = label_data_root if label_data_root is not None else data_root

        self.split_file = split
        self.image_files = sorted(glob.glob(os.path.join(data_root, image_grep)))
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.segmentation_mask_files = sorted(glob.glob(os.path.join(label_data_root, label_grep)))
        self.reduce_zero_label = reduce_zero_label
        self.expand_temporal_dimension = expand_temporal_dimension
        self.temporal_channel_major = temporal_channel_major
        self.time_steps = n_timesteps
        self.num_classes = num_classes

        if self.expand_temporal_dimension:
            if not self.temporal_channel_major:
                warnings.warn(
                    "expand_temporal_dimension=True assumes bands are grouped by time "
                    "(all bands of one timestep are stacked together). "
                    "If instead bands are grouped by channel "
                    "(all timesteps of one band are stacked together), "
                    "set temporal_channel_major=True.")

            if dataset_bands is None:
                raise ValueError(
                    "Please provide dataset_bands when expand_temporal_dimension=True.")

        self.image_files = []
        self.segmentation_mask_files = []
        self.image_files = sorted(glob.glob(os.path.join(data_root, image_grep)))
        self.segmentation_mask_files = sorted(glob.glob(os.path.join(label_data_root, label_grep)))
        if self.expand_temporal_dimension and output_bands is None:
            msg = "Please provide output_bands when expand_temporal_dimension is True"
            raise Exception(msg)
        with open(self.split_file) as f:
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
        if not self.split_file:
            # When prediction is enabled, we don't have mask files, so
            # we need to provide a way to run the dataloder in these cases.
            if not self.segmentation_mask_files:
                self.segmentation_mask_files = self.image_files
                # The masks can be `None` since they won't be used in fact.
        self.rgb_indices = [3, 2, 1]
        self.dataset_bands = generate_bands_intervals(dataset_bands)
        self.output_bands = generate_bands_intervals(output_bands)
        self.filter_indices = None
        self.transform = transform if transform else default_transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = self._load_file(self.image_files[index], nan_replace=self.no_data_replace).to_numpy()
        # to channels last
        if self.expand_temporal_dimension:
            if self.temporal_channel_major:
                image = rearrange(image, "(channels time) h w -> channels time h w", channels=len(self.dataset_bands))
            else:
                image = rearrange(image, "(time channels) h w -> channels time h w", channels=len(self.dataset_bands))
        image = np.moveaxis(image, 0, -1)
        if self.filter_indices:
            image = image[..., self.filter_indices]
        output = {
            "image": image.astype(np.float32),
        }
        if self.segmentation_mask_files:
            mask = self._load_file(self.segmentation_mask_files[index], nan_replace=self.no_label_replace)
            output["mask"] = mask.to_numpy()[0]
            if self.reduce_zero_label:
                output["mask"] -= 1
        if self.transform:
            output = self.transform(**output)
        output["filename"] = self.image_files[index]

        return output

    def _load_file(self, path, nan_replace: int | float | None = None) -> xr.DataArray:
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

        rgb_indices = [self.l2a_bands.index(band) for band in self.rgb_bands]
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
