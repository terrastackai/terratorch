import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import torch
from matplotlib.colors import Normalize
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from xarray import DataArray

from terratorch.datasets.utils import default_transform, validate_bands


class BurnIntensityNonGeo(NonGeoDataset):
    """Dataset implementation for [Burn Intensity classification](https://huggingface.co/datasets/ibm-nasa-geospatial/burn_intensity)."""

    all_band_names = (
        "BLUE",
        "GREEN",
        "RED",
        "NIR",
        "SWIR_1",
        "SWIR_2",
    )

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    class_names = ("No burn", "Unburned to Very Low", "Low Severity", "Moderate Severity", "High Severity")

    CSV_FILES = {
        "limited": "BS_files_with_less_than_25_percent_zeros.csv",
        "full": "BS_files_raw.csv",
    }

    num_classes = 5
    splits = {"train": "train", "val": "val"}
    time_steps = ["pre", "during", "post"]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        use_full_data: bool = True,
        no_data_replace: float | None = 0.0001,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,
    ) -> None:
        """Initialize the BurnIntensity dataset.

        Args:
            data_root (str): Path to the data root directory.
            split (str): One of 'train' or 'val'.
            bands (Sequence[str]): Bands to output. Defaults to all bands.
            transform (Optional[A.Compose]): Albumentations transform to be applied.
            use_metadata (bool): Whether to return metadata info (location).
            use_full_data (bool): Wheter to use full data or data with less than 25 percent zeros.
            no_data_replace (Optional[float]): Value to replace NaNs in images.
            no_label_replace (Optional[int]): Value to replace NaNs in labels.
        """
        super().__init__()
        if split not in self.splits:
            msg = f"Incorrect split '{split}', please choose one of {list(self.splits.keys())}."
            raise ValueError(msg)
        self.split = split

        validate_bands(bands, self.all_band_names)
        self.bands = bands
        self.band_indices = np.asarray([self.all_band_names.index(b) for b in bands])

        self.data_root = Path(data_root)

        # Read the CSV file to get the list of cases to include
        csv_file_key = "full" if use_full_data else "limited"
        csv_path = self.data_root / self.CSV_FILES[csv_file_key]
        df = pd.read_csv(csv_path)
        casenames = df["Case_Name"].tolist()

        split_file = self.data_root / f"{split}.txt"
        with open(split_file) as f:
            split_images = [line.strip() for line in f.readlines()]

        split_images = [img for img in split_images if self._extract_casename(img) in casenames]

        # Build the samples list
        self.samples = []
        for image_filename in split_images:
            image_files = []
            for time_step in self.time_steps:
                image_file = self.data_root / time_step / image_filename
                image_files.append(str(image_file))
            mask_filename = image_filename.replace("HLS_", "BS_")
            mask_file = self.data_root / "pre" / mask_filename
            self.samples.append(
                {
                    "image_files": image_files,
                    "mask_file": str(mask_file),
                    "casename": self._extract_casename(image_filename),
                }
            )

        self.use_metadata = use_metadata
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace

        self.transform = transform if transform else default_transform

    def _extract_basename(self, filepath: str) -> str:
        """Extract the base filename without extension."""
        return os.path.splitext(os.path.basename(filepath))[0]

    def _extract_casename(self, filename: str) -> str:
        """Extract the casename from the filename."""
        basename = self._extract_basename(filename)
        # Remove 'HLS_' or 'BS_' prefix
        casename = basename.replace("HLS_", "").replace("BS_", "")
        return casename

    def __len__(self) -> int:
        return len(self.samples)

    def _get_coords(self, image: DataArray) -> torch.Tensor:
        pixel_scale = image.rio.resolution()
        width, height = image.rio.width, image.rio.height

        left, bottom, right, top = image.rio.bounds()
        tie_point_x, tie_point_y = left, top

        center_col = width / 2
        center_row = height / 2

        center_lon = tie_point_x + (center_col * pixel_scale[0])
        center_lat = tie_point_y - (center_row * pixel_scale[1])

        lat_lon = np.asarray([center_lat, center_lon])
        return torch.tensor(lat_lon, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_files = sample["image_files"]
        mask_file = sample["mask_file"]

        images = []
        for idx, image_file in enumerate(image_files):
            image = self._load_file(Path(image_file), nan_replace=self.no_data_replace)
            if idx == 0 and self.use_metadata:
                location_coords = self._get_coords(image)
            image = image.to_numpy()
            image = np.moveaxis(image, 0, -1)
            image = image[..., self.band_indices]
            images.append(image)

        images = np.stack(images, axis=0)  # (T, H, W, C)

        output = {
            "image": images.astype(np.float32),
            "mask": self._load_file(Path(mask_file), nan_replace=self.no_label_replace).to_numpy()[0],
        }

        if self.transform:
            output = self.transform(**output)

        output["mask"] = output["mask"].long()
        if self.use_metadata:
            output["location_coords"] = location_coords

        return output

    def _load_file(self, path: Path, nan_replace: float | int | None = None) -> DataArray:
        data = rioxarray.open_rasterio(path, masked=True)
        if nan_replace is not None:
            data = data.fillna(nan_replace)
        return data

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Any:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by `__getitem__`.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        num_images = len(self.time_steps) + 2
        if "prediction" in sample:
            num_images += 1

        rgb_indices = [self.bands.index(band) for band in self.rgb_bands if band in self.bands]
        if len(rgb_indices) != 3:
            msg = "Dataset doesn't contain some of the RGB bands"
            raise ValueError(msg)

        images = sample["image"]  # (C, T, H, W)
        mask = sample["mask"].numpy()
        num_classes = len(np.unique(mask))

        fig, ax = plt.subplots(1, num_images, figsize=(num_images * 5, 5))

        for i in range(len(self.time_steps)):
            image = images[:, i, :, :]  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # (H, W, C)
            rgb_image = image[..., rgb_indices]
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
            rgb_image = np.clip(rgb_image, 0, 1)
            ax[i].imshow(rgb_image)
            ax[i].axis("off")
            ax[i].set_title(f"{self.time_steps[i].capitalize()} Image")

        cmap = plt.get_cmap("jet", num_classes)
        norm = Normalize(vmin=0, vmax=num_classes - 1)

        mask_ax_index = len(self.time_steps)
        ax[mask_ax_index].imshow(mask, cmap=cmap, norm=norm)
        ax[mask_ax_index].axis("off")
        ax[mask_ax_index].set_title("Ground Truth Mask")

        if "prediction" in sample:
            prediction = sample["prediction"].numpy()
            pred_ax_index = mask_ax_index + 1
            ax[pred_ax_index].imshow(prediction, cmap=cmap, norm=norm)
            ax[pred_ax_index].axis("off")
            ax[pred_ax_index].set_title("Predicted Mask")

        legend_ax_index = -1
        class_names = sample.get("class_names", self.class_names)
        positions = np.linspace(0, 1, num_classes) if num_classes > 1 else [0.5]

        legend_handles = [mpatches.Patch(color=cmap(pos), label=class_names[i]) for i, pos in enumerate(positions)]
        ax[legend_ax_index].legend(handles=legend_handles, loc="center")
        ax[legend_ax_index].axis("off")

        if suptitle:
            plt.suptitle(suptitle)

        plt.tight_layout()
        return fig
