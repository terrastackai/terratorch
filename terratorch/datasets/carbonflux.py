import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rioxarray
import torch
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.generic_multimodal_dataset import MultimodalToTensor
from terratorch.datasets.transforms import MultimodalTransforms
from terratorch.datasets.utils import default_transform, validate_bands


class CarbonFluxNonGeo(NonGeoDataset):
    """Dataset for [Carbon Flux](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_merra2_gppFlux) regression from HLS images and MERRA data."""

    all_band_names = (
        "BLUE",
        "GREEN",
        "RED",
        "NIR",
        "SWIR_1",
        "SWIR_2",
    )

    rgb_bands = (
        "RED",
        "GREEN",
        "BLUE",
    )

    merra_var_names = (
        "T2MIN",
        "T2MAX",
        "T2MEAN",
        "TSMDEWMEAN",
        "GWETROOT",
        "LHLAND",
        "SHLAND",
        "SWLAND",
        "PARDFLAND",
        "PRECTOTLAND",
    )

    splits = {"train": "train", "test": "test"}

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    metadata_file = "data_train_hls_37sites_v0_1.csv"

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        gpp_mean: float | None = None,
        gpp_std: float | None = None,
        no_data_replace: float | None = 0.0001,
        use_metadata: bool = False,
        modalities: Sequence[str] = ("image", "merra_vars"),
    ) -> None:
        """Initialize the CarbonFluxNonGeo dataset.

        Args:
            data_root (str): Path to the data root directory.
            split (str): 'train' or 'test'.
            bands (Sequence[str]): Bands to use. Defaults to all bands.
            transform (Optional[A.Compose]): Albumentations transform to be applied.
            use_metadata (bool): Whether to return metadata (coordinates and date).
            merra_means (Sequence[float]): Means for MERRA data normalization.
            merra_stds (Sequence[float]): Standard deviations for MERRA data normalization.
            gpp_mean (float): Mean for GPP normalization.
            gpp_std (float): Standard deviation for GPP normalization.
            no_data_replace (Optional[float]): Value to replace NO_DATA values in images.
        """
        super().__init__()
        if split not in self.splits:
            msg = f"Incorrect split '{split}', please choose one of {list(self.splits.keys())}."
            raise ValueError(msg)

        self.split = split

        validate_bands(bands, self.all_band_names)
        self.bands = bands
        self.band_indices = [self.all_band_names.index(band) for band in bands]

        self.data_root = Path(data_root)

        # Load the CSV file with metadata
        csv_file = self.data_root / self.metadata_file
        df = pd.read_csv(csv_file)

        # Get list of image filenames in the split directory
        image_dir = self.data_root / self.split
        image_files = [f.name for f in image_dir.glob("*.tiff")]

        df["Chip"] = df["Chip"].str.replace(".tif$", ".tiff", regex=True)
        # Filter the DataFrame to include only rows with 'Chip' in image_files
        df = df[df["Chip"].isin(image_files)]

        # Build the samples list
        self.samples = []
        for _, row in df.iterrows():
            image_filename = row["Chip"]
            image_path = image_dir / image_filename
            # MERRA vectors
            merra_vars = row[list(self.merra_var_names)].values.astype(np.float32)
            # GPP target
            gpp = row["GPP"]

            image_path = image_dir / row["Chip"]
            merra_vars = row[list(self.merra_var_names)].values.astype(np.float32)
            gpp = row["GPP"]
            self.samples.append(
                {
                    "image_path": str(image_path),
                    "merra_vars": merra_vars,
                    "gpp": gpp,
                }
            )

        if gpp_mean is None or gpp_std is None:
            msg = "Mean and standard deviation for GPP must be provided."
            raise ValueError(msg)
        self.gpp_mean = gpp_mean
        self.gpp_std = gpp_std

        self.use_metadata = use_metadata
        self.modalities = modalities
        self.no_data_replace = no_data_replace

        if transform is None:
            self.transform = MultimodalToTensor(self.modalities)
        else:
            transform = {m: transform[m] if m in transform else default_transform for m in self.modalities}
            self.transform = MultimodalTransforms(transform, shared=False)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_file(self, path: str, nan_replace: float | int | None = None):
        data = rioxarray.open_rasterio(path, masked=True)
        if nan_replace is not None:
            data = data.fillna(nan_replace)
        return data

    def _get_coords(self, image) -> torch.Tensor:
        """Extract the center coordinates from the image geospatial metadata."""
        pixel_scale = image.rio.resolution()
        width, height = image.rio.width, image.rio.height

        left, bottom, right, top = image.rio.bounds()
        tie_point_x, tie_point_y = left, top

        center_col = width / 2
        center_row = height / 2

        center_lon = tie_point_x + (center_col * pixel_scale[0])
        center_lat = tie_point_y - (center_row * pixel_scale[1])

        src_crs = image.rio.crs
        dst_crs = "EPSG:4326"

        transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        lon, lat = transformer.transform(center_lon, center_lat)

        coords = np.array([lat, lon], dtype=np.float32)
        return torch.from_numpy(coords)

    def _get_date(self, filename: str) -> torch.Tensor:
        """Extract the date from the filename."""
        base_filename = os.path.basename(filename)
        pattern = r"HLS\..{3}\.[A-Z0-9]{6}\.(?P<date>\d{7}T\d{6})\..*\.tiff$"
        match = re.match(pattern, base_filename)
        if not match:
            msg = f"Filename {filename} does not match expected pattern."
            raise ValueError(msg)

        date_str = match.group("date")
        year = int(date_str[:4])
        julian_day = int(date_str[4:7])

        date_tensor = torch.tensor([year, julian_day], dtype=torch.int32)
        return date_tensor

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_path = sample["image_path"]

        image = self._load_file(image_path, nan_replace=self.no_data_replace)

        if self.use_metadata:
            location_coords = self._get_coords(image)
            temporal_coords = self._get_date(os.path.basename(image_path))

        image = image.to_numpy()  # (C, H, W)
        image = image[self.band_indices, ...]
        image = np.moveaxis(image, 0, -1)  # (H, W, C)

        merra_vars = np.array(sample["merra_vars"])
        target = np.array(sample["gpp"])
        target_norm = (target - self.gpp_mean) / self.gpp_std
        target_norm = torch.tensor(target_norm, dtype=torch.float32)
        output = {
            "image": image.astype(np.float32),
            "merra_vars": merra_vars,
        }

        if self.transform:
            output = self.transform(output)

        output = {"image": {m: output[m] for m in self.modalities if m in output}, "mask": target_norm}
        if self.use_metadata:
            output["location_coords"] = location_coords
            output["temporal_coords"] = temporal_coords

        return output

    def plot(self, sample: dict[str, Any], suptitle: str | None = None) -> Any:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by `__getitem__`.
            suptitle: Optional title for the figure.

        Returns:
            A matplotlib figure with the rendered sample.
        """
        image = sample["image"].numpy()

        image = np.transpose(image, (1, 2, 0))  # (H, W, C)

        rgb_indices = [self.bands.index(band) for band in self.rgb_bands if band in self.bands]
        if len(rgb_indices) != 3:
            msg = "Dataset doesn't contain some of the RGB bands"
            raise ValueError(msg)

        rgb_image = image[..., rgb_indices]

        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
        rgb_image = np.clip(rgb_image, 0, 1)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(rgb_image)
        ax.axis("off")
        ax.set_title("Image")

        if suptitle:
            plt.suptitle(suptitle)

        plt.tight_layout()
        return fig
