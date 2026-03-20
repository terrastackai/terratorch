import glob
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A
import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from xarray import DataArray

from terratorch.datasets.utils import clip_image, default_transform, filter_valid_files, validate_bands


class Sen1Floods11NonGeo(NonGeoDataset):
    """NonGeo dataset implementation for [sen1floods11](https://github.com/cloudtostreet/Sen1Floods11)."""

    all_band_names = (
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        "CIRRUS",
        "SWIR_1",
        "SWIR_2",
    )
    rgb_bands = ("RED", "GREEN", "BLUE")
    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}
    num_classes = 2
    splits = {"train": "train", "val": "valid", "test": "test"}
    data_dir = "v1.1/data/flood_events/HandLabeled/S2Hand"
    label_dir = "v1.1/data/flood_events/HandLabeled/LabelHand"
    split_dir = "v1.1/splits/flood_handlabeled"
    metadata_file = "v1.1/Sen1Floods11_Metadata.geojson"

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        constant_scale: float = 0.0001,
        no_data_replace: float | None = 0,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Constructor

        Args:
            data_root (str): Path to the data root directory.
            split (str): one of 'train', 'val' or 'test'.
            bands (list[str]): Bands that should be output by the dataset. Defaults to all bands.
            transform (A.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). Defaults to None, which applies ToTensorV2().
            constant_scale (float): Factor to multiply image values by. Defaults to 0.0001.
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
        self.constant_scale = constant_scale
        self.data_root = Path(data_root)

        data_dir = self.data_root / self.data_dir
        label_dir = self.data_root / self.label_dir

        self.image_files = sorted(glob.glob(os.path.join(data_dir, "*_S2Hand.tif")))
        self.segmentation_mask_files = sorted(glob.glob(os.path.join(label_dir, "*_LabelHand.tif")))

        split_file = self.data_root / self.split_dir / f"flood_{split_name}_data.txt"
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
        self.use_metadata = use_metadata
        self.metadata = None
        if self.use_metadata:
            self.metadata = geopandas.read_file(self.data_root / self.metadata_file)

        # If no transform is given, apply only to transform to torch tensor
        self.transform = transform if transform else default_transform

    def __len__(self) -> int:
        return len(self.image_files)

    def _get_date(self, index: int) -> torch.Tensor:
        file_name = self.image_files[index]
        location = os.path.basename(file_name).split("_")[0]
        if self.metadata[self.metadata["location"] == location].shape[0] != 1:
            date = pd.to_datetime("13-10-1998", dayfirst=True)
        else:
            date = pd.to_datetime(self.metadata[self.metadata["location"] == location]["s2_date"].item())

        return torch.tensor([[date.year, date.dayofyear - 1]], dtype=torch.float32)  # (n_timesteps, coords)

    def _get_coords(self, image: DataArray) -> torch.Tensor:

        center_lat = image.y[image.y.shape[0] // 2]
        center_lon = image.x[image.x.shape[0] // 2]
        lat_lon = np.asarray([center_lat, center_lon])

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
            "image": image.astype(np.float32) * self.constant_scale,
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

        image = clip_image(image)

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
