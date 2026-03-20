import json
import pdb
from collections.abc import Sequence
from pathlib import Path

import albumentations as A
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import (
    clip_image,
    default_transform,
    validate_bands,
)


class MSACropTypeNonGeo(NonGeoDataset):
    """NonGeo dataset implementation for [M-SA-Crop-Type](https://github.com/ServiceNow/geo-bench?tab=readme-ov-file)."""

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
        "SWIR_1",
        "SWIR_2",
        "CLOUD_PROBABILITY",
    )

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    splits = {"train": "train", "val": "valid", "test": "test"}

    data_dir = "m-SA-crop-type"
    partition_file_template = "{partition}_partition.json"

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        partition: str = "default",
    ) -> None:
        """Initialize the dataset.

        Args:
            data_root (str): Path to the data root directory.
            split (str): One of 'train', 'val', or 'test'.
            bands (Sequence[str]): Bands to be used. Defaults to all bands.
            transform (A.Compose | None): Albumentations transform to be applied.
                Defaults to None, which applies default_transform().
            partition (str): Partition name for the dataset splits. Defaults to 'default'.
        """
        super().__init__()

        if split not in self.splits:
            msg = f"Incorrect split '{split}', please choose one of {list(self.splits.keys())}."
            raise ValueError(msg)
        split_name = self.splits[split]
        self.split = split

        validate_bands(bands, self.all_band_names)
        self.bands = bands
        self.band_indices = [self.all_band_names.index(b) for b in bands]

        self.data_root = Path(data_root)
        self.data_directory = self.data_root / self.data_dir

        partition_file = self.data_directory / self.partition_file_template.format(partition=partition)
        with open(partition_file) as file:
            partitions = json.load(file)

        if split_name not in partitions:
            msg = f"Split '{split_name}' not found."
            raise ValueError(msg)

        self.image_files = [self.data_directory / f"{filename}.hdf5" for filename in partitions[split_name]]

        self.transform = transform if transform else default_transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_path = self.image_files[index]

        with h5py.File(file_path, "r") as h5file:
            keys = sorted(h5file.keys())
            keys = np.array([key for key in keys if key != "label"])[self.band_indices]
            bands = [np.array(h5file[key]) for key in keys]

            image = np.stack(bands, axis=-1)
            mask = np.array(h5file["label"])

        output = {"image": image.astype(np.float32), "mask": mask}

        if self.transform:
            output = self.transform(**output)
        output["mask"] = output["mask"].long()
        return output

    def plot(self, sample: dict[str, torch.Tensor], suptitle: str | None = None) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample (dict[str, torch.Tensor]): A sample returned by :meth:`__getitem__`.
            suptitle (str | None): Optional string to use as a suptitle.

        Returns:
            matplotlib.figure.Figure: A matplotlib Figure with the rendered sample.
        """

        rgb_indices = [self.bands.index(band) for band in self.rgb_bands if band in self.bands]

        if len(rgb_indices) != 3:
            msg = "Dataset doesn't contain some of the RGB bands"
            raise ValueError(msg)

        image = sample["image"]
        mask = sample["mask"].numpy()
        if (len(mask.shape) == 3) & (mask.shape[0] == 1):
            mask = mask[0]

        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).numpy()

        rgb_image = image[:, :, rgb_indices]
        rgb_image = clip_image(rgb_image)

        num_classes = len(np.unique(mask))
        cmap = plt.get_cmap("jet")
        norm = plt.Normalize(vmin=0, vmax=num_classes - 1)

        num_images = 4 if "prediction" in sample else 3
        fig, ax = plt.subplots(1, num_images, figsize=(num_images * 4, 4), tight_layout=True)

        ax[0].imshow(rgb_image)
        ax[0].set_title("Image")
        ax[0].axis("off")

        ax[1].imshow(mask, cmap=cmap, norm=norm)
        ax[1].set_title("Ground Truth Mask")
        ax[1].axis("off")

        ax[2].imshow(rgb_image)
        ax[2].imshow(mask, cmap=cmap, alpha=0.3, norm=norm)
        ax[2].set_title("GT Mask on Image")
        ax[2].axis("off")

        if "prediction" in sample:
            prediction = sample["prediction"].numpy()
            ax[3].imshow(prediction, cmap=cmap, norm=norm)
            ax[3].set_title("Predicted Mask")
            ax[3].axis("off")

        if suptitle:
            plt.suptitle(suptitle)

        return fig
