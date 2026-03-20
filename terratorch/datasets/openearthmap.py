from collections.abc import Sequence
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import to_tensor


class OpenEarthMapNonGeo(NonGeoDataset):
    """
    [OpenEarthMapNonGeo](https://open-earth-map.org/) Dataset for non-georeferenced imagery.

    This dataset class handles non-georeferenced image data from the OpenEarthMap dataset.
    It supports configurable band sets and transformations, and performs cropping operations
    to ensure that the images conform to the required input dimensions. The dataset is split
    into "train", "test", and "val" subsets based on the provided split parameter.
    """

    all_band_names = ("BLUE", "GREEN", "RED")

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    def __init__(
        self,
        data_root: str,
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        split="train",
        crop_size: int = 256,
        random_crop: bool = True,
    ) -> None:
        """
        Initialize a new instance of the OpenEarthMapNonGeo dataset.

        Args:
            data_root (str): The root directory containing the dataset files.
            bands (Sequence[str], optional): A list of band names to be used. Default is BAND_SETS["all"].
            transform (A.Compose or None, optional): A transformation pipeline to be applied to the data.
                If None, a default transform converting the data to a tensor is applied.
            split (str, optional): The dataset split to use ("train", "test", or "val"). Default is "train".
            crop_size (int, optional): The size (in pixels) of the crop to apply to images. Must be greater than 0.
                Default is 256.
            random_crop (bool, optional): If True, performs a random crop; otherwise, performs a center crop.
                Default is True.

        Raises:
            Exception: If the provided split is not one of "train", "test", or "val".
            AssertionError: If crop_size is not greater than 0.
        """
        super().__init__()
        if split not in ["train", "test", "val"]:
            msg = "Split must be one of train, test, val."
            raise Exception(msg)

        self.transform = transform if transform else lambda **batch: to_tensor(batch, transpose=False)
        self.split = split
        self.data_root = data_root

        # images in openearthmap are not all 1024x1024 and must be cropped
        self.crop_size = crop_size
        self.random_crop = random_crop

        assert self.crop_size > 0, "Crop size must be greater than 0"

        self.image_files = self._get_file_paths(Path(self.data_root, f"{split}.txt"))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, label_path = self.image_files[index]

        with rasterio.open(image_path) as src:
            image = src.read()
        with rasterio.open(label_path) as src:
            mask = src.read()

        # some images in the dataset are not perfect squares
        # cropping to fit to the prepare_features_for_image_model call
        if self.random_crop:
            image, mask = self._random_crop(image, mask)
        else:
            image, mask = self._center_crop(image, mask)

        output = {"image": image.astype(np.float32), "mask": mask}

        output = self.transform(**output)
        output["mask"] = output["mask"].long()

        return output

    def _parse_file_name(self, file_name: str):
        underscore_pos = file_name.rfind("_")
        folder_name = file_name[:underscore_pos]
        region_path = Path(self.data_root, folder_name)
        image_path = Path(region_path, "images", file_name)
        label_path = Path(region_path, "labels", file_name)
        return image_path, label_path

    def _get_file_paths(self, text_file_path: str):
        with open(text_file_path) as file:
            lines = file.readlines()
            file_paths = [self._parse_file_name(line.strip()) for line in lines]
        return file_paths

    def __len__(self):
        return len(self.image_files)

    def _random_crop(self, image, mask):
        h, w = image.shape[1:]
        top = np.random.randint(0, h - self.crop_size)
        left = np.random.randint(0, w - self.crop_size)

        image = image[:, top : top + self.crop_size, left : left + self.crop_size]
        mask = mask[:, top : top + self.crop_size, left : left + self.crop_size]

        return image, mask

    def _center_crop(self, image, mask):
        h, w = image.shape[1:]
        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2

        image = image[:, top : top + self.crop_size, left : left + self.crop_size]
        mask = mask[:, top : top + self.crop_size, left : left + self.crop_size]

        return image, mask

    def plot(self, arg, suptitle: str | None = None) -> None:
        pass

    def plot_sample(self, sample, prediction=None, suptitle: str | None = None, class_names=None):
        pass
