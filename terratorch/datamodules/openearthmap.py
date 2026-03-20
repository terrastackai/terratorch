from typing import Any

import albumentations as A
import kornia.augmentation as K
import torch
from kornia.augmentation import AugmentationSequential
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import OpenEarthMapNonGeo

MEANS = {"BLUE": 116.628328, "GREEN": 119.65935, "RED": 113.385309}

STDS = {
    "BLUE": 44.668890717415586,
    "GREEN": 48.282311849967364,
    "RED": 54.19692448815262,
}


class OpenEarthMapNonGeoDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule implementation for Open Earth Map."""

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        train_transform: A.Compose | None | list = None,
        val_transform: A.Compose | None | list = None,
        test_transform: A.Compose | None | list = None,
        predict_transform: A.Compose | None | list = None,
        aug: AugmentationSequential = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the OpenEarthMapNonGeoDataModule for the Open Earth Map dataset.

        Args:
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 8.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            data_root (str, optional): Root directory of the dataset. Defaults to "./".
            train_transform (A.Compose | None | list, optional): Transformations for training data.
            val_transform (A.Compose | None | list, optional): Transformations for validation data.
            test_transform (A.Compose | None | list, optional): Transformations for test data.
            predict_transform (A.Compose | None | list, optional): Transformations for prediction data.
            aug (AugmentationSequential, optional): Augmentation pipeline; if None, defaults to normalization using computed means and stds.
            **kwargs: Additional keyword arguments. Can include 'bands' (list[str]) to specify the bands; defaults to OpenEarthMapNonGeo.all_band_names if not provided.
        """
        super().__init__(OpenEarthMapNonGeo, batch_size, num_workers, **kwargs)

        bands = kwargs.get("bands", OpenEarthMapNonGeo.all_band_names)
        self.means = torch.tensor([MEANS[b] for b in bands])
        self.stds = torch.tensor([STDS[b] for b in bands])
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)
        self.data_root = data_root
        self.aug = AugmentationSequential(K.Normalize(self.means, self.stds), data_keys=None) if aug is None else aug

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either fit, validate, test, or predict.
        """
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train", data_root=self.data_root, transform=self.train_transform, **self.kwargs
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val", data_root=self.data_root, transform=self.val_transform, **self.kwargs
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test", data_root=self.data_root, transform=self.test_transform, **self.kwargs
            )
        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                split="test", data_root=self.data_root, transform=self.predict_transform, **self.kwargs
            )
