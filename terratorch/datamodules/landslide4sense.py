from collections.abc import Sequence
from typing import Any

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from kornia.augmentation import AugmentationSequential
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.generic_multimodal_data_module import wrap_in_compose_is_list
from terratorch.datasets import Landslide4SenseNonGeo

MEANS = {
    "COASTAL AEROSOL": -0.4914,
    "BLUE": -0.3074,
    "GREEN": -0.1277,
    "RED": -0.0625,
    "RED_EDGE_1": 0.0439,
    "RED_EDGE_2": 0.0803,
    "RED_EDGE_3": 0.0644,
    "NIR_BROAD": 0.0802,
    "WATER_VAPOR": 0.3000,
    "CIRRUS": 0.4082,
    "SWIR_1": 0.0823,
    "SWIR_2": 0.0516,
    "SLOPE": 0.3338,
    "DEM": 0.7819,
}

STDS = {
    "COASTAL AEROSOL": 0.9325,
    "BLUE": 0.8775,
    "GREEN": 0.8860,
    "RED": 0.8869,
    "RED_EDGE_1": 0.8857,
    "RED_EDGE_2": 0.8418,
    "RED_EDGE_3": 0.8354,
    "NIR_BROAD": 0.8491,
    "WATER_VAPOR": 0.9061,
    "CIRRUS": 1.6072,
    "SWIR_1": 0.8848,
    "SWIR_2": 0.9232,
    "SLOPE": 0.9018,
    "DEM": 1.2913,
}


class Landslide4SenseNonGeoDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule implementation for Landslide4Sense dataset."""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        bands: Sequence[str] = Landslide4SenseNonGeo.all_band_names,
        train_transform: A.Compose | None | list = None,
        val_transform: A.Compose | None | list = None,
        test_transform: A.Compose | None | list = None,
        predict_transform: A.Compose | None | list = None,
        aug: AugmentationSequential = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Landslide4SenseNonGeoDataModule.

        Args:
            data_root (str): Root directory of the dataset.
            batch_size (int, optional): Batch size for data loaders. Defaults to 4.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            bands (Sequence[str], optional): List of band names to use. Defaults to Landslide4SenseNonGeo.all_band_names.
            train_transform (A.Compose | None | list, optional): Transformations for training data.
            val_transform (A.Compose | None | list, optional): Transformations for validation data.
            test_transform (A.Compose | None | list, optional): Transformations for testing data.
            predict_transform (A.Compose | None | list, optional): Transformations for prediction data.
            aug (AugmentationSequential, optional): Augmentation pipeline; if None, applies normalization using computed means and stds.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(Landslide4SenseNonGeo, batch_size, num_workers, **kwargs)
        self.data_root = data_root

        self.means = [MEANS[b] for b in bands]
        self.stds = [STDS[b] for b in bands]
        self.bands = bands
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)
        self.aug = AugmentationSequential(K.Normalize(self.means, self.stds), data_keys=None) if aug is None else aug

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either fit, validate, test, or predict.
        """
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train", data_root=self.data_root, transform=self.train_transform, bands=self.bands
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val", data_root=self.data_root, transform=self.val_transform, bands=self.bands
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test", data_root=self.data_root, transform=self.test_transform, bands=self.bands
            )
        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                split="test", data_root=self.data_root, transform=self.predict_transform, bands=self.bands
            )
