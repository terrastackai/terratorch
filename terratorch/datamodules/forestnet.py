from collections.abc import Sequence
from typing import Any

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from kornia.augmentation import AugmentationSequential
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.generic_multimodal_data_module import wrap_in_compose_is_list
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datasets import ForestNetNonGeo

MEANS = {"BLUE": 19.8680, "GREEN": 28.1656, "RED": 14.9309, "NIR": 82.1076, "SWIR_1": 39.4819, "SWIR_2": 17.7241}

STDS = {"BLUE": 17.4523, "GREEN": 15.8399, "RED": 17.9444, "NIR": 21.4439, "SWIR_1": 14.4642, "SWIR_2": 9.9120}


class ForestNetNonGeoDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule implementation for Landslide4Sense dataset."""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        label_map: dict[str, int] = ForestNetNonGeo.default_label_map,
        bands: Sequence[str] = ForestNetNonGeo.all_band_names,
        train_transform: A.Compose | None | list = None,
        val_transform: A.Compose | None | list = None,
        test_transform: A.Compose | None | list = None,
        predict_transform: A.Compose | None | list = None,
        fraction: float = 1.0,
        aug: AugmentationSequential = None,
        use_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the ForestNetNonGeoDataModule.

        Args:
            data_root (str): Directory containing the dataset.
            batch_size (int, optional): Batch size for data loaders. Defaults to 4.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            label_map (dict[str, int], optional): Mapping of labels to integers. Defaults to ForestNetNonGeo.default_label_map.
            bands (Sequence[str], optional): List of band names to use. Defaults to ForestNetNonGeo.all_band_names.
            train_transform (A.Compose | None | list, optional): Transformations for training data.
            val_transform (A.Compose | None | list, optional): Transformations for validation data.
            test_transform (A.Compose | None | list, optional): Transformations for testing data.
            predict_transform (A.Compose | None | list, optional): Transformations for prediction.
            fraction (float, optional): Fraction of data to use. Defaults to 1.0.
            aug (AugmentationSequential, optional): Augmentation/normalization pipeline; if None, uses Normalize.
            use_metadata (bool): Whether to return metadata info.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(ForestNetNonGeo, batch_size, num_workers, **kwargs)
        self.data_root = data_root

        self.means = [MEANS[b] for b in bands]
        self.stds = [STDS[b] for b in bands]
        self.label_map = label_map
        self.bands = bands
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)
        self.aug = Normalize(self.means, self.stds) if aug is None else aug
        self.fraction = fraction
        self.use_metadata = use_metadata

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either fit, validate, test, or predict.
        """
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                data_root=self.data_root,
                label_map=self.label_map,
                transform=self.train_transform,
                bands=self.bands,
                fraction=self.fraction,
                use_metadata=self.use_metadata,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                label_map=self.label_map,
                transform=self.val_transform,
                bands=self.bands,
                fraction=self.fraction,
                use_metadata=self.use_metadata,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                label_map=self.label_map,
                transform=self.test_transform,
                bands=self.bands,
                fraction=self.fraction,
                use_metadata=self.use_metadata,
            )
        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                label_map=self.label_map,
                transform=self.predict_transform,
                bands=self.bands,
                fraction=self.fraction,
                use_metadata=self.use_metadata,
            )
