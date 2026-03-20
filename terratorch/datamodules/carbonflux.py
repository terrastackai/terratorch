from collections.abc import Sequence
from typing import Any

import albumentations as A
from kornia.augmentation import AugmentationSequential
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.generic_multimodal_data_module import MultimodalNormalize, wrap_in_compose_is_list
from terratorch.datasets import CarbonFluxNonGeo

MEANS = {
    "image": {
        "BLUE": 0.07372144372093026,
        "GREEN": 0.10117611215116282,
        "RED": 0.11269885680232558,
        "NIR": 0.2775572554069766,
        "SWIR_1": 0.21387001372093037,
        "SWIR_2": 0.14144541145348838,
    },
    "merra_vars": [
        282.373169,
        296.706468,
        288.852922,
        278.612209,
        0.540145,
        53.830276,
        53.827718,
        206.817980,
        23.077581,
        0.000003,
    ],
    "mask": 3.668982,
}

STDS = {
    "image": {
        "BLUE": 0.13324302628303733,
        "GREEN": 0.13308921403475235,
        "RED": 0.13829909331863693,
        "NIR": 0.12039809083338567,
        "SWIR_1": 0.1088096350639653,
        "SWIR_2": 0.09366368859284444,
    },
    "merra_vars": [
        9.296960,
        11.402008,
        10.311107,
        8.064209,
        0.171909,
        49.945953,
        48.907351,
        74.591578,
        8.746668,
        0.000014,
    ],
    "mask": 3.804261,
}


class CarbonFluxNonGeoDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule implementation for Carbon FLux dataset."""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        bands: Sequence[str] = CarbonFluxNonGeo.all_band_names,
        train_transform: A.Compose | None | list = None,
        val_transform: A.Compose | None | list = None,
        test_transform: A.Compose | None | list = None,
        predict_transform: A.Compose | None | list = None,
        aug: AugmentationSequential = None,
        no_data_replace: float | None = 0.0001,
        use_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the CarbonFluxNonGeoDataModule.

        Args:
            data_root (str): Root directory of the dataset.
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 4.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            bands (Sequence[str], optional): List of bands to use. Defaults to CarbonFluxNonGeo.all_band_names.
            train_transform (A.Compose | None | list, optional): Transformations for training data.
            val_transform (A.Compose | None | list, optional): Transformations for validation data.
            test_transform (A.Compose | None | list, optional): Transformations for testing data.
            predict_transform (A.Compose | None | list, optional): Transformations for prediction data.
            aug (AugmentationSequential, optional): Augmentation sequence; if None, applies multimodal normalization.
            no_data_replace (float | None, optional): Value to replace missing data. Defaults to 0.0001.
            use_metadata (bool): Whether to return metadata info.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(CarbonFluxNonGeo, batch_size, num_workers, **kwargs)
        self.data_root = data_root

        means = {m: ([MEANS[m][band] for band in bands] if m == "image" else MEANS[m]) for m in MEANS.keys()}
        stds = {m: ([STDS[m][band] for band in bands] if m == "image" else STDS[m]) for m in STDS.keys()}
        self.mask_means = MEANS["mask"]
        self.mask_std = STDS["mask"]
        self.bands = bands
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)
        self.aug = MultimodalNormalize(means, stds) if aug is None else aug
        self.no_data_replace = no_data_replace
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
                transform=self.train_transform,
                bands=self.bands,
                gpp_mean=self.mask_means,
                gpp_std=self.mask_std,
                no_data_replace=self.no_data_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                gpp_mean=self.mask_means,
                gpp_std=self.mask_std,
                no_data_replace=self.no_data_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                gpp_mean=self.mask_means,
                gpp_std=self.mask_std,
                no_data_replace=self.no_data_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                transform=self.predict_transform,
                bands=self.bands,
                gpp_mean=self.mask_means,
                gpp_std=self.mask_std,
                no_data_replace=self.no_data_replace,
                use_metadata=self.use_metadata,
            )
