from collections.abc import Sequence
from typing import Any

import albumentations as A
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import MultiTemporalCropClassification

MEANS = {
    "BLUE": 494.905781,
    "GREEN": 815.239594,
    "RED": 924.335066,
    "NIR_NARROW": 2968.881459,
    "SWIR_1": 2634.621962,
    "SWIR_2": 1739.579917,
}

STDS = {
    "BLUE": 284.925432,
    "GREEN": 357.84876,
    "RED": 575.566823,
    "NIR_NARROW": 896.601013,
    "SWIR_1": 951.900334,
    "SWIR_2": 921.407808,
}


class MultiTemporalCropClassificationDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule implementation for multi-temporal crop classification."""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        bands: Sequence[str] = MultiTemporalCropClassification.all_band_names,
        train_transform: list[Any] | None = None,
        val_transform: list[Any] | None = None,
        test_transform: list[Any] | None = None,
        predict_transform: A.Compose | None | list = None,
        drop_last: bool = True,
        no_data_replace: float | None = 0,
        no_label_replace: int | None = -1,
        expand_temporal_dimension: bool = True,
        reduce_zero_label: bool = True,
        use_metadata: bool = False,
        metadata_file_name: str = "chips_df.csv",
        **kwargs: Any,
    ) -> None:
        """
        Initializes the MultiTemporalCropClassificationDataModule for multi-temporal crop classification.

        Args:
            data_root (str): Directory containing the dataset.
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 4.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            bands (Sequence[str], optional): List of bands to use. Defaults to MultiTemporalCropClassification.all_band_names.
            train_transform (A.Compose | None | list, optional): Transformations for training data.
            val_transform (A.Compose | None | list, optional): Transformations for validation data.
            test_transform (A.Compose | None | list, optional): Transformations for testing data.
            predict_transform (A.Compose | None | list, optional): Transformations for prediction data.
            drop_last (bool, optional): Whether to drop the last incomplete batch during training. Defaults to True.
            no_data_replace (float | None, optional): Replacement value for missing data. Defaults to 0.
            no_label_replace (int | None, optional): Replacement value for missing labels. Defaults to -1.
            expand_temporal_dimension (bool, optional): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to True.
            reduce_zero_label (bool, optional): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to True.
            use_metadata (bool): Whether to return metadata info (time and location).
            **kwargs: Additional keyword arguments.
        """
        super().__init__(MultiTemporalCropClassification, batch_size, num_workers, **kwargs)
        self.data_root = data_root

        self.means = [MEANS[b] for b in bands]
        self.stds = [STDS[b] for b in bands]
        self.bands = bands
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)
        self.aug = Normalize(self.means, self.stds)
        self.drop_last = drop_last
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.expand_temporal_dimension = expand_temporal_dimension
        self.reduce_zero_label = reduce_zero_label
        self.use_metadata = use_metadata
        self.metadata_file_name = metadata_file_name

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
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
                use_metadata=self.use_metadata,
                metadata_file_name=self.metadata_file_name,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
                use_metadata=self.use_metadata,
                metadata_file_name=self.metadata_file_name,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
                use_metadata=self.use_metadata,
                metadata_file_name=self.metadata_file_name,
            )
        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.predict_transform,
                bands=self.bands,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
                use_metadata=self.use_metadata,
                metadata_file_name=self.metadata_file_name,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
        )
