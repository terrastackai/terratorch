from collections.abc import Sequence
from typing import Any

from pathlib import Path

import albumentations as A
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
import logging

from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import CDLMultiTemporalCropClassification
from terratorch.io.file import load_from_file_or_attribute

from .utils import check_dataset_stackability

logger = logging.getLogger("terratorch")

L2ABANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12"
]


class CDLMultiTemporalCropClassificationDataModule(NonGeoDataModule):
    """
    NonGeo LightningDataModule implementation for CDL multi-temporal crop classification
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 2,
        num_classes: int = 6,
        n_timesteps: int = 2,
        train_data_root: Path | None = None,
        val_data_root: Path | None = None,
        test_data_root: Path | None = None,
        img_grep: str = "*",
        label_grep: str = "*",
        means: list[float] | None = None,
        stds: list[float] | None = None,
        predict_data_root: Path | None = None,
        train_label_data_root: Path | None = None,
        val_label_data_root: Path | None = None,
        test_label_data_root: Path | None = None,
        train_split: Path | None = None,
        val_split: Path | None = None,
        test_split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        dataset_bands: list[int | str] = L2ABANDS,
        output_bands: list[int | str] = L2ABANDS,
        predict_dataset_bands: list[int | str] = L2ABANDS,
        predict_output_bands: list[int | str] = L2ABANDS,
        constant_scale: float = 1,
        rgb_indices: list[int] = [3, 2, 1],
        train_transform: list[Any] | None = None,
        val_transform: list[Any] | None = None,
        test_transform: list[Any] | None = None,
        expand_temporal_dimension: bool = True,
        reduce_zero_label: bool = True,
        no_data_replace: float | None = None,
        no_label_replace: float | None = None,
        drop_last: bool = True,
        pin_memory: bool = False,
        check_stackability: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(CDLMultiTemporalCropClassification, batch_size, num_workers, **kwargs)
        self.num_classes = num_classes
        self.n_timesteps = n_timesteps
        self.img_grep = img_grep
        self.label_grep = label_grep
        self.train_root = train_data_root
        self.val_root = val_data_root
        self.test_root = test_data_root
        self.predict_root = predict_data_root
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.ignore_split_file_extensions = ignore_split_file_extensions
        self.allow_substring_split_file = allow_substring_split_file
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.drop_last = drop_last

        self.train_label_data_root = train_label_data_root
        self.val_label_data_root = val_label_data_root
        self.test_label_data_root = test_label_data_root

        self.dataset_bands = dataset_bands
        self.predict_dataset_bands = predict_dataset_bands if predict_dataset_bands else dataset_bands
        self.predict_output_bands = predict_output_bands if predict_output_bands else output_bands
        self.output_bands = output_bands
        self.rgb_indices = rgb_indices
        self.expand_temporal_dimension = expand_temporal_dimension
        self.reduce_zero_label = reduce_zero_label

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        
        self.check_stackability = check_stackability
        self.pin_memory = pin_memory

        if means and stds:
            means = load_from_file_or_attribute(means)
            stds = load_from_file_or_attribute(stds)

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                self.train_root,
                self.num_classes,
                self.n_timesteps,
                label_data_root=self.train_label_data_root,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                split=self.train_split,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                transform=self.train_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                self.val_root,
                self.num_classes,
                self.n_timesteps,
                label_data_root=self.val_label_data_root,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                split=self.val_split,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                transform=self.val_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                self.test_root,
                self.num_classes,
                self.n_timesteps,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                split=self.test_split,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label
            )
        if stage in ["predict"] and self.predict_root:
            self.predict_dataset = self.dataset_class(
                self.predict_root,
                self.num_classes,
                self.n_timesteps,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                dataset_bands=self.predict_dataset_bands,
                output_bands=self.predict_output_bands,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label
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

        if self.check_stackability:
            logger.info("Checking stackability.")
            batch_size = check_dataset_stackability(dataset, batch_size)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
            pin_memory=self.pin_memory,
        )