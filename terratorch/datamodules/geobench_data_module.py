from collections.abc import Sequence
from typing import Any

import albumentations as A
import kornia.augmentation as K  # noqa: N812
import torch
from kornia.augmentation import AugmentationSequential
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.utils import wrap_in_compose_is_list


class GeobenchDataModule(NonGeoDataModule):
    def __init__(
        self,
        dataset_class: type,
        means: dict[str, float],
        stds: dict[str, float],
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        bands: Sequence[str] | None = None,
        train_transform: A.Compose | None | list = None,
        val_transform: A.Compose | None | list = None,
        test_transform: A.Compose | None | list = None,
        predict_transform: A.Compose | None | list = None,
        aug: AugmentationSequential = None,
        partition: str = "default",
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset_class, batch_size, num_workers, **kwargs)

        self.bands = dataset_class.all_band_names if bands is None else bands
        self.means = torch.tensor([means[b] for b in self.bands])
        self.stds = torch.tensor([stds[b] for b in self.bands])
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)
        self.data_root = data_root
        self.partition = partition
        self.aug = AugmentationSequential(K.Normalize(self.means, self.stds), data_keys=None) if aug is None else aug

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                partition=self.partition,
                bands=self.bands,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                partition=self.partition,
                bands=self.bands,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                transform=self.test_transform,
                partition=self.partition,
                bands=self.bands,
                **self.kwargs,
            )
        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                transform=self.predict_transform,
                partition=self.partition,
                bands=self.bands,
                **self.kwargs,
            )
