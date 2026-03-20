from typing import Any

import albumentations as A  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import PASTIS


class PASTISDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule implementation for PASTIS."""

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        truncate_image: int | None = None,
        pad_image: int | None = None,
        train_transform: A.Compose | None | list = None,
        val_transform: A.Compose | None | list = None,
        test_transform: A.Compose | None | list = None,
        predict_transform: A.Compose | None | list = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the PASTISDataModule for the PASTIS dataset.

        Args:
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 8.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            data_root (str, optional): Directory containing the dataset. Defaults to "./".
            truncate_image (int, optional): Truncate the time dimension of the image to
                a specified number of timesteps. If None, no truncation is performed.
            pad_image (int, optional): Pad the time dimension of the image to a specified
                number of timesteps. If None, no padding is applied.
            train_transform (A.Compose | None | list, optional): Transformations for training data.
            val_transform (A.Compose | None | list, optional): Transformations for validation data.
            test_transform (A.Compose | None | list, optional): Transformations for testing data.
            predict_transform (A.Compose | None | list, optional): Transformations for prediction data.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            PASTIS,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.truncate_image = truncate_image
        self.pad_image = pad_image
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)
        self.data_root = data_root
        self.kwargs = kwargs

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either fit, validate, test, or predict.
        """
        if stage in ["fit"]:
            self.train_dataset = PASTIS(
                folds=[1, 2, 3],
                data_root=self.data_root,
                transform=self.train_transform,
                truncate_image=self.truncate_image,
                pad_image=self.pad_image,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = PASTIS(
                folds=[4],
                data_root=self.data_root,
                transform=self.val_transform,
                truncate_image=self.truncate_image,
                pad_image=self.pad_image,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = PASTIS(
                folds=[5],
                data_root=self.data_root,
                transform=self.test_transform,
                truncate_image=self.truncate_image,
                pad_image=self.pad_image,
                **self.kwargs,
            )
        if stage in ["predict"]:
            self.predict_dataset = PASTIS(
                folds=[5],
                data_root=self.data_root,
                transform=self.predict_transform,
                truncate_image=self.truncate_image,
                pad_image=self.pad_image,
                **self.kwargs,
            )
