from typing import Any

import albumentations as A  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import OpenSentinelMap


class OpenSentinelMapDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule implementation for Open Sentinel Map."""

    def __init__(
        self,
        bands: list[str] | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        train_transform: A.Compose | None | list = None,
        val_transform: A.Compose | None | list = None,
        test_transform: A.Compose | None | list = None,
        predict_transform: A.Compose | None | list = None,
        spatial_interpolate_and_stack_temporally: bool = True,  # noqa: FBT001, FBT002
        pad_image: int | None = None,
        truncate_image: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the OpenSentinelMapDataModule for the Open Sentinel Map dataset.

        Args:
            bands (list[str] | None, optional): List of bands to use. Defaults to None.
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 8.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            data_root (str, optional): Root directory of the dataset. Defaults to "./".
            train_transform (A.Compose | None | list, optional): Transformations for training data.
            val_transform (A.Compose | None | list, optional): Transformations for validation data.
            test_transform (A.Compose | None | list, optional): Transformations for testing data.
            predict_transform (A.Compose | None | list, optional): Transformations for prediction data.
            spatial_interpolate_and_stack_temporally (bool, optional): If True, the bands are interpolated and concatenated over time.
                Default is True.
            pad_image (int | None, optional): Number of timesteps to pad the time dimension of the image.
                If None, no padding is applied.
            truncate_image (int | None, optional):  Number of timesteps to truncate the time dimension of the image.
                If None, no truncation is performed.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            OpenSentinelMap,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.bands = bands
        self.spatial_interpolate_and_stack_temporally = spatial_interpolate_and_stack_temporally
        self.pad_image = pad_image
        self.truncate_image = truncate_image
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
            self.train_dataset = OpenSentinelMap(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                bands=self.bands,
                spatial_interpolate_and_stack_temporally=self.spatial_interpolate_and_stack_temporally,
                pad_image=self.pad_image,
                truncate_image=self.truncate_image,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = OpenSentinelMap(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                spatial_interpolate_and_stack_temporally=self.spatial_interpolate_and_stack_temporally,
                pad_image=self.pad_image,
                truncate_image=self.truncate_image,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = OpenSentinelMap(
                split="test",
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                spatial_interpolate_and_stack_temporally=self.spatial_interpolate_and_stack_temporally,
                pad_image=self.pad_image,
                truncate_image=self.truncate_image,
                **self.kwargs,
            )
        if stage in ["predict"]:
            self.predict_dataset = OpenSentinelMap(
                split="test",
                data_root=self.data_root,
                transform=self.predict_transform,
                bands=self.bands,
                spatial_interpolate_and_stack_temporally=self.spatial_interpolate_and_stack_temporally,
                pad_image=self.pad_image,
                truncate_image=self.truncate_image,
                **self.kwargs,
            )
