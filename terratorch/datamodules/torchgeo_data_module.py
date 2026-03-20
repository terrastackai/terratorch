# Copyright contributors to the Terratorch project

"""Ugly proxy objects so parsing config file works with transforms.

These are necessary since, for LightningCLI to instantiate arguments as
objects from the config, they must have type annotations

In TorchGeo, `transforms` is passed in **kwargs, so it has no type annotations!
To get around that, we create these wrappers that have transforms type annotated.
They create the transforms and forward all method and attribute calls to the
original TorchGeo datamodule.

Additionally, TorchGeo datasets pass the data to the transforms callable
as a dict, and as a tensor.

Albumentations expects this data not as a dict but as different key-value
arguments, and as numpy. We handle that conversion here.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from albumentations import BasicTransform
from torch import Tensor
from torchgeo.datamodules import GeoDataModule, NonGeoDataModule

from terratorch.datasets.transforms import albumentations_to_callable_with_dict

ALBUMENTATIONS_TARGETS = ["image", "mask"]


def build_callable_transform_from_torch_tensor(
    callable_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]],
) -> Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    # Take a function from dicts of str->Tensor to dicts of str->Tensor
    # Return a function from dicts of str->np to dicts of str->np
    # Additionally permute the image back to channels last

    def transforms_from_torch_tensor(tensor_dict: dict[str, Tensor]):
        numpy_dict = {k: (v.numpy() if k in ALBUMENTATIONS_TARGETS else v) for k, v in tensor_dict.items()}
        numpy_dict["image"] = np.moveaxis(numpy_dict["image"], 0, -1)  # image to channels last
        return callable_transform(numpy_dict)

    return transforms_from_torch_tensor


class TorchNonGeoDataModule(NonGeoDataModule):
    """Proxy object for using NonGeo data modules defined by TorchGeo.

    Allows for transforms to be defined and passed using config files.
    The only reason this class exists is so that we can annotate the transforms argument with a type.
    This is required for lightningcli and config files.
    As such, all getattr and setattr will be redirected to the underlying class.
    """

    def __init__(
        self,
        cls: type[NonGeoDataModule],
        batch_size: int | None = None,
        num_workers: int = 0,
        transforms: None | list[Any] = None,
        **kwargs: Any,
    ):
        """Constructor

        Args:
            cls (type[NonGeoDataModule]): TorchGeo DataModule class to be instantiated
            batch_size (int | None, optional): batch_size. Defaults to None.
            num_workers (int, optional): num_workers. Defaults to 0.
            transforms (None | list[BasicTransform], optional): List of Albumentations Transforms.
                Should enc with ToTensorV2. Defaults to None.
            **kwargs (Any): Arguments passed to instantiate `cls`.
        """
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if transforms is not None:
            transforms_as_callable = albumentations_to_callable_with_dict(transforms)
            kwargs["transforms"] = build_callable_transform_from_torch_tensor(transforms_as_callable)
        # self.__dict__["datamodule"] = cls(num_workers=num_workers, **kwargs)
        self._proxy = cls(num_workers=num_workers, **kwargs)
        super().__init__(self._proxy.dataset_class)  # dummy arg

    @property
    def collate_fn(self):
        return self._proxy.collate_fn

    @collate_fn.setter
    def collate_fn(self, value):
        self._proxy.collate_fn = value

    def setup(self, stage: str):
        return self._proxy.setup(stage)

    def train_dataloader(self):
        return self._proxy.train_dataloader()

    def val_dataloader(self):
        return self._proxy.val_dataloader()

    def test_dataloader(self):
        return self._proxy.test_dataloader()

    def predict_dataloader(self):
        return self._proxy.predict_dataloader()


class TorchGeoDataModule(GeoDataModule):
    """Proxy object for using Geo data modules defined by TorchGeo.

    Allows for transforms to be defined and passed using config files.
    The only reason this class exists is so that we can annotate the transforms argument with a type.
    This is required for lightningcli and config files.
    As such, all getattr and setattr will be redirected to the underlying class.
    """

    def __init__(
        self,
        cls: type[GeoDataModule],
        batch_size: int | None = None,
        num_workers: int = 0,
        transforms: None | list[Any] = None,
        **kwargs: Any,
    ):
        """Constructor

        Args:
            cls (type[GeoDataModule]): TorchGeo DataModule class to be instantiated
            batch_size (int | None, optional): batch_size. Defaults to None.
            num_workers (int, optional): num_workers. Defaults to 0.
            transforms (None | list[BasicTransform], optional): List of Albumentations Transforms.
                Should enc with ToTensorV2. Defaults to None.
            **kwargs (Any): Arguments passed to instantiate `cls`.
        """
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if transforms is not None:
            transforms_as_callable = albumentations_to_callable_with_dict(transforms)
            kwargs["transforms"] = build_callable_transform_from_torch_tensor(transforms_as_callable)
        # self.__dict__["datamodule"] = cls(num_workers=num_workers, **kwargs)
        self._proxy = cls(num_workers=num_workers, **kwargs)
        super().__init__(self._proxy.dataset_class)  # dummy arg

    @property
    def collate_fn(self):
        return self._proxy.collate_fn

    @collate_fn.setter
    def collate_fn(self, value):
        self._proxy.collate_fn = value

    @property
    def patch_size(self):
        return self._proxy.patch_size

    @property
    def length(self):
        return self._proxy.length

    def setup(self, stage: str):
        return self._proxy.setup(stage)

    def train_dataloader(self):
        return self._proxy.train_dataloader()

    def val_dataloader(self):
        return self._proxy.val_dataloader()

    def test_dataloader(self):
        return self._proxy.test_dataloader()

    def predict_dataloader(self):
        return self._proxy.predict_dataloader()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return self._proxy.predict_dataloader(batch, device, dataloader_idx)
