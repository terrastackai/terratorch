from typing import Any

import albumentations as A  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import Sen4AgriNet


class Sen4AgriNetDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule implementation for Sen4AgriNet."""

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
        seed: int = 42,
        scenario: str = "random",
        requires_norm: bool = True,
        binary_labels: bool = False,
        linear_encoder: dict = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Sen4AgriNetDataModule for the Sen4AgriNet dataset.

        Args:
            bands (list[str] | None, optional): List of bands to use. Defaults to None.
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 8.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            data_root (str, optional): Root directory of the dataset. Defaults to "./".
            train_transform (A.Compose | None | list, optional): Transformations for training data.
            val_transform (A.Compose | None | list, optional): Transformations for validation data.
            test_transform (A.Compose | None | list, optional): Transformations for test data.
            predict_transform (A.Compose | None | list, optional): Transformations for prediction data.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            scenario (str): Defines the splitting scenario to use. Options are:
                - 'random': Random split of the data.
                - 'spatial': Split by geographical regions (Catalonia and France).
                - 'spatio-temporal': Split by region and year (France 2019 and Catalonia 2020).
            requires_norm (bool, optional): Whether normalization is required. Defaults to True.
            binary_labels (bool, optional): Whether to use binary labels. Defaults to False.
            linear_encoder (dict, optional): Mapping for label encoding. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            Sen4AgriNet,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.bands = bands
        self.seed = seed
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)
        self.data_root = data_root
        self.scenario = scenario
        self.requires_norm = requires_norm
        self.binary_labels = binary_labels
        self.linear_encoder = linear_encoder
        self.kwargs = kwargs

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either fit, validate, test, or predict.
        """
        if stage in ["fit"]:
            self.train_dataset = Sen4AgriNet(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                bands=self.bands,
                seed=self.seed,
                scenario=self.scenario,
                requires_norm=self.requires_norm,
                binary_labels=self.binary_labels,
                linear_encoder=self.linear_encoder,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = Sen4AgriNet(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                seed=self.seed,
                scenario=self.scenario,
                requires_norm=self.requires_norm,
                binary_labels=self.binary_labels,
                linear_encoder=self.linear_encoder,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = Sen4AgriNet(
                split="test",
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                seed=self.seed,
                scenario=self.scenario,
                requires_norm=self.requires_norm,
                binary_labels=self.binary_labels,
                linear_encoder=self.linear_encoder,
                **self.kwargs,
            )
        if stage in ["predict"]:
            self.predict_dataset = Sen4AgriNet(
                split="test",
                data_root=self.data_root,
                transform=self.predict_transform,
                bands=self.bands,
                seed=self.seed,
                scenario=self.scenario,
                requires_norm=self.requires_norm,
                binary_labels=self.binary_labels,
                linear_encoder=self.linear_encoder,
                **self.kwargs,
            )
