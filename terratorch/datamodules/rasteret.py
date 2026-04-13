# Copyright contributors to the Terratorch project

"""Rasteret-backed TerraTorch geospatial datamodule."""

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any

from pyproj import CRS
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets.utils import lazy_import
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler

from terratorch.datasets.rasteret import RasteretDataset


class RasteretDataModule(GeoDataModule):
    """A ``GeoDataModule`` for Rasteret collections."""

    def __init__(
        self,
        bands: Sequence[str],
        collection: Any | None = None,
        collection_name: str | None = None,
        train_collection: Any | None = None,
        train_collection_name: str | None = None,
        val_collection: Any | None = None,
        val_collection_name: str | None = None,
        test_collection: Any | None = None,
        test_collection_name: str | None = None,
        predict_collection: Any | None = None,
        predict_collection_name: str | None = None,
        batch_size: int = 4,
        patch_size: int | tuple[int, int] = 256,
        length: int = 100,
        num_workers: int = 0,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        train_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        val_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        test_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        predict_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        train_geometries: Any = None,
        val_geometries: Any = None,
        test_geometries: Any = None,
        predict_geometries: Any = None,
        geometries_crs: int = 4326,
        cache: bool = True,  # noqa: FBT001, FBT002
        time_series: bool = False,  # noqa: FBT001, FBT002
        is_image: bool = True,  # noqa: FBT001, FBT002
        max_concurrent: int = 50,
        cloud_config: Any = None,
        backend: Any = None,
        allow_resample: bool = False,  # noqa: FBT001, FBT002
        label_field: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Rasteret-backed data module."""
        if not bands:
            msg = "At least one band is required"
            raise ValueError(msg)

        super().__init__(
            dataset_class=RasteretDataset,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )

        self.bands = list(bands)
        self.collection = collection
        self.collection_name = collection_name
        self.train_collection = train_collection
        self.train_collection_name = train_collection_name
        self.val_collection = val_collection
        self.val_collection_name = val_collection_name
        self.test_collection = test_collection
        self.test_collection_name = test_collection_name
        self.predict_collection = predict_collection
        self.predict_collection_name = predict_collection_name

        self.crs = crs
        self.res = res
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.predict_transform = predict_transform

        self.train_geometries = train_geometries
        self.val_geometries = val_geometries
        self.test_geometries = test_geometries
        self.predict_geometries = predict_geometries
        self.geometries_crs = geometries_crs

        self.cache = cache
        self.time_series = time_series
        self.is_image = is_image
        self.max_concurrent = max_concurrent
        self.cloud_config = cloud_config
        self.backend = backend
        self.allow_resample = allow_resample
        self.label_field = label_field

        self._collections_cache: dict[str, Any] = {}

    def _load_collection_by_name(self, collection_name: str) -> Any:
        if collection_name not in self._collections_cache:
            lazy_import("rasteret")
            rasteret = import_module("rasteret")
            self._collections_cache[collection_name] = rasteret.load(collection_name)
        return self._collections_cache[collection_name]

    def _resolve_collection(self, split: str) -> Any:
        split_collection = getattr(self, f"{split}_collection")
        if split_collection is not None:
            return split_collection

        split_collection_name = getattr(self, f"{split}_collection_name")
        if split_collection_name is not None:
            return self._load_collection_by_name(split_collection_name)

        if self.collection is not None:
            return self.collection

        if self.collection_name is not None:
            return self._load_collection_by_name(self.collection_name)

        msg = (
            f"No Rasteret collection configured for '{split}'. "
            "Set collection/collection_name or split-specific equivalents."
        )
        raise ValueError(msg)

    def _make_dataset(
        self,
        split: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None,
        geometries: Any,
    ) -> RasteretDataset:
        return self.dataset_class(
            collection=self._resolve_collection(split),
            bands=self.bands,
            crs=self.crs,
            res=self.res,
            transforms=transform,
            cache=self.cache,
            time_series=self.time_series,
            is_image=self.is_image,
            max_concurrent=self.max_concurrent,
            cloud_config=self.cloud_config,
            backend=self.backend,
            allow_resample=self.allow_resample,
            label_field=self.label_field,
            geometries=geometries,
            geometries_crs=self.geometries_crs,
        )

    def setup(self, stage: str | None) -> None:
        """Set up datasets and samplers."""
        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset("train", self.train_transform, self.train_geometries)
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )

        if stage in (None, "fit", "validate"):
            self.val_dataset = self._make_dataset("val", self.val_transform, self.val_geometries)
            self.val_sampler = GridGeoSampler(self.val_dataset, self.patch_size, self.patch_size)

        if stage in (None, "test"):
            self.test_dataset = self._make_dataset("test", self.test_transform, self.test_geometries)
            self.test_sampler = GridGeoSampler(self.test_dataset, self.patch_size, self.patch_size)

        if stage in (None, "predict"):
            self.predict_dataset = self._make_dataset("predict", self.predict_transform, self.predict_geometries)
            self.predict_sampler = GridGeoSampler(self.predict_dataset, self.patch_size, self.patch_size)
