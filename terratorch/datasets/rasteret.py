# Copyright contributors to the Terratorch project

"""Rasteret-backed TorchGeo raster dataset."""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from pyproj import CRS
from torchgeo.datasets.geo import RasterDataset, Sample
from torchgeo.datasets.utils import lazy_import


class RasteretDataset(RasterDataset):
    """A ``RasterDataset`` that delegates reads to Rasteret collections."""

    def __init__(
        self,
        collection: Any,
        bands: Sequence[str],
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,  # noqa: FBT001, FBT002
        time_series: bool = False,  # noqa: FBT001, FBT002
        is_image: bool = True,  # noqa: FBT001, FBT002
        max_concurrent: int = 50,
        cloud_config: Any = None,
        backend: Any = None,
        allow_resample: bool = False,  # noqa: FBT001, FBT002
        label_field: str | None = None,
        geometries: Any = None,
        geometries_crs: int = 4326,
    ) -> None:
        """Initialize a ``RasteretDataset`` from a Rasteret collection."""
        lazy_import("rasteret")

        if not bands:
            msg = "At least one band is required"
            raise ValueError(msg)

        to_torchgeo_dataset = getattr(collection, "to_torchgeo_dataset", None)
        if not callable(to_torchgeo_dataset):
            msg = "collection must be a rasteret.Collection with to_torchgeo_dataset(...)"
            raise TypeError(msg)

        target_crs: int | None = None
        if crs is not None:
            epsg = crs.to_epsg()
            if epsg is None:
                msg = "RasteretDataset requires an EPSG CRS"
                raise ValueError(msg)
            target_crs = int(epsg)

        self._delegate = to_torchgeo_dataset(
            bands=list(bands),
            is_image=is_image,
            allow_resample=allow_resample,
            label_field=label_field,
            geometries=geometries,
            geometries_crs=geometries_crs,
            transforms=transforms,
            max_concurrent=max_concurrent,
            cloud_config=cloud_config,
            backend=backend,
            time_series=time_series,
            target_crs=target_crs,
        )

        self.paths = ""
        self.bands = tuple(bands)
        self.all_bands = tuple(bands)
        self.transforms = transforms
        self.cache = cache
        self.time_series = time_series
        self.is_image = is_image
        self.separate_files = False
        self.band_indexes = None
        self.index = self._delegate.index
        self._res = self._delegate.res

        if res is not None:
            self.res = res

    def __getitem__(self, index: Any) -> Sample:
        """Retrieve a sample indexed by spatiotemporal slice."""
        return self._delegate[index]

    @property
    def crs(self) -> CRS:
        """Coordinate reference system of the dataset."""
        return self._delegate.crs

    @crs.setter
    def crs(self, _new_crs: CRS) -> None:
        """Reject post-init CRS changes."""
        msg = "RasteretDataset CRS is fixed after construction; create a new dataset with crs=..."
        raise AttributeError(msg)

    @property
    def res(self) -> tuple[float, float]:
        """Resolution of the dataset in units of CRS."""
        return self._delegate.res

    @res.setter
    def res(self, new_res: float | tuple[float, float]) -> None:
        """Change dataset resolution."""
        self._delegate.res = new_res
        self._res = self._delegate.res

    @property
    def dtype(self) -> torch.dtype:
        """The dtype used for outputs."""
        if self.is_image:
            return torch.float32
        return torch.long

    def close(self) -> None:
        """Close Rasteret background resources if supported by delegate."""
        close = getattr(self._delegate, "close", None)
        if callable(close):
            close()
