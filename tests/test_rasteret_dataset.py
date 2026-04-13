from datetime import UTC, datetime
from unittest.mock import MagicMock

import geopandas as gpd
import pandas as pd
import pytest
import shapely
import torch
from pyproj import CRS

from terratorch.datasets.rasteret import RasteretDataset


class _DummyRasteretGeoDataset:
    def __init__(self) -> None:
        interval_index = pd.IntervalIndex.from_tuples(
            [(datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC))],
            closed="both",
            name="datetime",
        )
        self.index = gpd.GeoDataFrame(
            {"rid": [0]},
            index=interval_index,
            geometry=[shapely.box(399960, 5390220, 400600, 5390860)],
            crs=CRS.from_epsg(32632),
        )
        self._res = (10.0, 10.0)
        self.closed = False

    @property
    def crs(self) -> CRS:
        return CRS.from_user_input(self.index.crs)

    @property
    def res(self) -> tuple[float, float]:
        return self._res

    @res.setter
    def res(self, new_res: float | tuple[float, float]) -> None:
        if isinstance(new_res, int | float):
            self._res = (float(new_res), float(new_res))
            return
        self._res = new_res

    def __getitem__(self, _index):
        return {
            "image": torch.ones((3, 16, 16), dtype=torch.float32),
            "bounds": torch.zeros(9, dtype=torch.float64),
            "transform": torch.zeros(9, dtype=torch.float64),
        }

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _disable_rasteret_import(monkeypatch):
    monkeypatch.setattr("terratorch.datasets.rasteret.lazy_import", lambda _name: None)


@pytest.fixture
def collection():
    delegate = _DummyRasteretGeoDataset()
    mock = MagicMock()
    mock.to_torchgeo_dataset.return_value = delegate
    return mock


def test_init_forwards_collection_adapter(collection):
    ds = RasteretDataset(collection=collection, bands=["B04", "B03", "B02"])
    call = collection.to_torchgeo_dataset.call_args

    assert call is not None
    assert call.kwargs["bands"] == ["B04", "B03", "B02"]
    assert call.kwargs["target_crs"] is None
    assert ds.bands == ("B04", "B03", "B02")
    assert ds.crs == CRS.from_epsg(32632)


def test_init_with_crs_and_res(collection):
    ds = RasteretDataset(
        collection=collection,
        bands=["B04"],
        crs=CRS.from_epsg(4326),
        res=0.0001,
    )
    call = collection.to_torchgeo_dataset.call_args

    assert call is not None
    assert call.kwargs["target_crs"] == 4326
    assert ds.res == (0.0001, 0.0001)


def test_init_requires_bands(collection):
    with pytest.raises(ValueError, match="At least one band"):
        RasteretDataset(collection=collection, bands=[])


def test_init_requires_callable_adapter():
    class _CollectionWithoutAdapter:
        to_torchgeo_dataset = "not-callable"

    with pytest.raises(TypeError, match="to_torchgeo_dataset"):
        RasteretDataset(collection=_CollectionWithoutAdapter(), bands=["B04"])


def test_non_epsg_crs_rejected(collection):
    non_epsg = CRS.from_wkt(
        'ENGCRS["foo",EDATUM["Unknown"],CS[Cartesian,2],'
        'AXIS["x",east,ORDER[1]],AXIS["y",north,ORDER[2]],'
        'LENGTHUNIT["metre",1]]'
    )

    with pytest.raises(ValueError, match="EPSG"):
        RasteretDataset(collection=collection, bands=["B04"], crs=non_epsg)


def test_getitem_and_dtype(collection):
    ds = RasteretDataset(collection=collection, bands=["B04", "B03", "B02"])
    sample = ds[ds.bounds]

    assert sample["image"].shape == (3, 16, 16)
    assert ds.dtype == torch.float32
    ds.is_image = False
    assert ds.dtype == torch.long


def test_res_setter(collection):
    ds = RasteretDataset(collection=collection, bands=["B04"])
    ds.res = 20.0

    assert ds.res == (20.0, 20.0)


def test_close(collection):
    ds = RasteretDataset(collection=collection, bands=["B04"])
    ds.close()

    assert ds._delegate.closed is True


def test_crs_setter_rejected(collection):
    ds = RasteretDataset(collection=collection, bands=["B04"])

    with pytest.raises(AttributeError, match="fixed after construction"):
        ds.crs = CRS.from_epsg(4326)
