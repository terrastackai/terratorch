from datetime import UTC, datetime

import geopandas as gpd
import pandas as pd
import pytest
import shapely
import torch
from pyproj import CRS
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler

from terratorch.datamodules import RasteretDataModule


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


@pytest.fixture(autouse=True)
def _disable_rasteret_import(monkeypatch):
    monkeypatch.setattr("terratorch.datasets.rasteret.lazy_import", lambda _name: None)
    monkeypatch.setattr("terratorch.datamodules.rasteret.lazy_import", lambda _name: None)


@pytest.fixture
def collection():
    class _Collection:
        def to_torchgeo_dataset(self, **_kwargs):
            return _DummyRasteretGeoDataset()

    return _Collection()


def test_setup_fit_builds_train_and_val(collection):
    dm = RasteretDataModule(
        collection=collection,
        bands=["B04", "B03", "B02"],
        batch_size=2,
        patch_size=16,
        length=4,
    )
    dm.setup("fit")

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert isinstance(dm.train_batch_sampler, RandomBatchGeoSampler)
    assert isinstance(dm.val_sampler, GridGeoSampler)


def test_setup_test_and_predict(collection):
    dm = RasteretDataModule(
        collection=collection,
        bands=["B04"],
        batch_size=2,
        patch_size=16,
        length=4,
    )

    dm.setup("test")
    assert dm.test_dataset is not None
    assert isinstance(dm.test_sampler, GridGeoSampler)

    dm.setup("predict")
    assert dm.predict_dataset is not None
    assert isinstance(dm.predict_sampler, GridGeoSampler)


def test_missing_collection_raises():
    dm = RasteretDataModule(
        bands=["B04"],
        batch_size=2,
        patch_size=16,
        length=4,
    )

    with pytest.raises(ValueError, match="No Rasteret collection"):
        dm.setup("fit")


def test_collection_name_loads_once(monkeypatch, collection):
    class _RasteretModule:
        def __init__(self):
            self.load_calls = []

        def load(self, collection_name: str):
            self.load_calls.append(collection_name)
            return collection

    fake_module = _RasteretModule()
    monkeypatch.setattr("terratorch.datamodules.rasteret.import_module", lambda _name: fake_module)

    dm = RasteretDataModule(
        collection_name="demo-collection",
        bands=["B04"],
        batch_size=2,
        patch_size=16,
        length=4,
    )
    dm.setup("fit")

    assert fake_module.load_calls == ["demo-collection"]
