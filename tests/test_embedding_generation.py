import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
import torch.nn as nn
import json

from terratorch.models.utils import TemporalWrapper
from terratorch.tasks.embedding_generation import EmbeddingGenerationTask

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@torch.no_grad()
def test_predict_step_non_temporal(tmp_path):
    task = EmbeddingGenerationTask(
        model_args={
            "backbone": "terramind_v1_tiny",
            "backbone_pretrained": False,
            "backbone_modalities": ["S2L2A"],
        },
        output_dir=str(tmp_path),
        output_format="parquet_joint",
        layers=[-1],
        embedding_pooling="mean",
    )

    B, C, H, W = 3, 12, 64, 64
    batch = {
        "image": torch.randn(B, C, H, W),
        "filename": [f"s{i}.tif" for i in range(B)],
        "metadata": {"time": ["2023-01-01"] * B},
    }

    task.predict_step(batch)
    task.on_predict_end()  # ensure parquet writer is closed/flushed

    out = tmp_path / "layer_00" / "embeddings.parquet"
    assert out.exists()

    df = pd.read_parquet(out)
    assert len(df) == B
    assert "embedding" in df.columns

    cfg = tmp_path / "configuration_summary.json"
    assert cfg.exists()

    with open(cfg, "r") as f:
        cfgj = json.load(f)
    assert cfgj["output_format"] == "parquet_joint"
    assert cfgj["n_layers_saved"] == 1


@torch.no_grad()
def test_predict_step_temporal(tmp_path):
    task = EmbeddingGenerationTask(
        model_args={"backbone": "terramind_v1_tiny",
                    "backbone_modalities": ["S2L2A"],
                    "backbone_use_temporal": True,
                    "backbone_temporal_pooling": "keep"},
        output_dir=str(tmp_path),
        output_format="parquet",
        layers=[-1],
        embedding_pooling="mean",
    )

    B, T, C, H, W = 2, 3, 12, 128, 128
    x = torch.randn(B, C, T, H, W)
    file_ids = [[f"s{b}_t{t}" for t in range(T)] for b in range(B)]
    batch = {"image": x, "filename": file_ids}
    task.predict_step(batch)

    out = tmp_path / "layer_00" / f"{file_ids[0][0]}_embedding.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 1
    assert df['embedding'][0].size == 192


@torch.no_grad()
@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_predict_step_non_temporal_tiff_no_pooling(tmp_path):
    task = EmbeddingGenerationTask(
        model_args={
            "backbone": "terramind_v1_tiny",
            "backbone_modalities": ["S2L2A"],
        },
        output_dir=str(tmp_path),
        output_format="tiff",
        layers=[-1],
        embedding_pooling=None,   # no pooling
    )

    B, C, H, W = 2, 12, 128, 128
    x = torch.randn(B, C, H, W)
    file_ids = [f"s{b}" for b in range(B)]

    batch = {
        "image": x,
        "filename": file_ids,
    }

    task.predict_step(batch)
    out = tmp_path / "layer_00" / f"{file_ids[0]}_embedding.tif"
    assert out.exists()

    with rasterio.open(out) as ds:
        data = ds.read()

    assert data.ndim == 3
    assert data.shape[0] == 192


@pytest.fixture
def mock_backbone_registry(request):
    with patch("terratorch.models.utils.BACKBONE_REGISTRY") as registry:
        backbone = MagicMock(spec=nn.Module, name="mock_backbone_model")
        backbone.out_channels = [384]
        registry.build.return_value = backbone
        yield registry


def _has_warning(ws, substr: str) -> bool:
    return any(substr in str(w.message) for w in ws)

def test_init(temp_dir, mock_backbone_registry):
    with patch("terratorch.tasks.embedding_generation.logger") as log:
        task = EmbeddingGenerationTask(
            model_args={"backbone": "dummy"},
            output_dir=temp_dir,
            layers=[-1],
            output_format="tiff",
        )

    assert task.output_path == Path(temp_dir)
    assert task.embed_file_key == "filename"
    assert task.has_cls is False
    assert task.embedding_pooling is None
    assert task.embedding_indices == [-1]
    assert task.output_format == "tiff"

    assert task.model_args["necks"][0] == {"name": "SelectIndices", "indices": [-1]}
    assert task.model_args["necks"][1] == {"name": "ReshapeTokensToImage", "remove_cls_token": False}

    task = EmbeddingGenerationTask(
        model_args={"backbone": "dummy"},
        output_dir=temp_dir,
        layers=[0, -1],
        output_format="parquet",
        embedding_pooling="keep",
    )

    assert task.output_format == "parquet"
    assert task.model_args["necks"] == [{"name": "SelectIndices", "indices": [0, -1]}]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        task = EmbeddingGenerationTask(
            model_args={"backbone": "dummy"},
            output_dir=temp_dir,
            layers=[-1],
            output_format="tiff",
            has_cls=True,
            embedding_pooling="mean",
        )

    assert task.model_args["necks"] == [{
        "name": "AggregateTokens",
        "pooling": "mean",
        "indices": [-1],
        "drop_cls": True,
    }]
    assert any("GeoTIFF output not recommended" in str(w.message) for w in caught)
    log.info.assert_called()

def test_init_unsupported_output_format_raises(temp_dir, mock_backbone_registry):
    with pytest.raises(ValueError, match="Unsupported output format"):
        EmbeddingGenerationTask(
            model_args={"backbone": "dummy"},
            output_dir=temp_dir,
            output_format="npy",
        )

def test_init_unsupported_pooling_raises(temp_dir, mock_backbone_registry):
    with pytest.raises(ValueError, match="EmbeddingPooling .* is not supported"):
        EmbeddingGenerationTask(
            model_args={"backbone": "dummy"},
            output_dir=temp_dir,
            embedding_pooling="median",
        )

def test_output_format_is_lowercased(temp_dir, mock_backbone_registry):
    task = EmbeddingGenerationTask(
        model_args={"backbone": "dummy"},
        output_dir=temp_dir,
        output_format="PARQUET_JOINT",
    )
    assert task.output_format == "parquet_joint"

def test_infer_bt_tensor_and_dict(mock_backbone_registry):
    task = EmbeddingGenerationTask(model_args={"backbone": "dummy"})

    x = torch.randn(2, 3, 224, 224)
    assert task.infer_bt(x) == (2, 1)

    x_dict = {"optical": torch.randn(2, 3, 4, 224, 224)}
    assert task.infer_bt(x_dict) == (2, 4)


def test_check_file_ids_valid_and_errors(mock_backbone_registry):
    task = EmbeddingGenerationTask(model_args={"backbone": "dummy"})
    x_4d = torch.randn(2, 3, 224, 224)

    # valid list (B,)
    task.check_file_ids(["0", "1"], x_4d)

    # invalid shape
    with pytest.raises(ValueError, match="length mismatch"):
        task.check_file_ids(["0", "1", "2"], x_4d)

    # invalid type
    with pytest.raises(
        TypeError,
        match="must be a tensor/ndarray or a \\(nested\\) list/tuple",
    ):
        task.check_file_ids("bad", x_4d)

    # valid nested list temporal
    x_5d = torch.randn(2, 3, 3, 224, 224)
    file_ids = [["t1", "t2", "t3"], ["t4", "t5", "t6"]]
    task.check_file_ids(file_ids, x_5d)

def test_check_file_ids_invalid_inner_length(mock_backbone_registry):
    task = EmbeddingGenerationTask(model_args={"backbone": "dummy"})
    x = torch.randn(2, 3, 4, 224, 224)  # T = 4
    file_ids = [["t1", "t2"], ["t3", "t4"]]  # inner length 2 != 4

    with pytest.raises(ValueError, match="inner length 4"):
        task.check_file_ids(file_ids, x)

def test_save_configuration_summary_writes_file(tmp_path, mock_backbone_registry):
    fake_encoder = MagicMock()
    fake_encoder.return_value = [
        torch.zeros(1, 16, 8),   # layer 0
        torch.zeros(1, 32, 8),   # layer 1
    ]

    fake_model = MagicMock()
    fake_model.encoder = fake_encoder
    fake_model.parameters.return_value = [
        torch.nn.Parameter(torch.zeros(10)),
        torch.nn.Parameter(torch.zeros(20)),
    ]

    task = MagicMock()
    task._config_saved = False
    task.model = fake_model
    task.embedding_indices = [-1]
    task.output_path = tmp_path
    task.output_format = "tiff"
    task.model_args = {"backbone": "dummy_backbone"}
    task.has_cls = False
    task.embedding_pooling = None

    # bind method under test
    task.save_configuration_summary = EmbeddingGenerationTask.save_configuration_summary.__get__(task)

    # --- run ---
    x = torch.zeros(1, 3, 224, 224)
    task.save_configuration_summary(x)

    # --- assertions ---
    out_file = tmp_path / "configuration_summary.json"
    assert out_file.exists()

    cfg = json.loads(out_file.read_text())

    assert cfg["backbone"] == "dummy_backbone"
    assert cfg["model_layer_count"] == 2
    assert cfg["n_layers_saved"] == 1
    assert cfg["layers"][0]["requested_index"] == -1
    assert cfg["layers"][0]["layer_number"] == 2  # resolved -1 → last layer
    assert cfg["layers"][0]["layer_output_shape"] == [32, 8]

    assert task._config_saved is True

def test_pull_metadata_full_and_empty(mock_backbone_registry):
    task = EmbeddingGenerationTask(model_args={"backbone": "dummy"})

    data = {
        "file_id": "fid",
        "time_": "2023-01-01",
        "centre_lat": 1.0,
        "centre_lon": 2.0,
        "extra": "keep",
    }
    meta = task.pull_metadata(data)
    assert meta["file_id"] == "fid"
    assert meta["time"] == "2023-01-01"
    assert meta["center_lat"] == 1.0
    assert meta["center_lon"] == 2.0
    assert "extra" in data
    assert "file_id" not in data

    data2 = {"x": 1}
    assert task.pull_metadata(data2) == {}

def test_write_parquet(temp_dir, mock_backbone_registry):
    task = EmbeddingGenerationTask(
        model_args={"backbone": "dummy"},
        output_dir=temp_dir,
        output_format="parquet",
    )
    emb = np.random.random((768, 1024))
    meta = {"time": np.array("2023-01-01")}
    task.write_parquet(emb, "p", meta, Path(temp_dir))

    out = Path(temp_dir) / "p_embedding.parquet"
    df = pd.read_parquet(out)
    assert len(df) == 1
    assert len(df["embedding"][0]) == 768
    assert len(df["embedding"][0][0]) == 1024
    assert df["time"][0] == "2023-01-01"


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_write_tiff(temp_dir, mock_backbone_registry):
    task = EmbeddingGenerationTask(
        model_args={"backbone": "dummy"},
        output_dir=temp_dir,
        output_format="tiff",
    )

    emb = np.random.random((4, 8, 8)).astype(np.float32)
    meta = {"time": "2023-01-01"}

    task.write_tiff(emb, "p.tif", meta, Path(temp_dir))

    out = Path(temp_dir) / "p_embedding.tif"
    assert out.exists()

    with rasterio.open(out) as ds:
        data = ds.read()
        tags = ds.tags()

    assert data.shape == (4, 8, 8)
    assert tags["time"] == "2023-01-01"

def test_write_parquet_batch_and_close_parquet(temp_dir, mock_backbone_registry):
    task = EmbeddingGenerationTask(
        model_args={"backbone": "dummy"},
        output_dir=temp_dir,
        output_format="parquet_joint",
    )

    task.write_parquet_batch(
        emb_np=np.ones((4,1024)),
        file_ids=["a", "b", "c", "d"],
        metadata={},
        is_temporal=False,
        dir_path=Path(temp_dir) / "layer_00",
    )
    task.join_parquet_files()

    out = Path(temp_dir) / "layer_00" / "embeddings.parquet"
    assert out.exists()

    df = pd.read_parquet(out)
    assert len(df) == 4
    assert df["file_id"][0] == "a"
    assert len(df["embedding"][0]) == 1024

def test_predict_step_missing_filename(temp_dir, mock_backbone_registry):
    model = MagicMock()
    model.return_value = torch.randn(2, 4, 3, 3)

    with patch("terratorch.models.utils.BACKBONE_REGISTRY") as reg:
        reg.build.return_value = model
        task = EmbeddingGenerationTask(model_args={"backbone": "dummy"}, output_dir=temp_dir)

    batch = {
        "image": torch.randn(2, 3, 224, 224),
        "time": ["2023-01-01", "2023-01-02"],
    }
    with pytest.raises(KeyError, match="not found in input dictionary"):
        task.predict_step(batch)

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
