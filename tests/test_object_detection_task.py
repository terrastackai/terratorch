import gc

import pytest
import torch

from terratorch.tasks.object_detection_task import ObjectDetectionTask


def task1():
    return ObjectDetectionTask(
        model_factory="ObjectDetectionModelFactory",
        model_args={
            "framework": "faster-rcnn",
            "backbone": "prithvi_eo_v2_300",
            "num_classes": 12,
            "backbone_pretrained": False,
            "backbone_bands": ["RED", "GREEN", "BLUE"],
            "necks": [
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage"},
                {"name": "LearnedInterpolateToPyramidal"},
                {"name": "FeaturePyramidNetworkNeck"},
            ],
        },
        lr=0.001,
        optimizer="Adam",
        optimizer_hparams={},
        scheduler=None,
        scheduler_hparams={},
        freeze_backbone=False,
        freeze_decoder=False,
        class_names=None,
        iou_threshold=0.5,
        score_threshold=0.5,
        boxes_field="boxes",
        labels_field="labels",
        masks_field="masks",
    )


def task2():
    return ObjectDetectionTask(
        model_factory="ObjectDetectionModelFactory",
        model_args={
            "framework": "faster-rcnn",
            "backbone": "timm_resnet50",
            "backbone_pretrained": False,
            "num_classes": 12,
            "in_channels": 3,
            "necks": [{"name": "FeaturePyramidNetworkNeck"}],
        },
        lr=0.001,
        optimizer="Adam",
        optimizer_hparams={},
        scheduler=None,
        scheduler_hparams={},
        freeze_backbone=False,
        freeze_decoder=False,
        class_names=None,
        iou_threshold=0.5,
        score_threshold=0.5,
        boxes_field="bbox_xyxy",
        labels_field="label",
        masks_field="mask",
    )


def dummy_batch1():
    return {
        "image": torch.randn(2, 3, 256, 256),
        "boxes": [
            torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32),
            torch.tensor([[20, 20, 60, 60]], dtype=torch.float32),
        ],
        "labels": [torch.tensor([1, 2]), torch.tensor([2])],
    }


def dummy_batch2():
    return {
        "image": torch.randn(2, 3, 256, 256),
        "bbox_xyxy": [
            torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32),
            torch.tensor([[20, 20, 60, 60]], dtype=torch.float32),
        ],
        "label": [torch.tensor([1, 2]), torch.tensor([2])],
    }


tasks = [task1(), task2()]

dummy_batches = [dummy_batch1(), dummy_batch2()]


@pytest.mark.parametrize("task", tasks)
def test_initialization(task):
    assert hasattr(task, "model_factory")
    assert hasattr(task, "monitor")
    assert task.score_threshold == 0.5


@pytest.mark.parametrize("task", tasks)
def test_configure_models(task):
    task.configure_models()
    assert hasattr(task, "model")
    assert callable(task.model.forward)


@pytest.mark.parametrize("task", tasks)
def test_configure_metrics(task):
    task.configure_metrics()
    assert hasattr(task, "train_metrics")
    assert hasattr(task, "val_metrics")
    assert hasattr(task, "test_metrics")


@pytest.mark.parametrize("task", tasks)
def test_configure_optimizers(task):
    opt = task.configure_optimizers()
    assert isinstance(opt, dict) or hasattr(opt, "optimizer")


@pytest.mark.parametrize("task,dummy_batch", zip(tasks, dummy_batches, strict=False))
def test_reformat_batch(task, dummy_batch):
    batch_size = 2
    reformatted = task.reformat_batch(dummy_batch, batch_size)
    assert isinstance(reformatted, list)
    assert "boxes" in reformatted[0]
    assert "labels" in reformatted[0]


@pytest.mark.parametrize("task,dummy_batch", zip(tasks, dummy_batches, strict=False))
def test_apply_ignore_index(task, dummy_batch):
    task.ignore_index = 1
    batch_size = 2

    filtered = task.apply_ignore_index(dummy_batch, task.ignore_index)
    reformatted = task.reformat_batch(filtered, batch_size)

    for i in range(len(reformatted)):
        assert not (reformatted[i]["labels"] == 1).any()


@pytest.mark.parametrize("task", tasks)
def test_apply_nms_sample(task):
    sample = {
        "boxes": torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=torch.float32),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([1, 1]),
    }
    filtered = task.apply_nms_sample(sample)
    assert filtered["boxes"].shape[0] <= 2
    gc.collect()


@pytest.mark.parametrize("task", tasks)
def test_apply_nms_batch(task):
    batch = [
        {
            "boxes": torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=torch.float32),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([1, 1]),
        },
        {
            "boxes": torch.tensor([[5, 5, 15, 15]], dtype=torch.float32),
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([2]),
        },
    ]
    filtered = task.apply_nms_batch(batch, batch_size=2)
    assert isinstance(filtered, list)
    assert all("boxes" in pred for pred in filtered)
    gc.collect()


@pytest.mark.parametrize("task,dummy_batch", zip(tasks, dummy_batches, strict=False))
def test_training_step(task, dummy_batch):
    task.configure_models()
    loss = task.training_step(dummy_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    gc.collect()


@pytest.mark.parametrize("task,dummy_batch", zip(tasks, dummy_batches, strict=False))
def test_predict_step(task, dummy_batch):
    task.configure_models()

    task.model.eval()
    predictions = task.predict_step(dummy_batch, batch_idx=0)
    assert isinstance(predictions, list)
    assert all("boxes" in pred for pred in predictions)
    gc.collect()
