# Copyright contributors to the Terratorch project

import gc
import unittest

import pytest
import torch

from terratorch.models.object_detection_model_factory import (
    ObjectDetectionModel,
    ObjectDetectionModelFactory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_factory_and_kwargs():
    """Common factory setup for prithvi backbone."""
    factory = ObjectDetectionModelFactory()
    necks = [
        {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
        {"name": "ReshapeTokensToImage"},
        {"name": "LearnedInterpolateToPyramidal"},
        {"name": "FeaturePyramidNetworkNeck"},
    ]
    kwargs = {"backbone_pretrained": False, "backbone_bands": ["RED", "GREEN", "BLUE"]}
    return factory, necks, kwargs


def _dummy_targets(batch_size: int = 2, img_h: int = 256, img_w: int = 256):
    """Create dummy detection targets (xyxy format)."""
    targets = []
    for _ in range(batch_size):
        n = torch.randint(1, 5, (1,)).item()
        x1 = torch.randint(0, img_w // 2, (n,)).float()
        y1 = torch.randint(0, img_h // 2, (n,)).float()
        x2 = x1 + torch.randint(10, img_w // 2, (n,)).float()
        y2 = y1 + torch.randint(10, img_h // 2, (n,)).float()
        x2 = x2.clamp(max=img_w)
        y2 = y2.clamp(max=img_h)
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        labels = torch.randint(1, 10, (n,))
        targets.append({"boxes": boxes, "labels": labels})
    return targets


# ---------------------------------------------------------------------------
# Vanilla DETR tests
# ---------------------------------------------------------------------------


class TestDETRFactory(unittest.TestCase):
    """Test building DETR via the factory."""

    def test_build_model_detr(self):
        factory, necks, kwargs = _make_factory_and_kwargs()
        model = factory.build_model(
            "object_detection",
            "prithvi_eo_v2_300",
            "detr",
            num_classes=10,
            necks=necks,
            **kwargs,
        )
        assert isinstance(model, ObjectDetectionModel)
        from terratorch.models.detr import DETR  # noqa: PLC0415

        assert isinstance(model.torchvision_model, DETR)


class TestDETRForward(unittest.TestCase):
    """Test DETR forward pass in train and eval modes."""

    @classmethod
    def setUpClass(cls):
        factory, necks, kwargs = _make_factory_and_kwargs()
        cls.model = factory.build_model(
            "object_detection",
            "prithvi_eo_v2_300",
            "detr",
            num_classes=10,
            necks=necks,
            framework_num_queries=20,
            framework_d_model=64,
            framework_nhead=4,
            framework_num_encoder_layers=1,
            framework_num_decoder_layers=1,
            framework_dim_feedforward=128,
            **kwargs,
        )

    def test_train_forward(self):
        self.model.train()
        images = torch.randn(2, 3, 128, 128)
        targets = _dummy_targets(2, 128, 128)
        output = self.model(images, targets)
        losses = output.output
        assert isinstance(losses, dict)
        assert "loss_ce" in losses
        assert "loss_bbox" in losses
        assert "loss_giou" in losses
        for v in losses.values():
            assert v.requires_grad
        gc.collect()

    def test_eval_forward(self):
        self.model.eval()
        images = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            output = self.model(images)
        preds = output.output
        assert isinstance(preds, list)
        assert len(preds) == 2
        for pred in preds:
            assert "boxes" in pred
            assert "scores" in pred
            assert "labels" in pred
            assert pred["boxes"].shape[1] == 4
        gc.collect()


class TestDETRFreezeDecoder(unittest.TestCase):
    """Test that freeze_decoder works for DETR."""

    def test_freeze_decoder_detr(self):
        factory, necks, kwargs = _make_factory_and_kwargs()
        model = factory.build_model(
            "object_detection",
            "prithvi_eo_v2_300",
            "detr",
            num_classes=10,
            necks=necks,
            framework_num_queries=10,
            framework_d_model=64,
            framework_nhead=4,
            framework_num_encoder_layers=1,
            framework_num_decoder_layers=1,
            **kwargs,
        )
        model.freeze_decoder()
        # Transformer and head params should be frozen
        detr = model.torchvision_model
        for name, param in detr.named_parameters():
            if name.startswith(("transformer", "class_embed", "bbox_embed", "query_embed", "input_proj")):
                assert not param.requires_grad, f"{name} should be frozen"
        # Backbone params should still be trainable
        for param in detr.backbone.parameters():
            assert param.requires_grad
        gc.collect()


# ---------------------------------------------------------------------------
# Deformable DETR tests (requires CUDA extension)
# ---------------------------------------------------------------------------

_has_msdeform = True
try:
    import MultiScaleDeformableAttention  # noqa: F401
except ImportError:
    _has_msdeform = False

requires_msdeform = pytest.mark.skipif(
    not _has_msdeform, reason="MultiScaleDeformableAttention CUDA extension not installed"
)


@requires_msdeform
class TestDeformableDETRFactory(unittest.TestCase):
    """Test building Deformable DETR via the factory."""

    def test_build_model_deformable_detr(self):
        factory, necks, kwargs = _make_factory_and_kwargs()
        model = factory.build_model(
            "object_detection",
            "prithvi_eo_v2_300",
            "deformable-detr",
            num_classes=10,
            necks=necks,
            **kwargs,
        )
        assert isinstance(model, ObjectDetectionModel)
        from terratorch.models.detr import DeformableDETR  # noqa: PLC0415

        assert isinstance(model.torchvision_model, DeformableDETR)


@requires_msdeform
class TestDeformableDETRForward(unittest.TestCase):
    """Test Deformable DETR forward pass."""

    @classmethod
    def setUpClass(cls):
        factory, necks, kwargs = _make_factory_and_kwargs()
        cls.model = factory.build_model(
            "object_detection",
            "prithvi_eo_v2_300",
            "deformable-detr",
            num_classes=10,
            necks=necks,
            framework_num_queries=20,
            framework_d_model=64,
            framework_nhead=4,
            framework_num_encoder_layers=1,
            framework_num_decoder_layers=1,
            framework_dim_feedforward=128,
            **kwargs,
        )

    def test_train_forward(self):
        self.model.train()
        images = torch.randn(2, 3, 128, 128)
        targets = _dummy_targets(2, 128, 128)
        output = self.model(images, targets)
        losses = output.output
        assert isinstance(losses, dict)
        assert "loss_ce" in losses
        assert "loss_bbox" in losses
        assert "loss_giou" in losses
        gc.collect()

    def test_eval_forward(self):
        self.model.eval()
        images = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            output = self.model(images)
        preds = output.output
        assert isinstance(preds, list)
        assert len(preds) == 2
        for pred in preds:
            assert "boxes" in pred
            assert "scores" in pred
            assert "labels" in pred
        gc.collect()


if __name__ == "__main__":
    unittest.main()
