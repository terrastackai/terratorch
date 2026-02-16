# Copyright contributors to the Terratorch project

import gc
import unittest
from unittest.mock import patch

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


# ---------------------------------------------------------------------------
# Distributed num_boxes normalization tests
# ---------------------------------------------------------------------------


class TestDistributedNumBoxes(unittest.TestCase):
    """Test all_reduce is called correctly in SetCriterion for DDP."""

    def _make_detr_criterion_and_inputs(self):
        """Build a DETR SetCriterion and dummy inputs."""
        from terratorch.models.detr.detr import SetCriterion  # noqa: PLC0415
        from terratorch.models.detr.matcher import HungarianMatcher  # noqa: PLC0415

        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        criterion = SetCriterion(
            num_classes=10,
            matcher=matcher,
            weight_dict={"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2},
            eos_coef=0.1,
            losses=["labels", "boxes", "cardinality"],
        )
        outputs = {
            "pred_logits": torch.randn(2, 20, 11),
            "pred_boxes": torch.rand(2, 20, 4).sigmoid(),
        }
        targets = [
            {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
            {"labels": torch.tensor([3]), "boxes": torch.rand(1, 4)},
        ]
        return criterion, outputs, targets

    def test_detr_calls_allreduce_when_distributed(self):
        criterion, outputs, targets = self._make_detr_criterion_and_inputs()
        with (
            patch("terratorch.models.detr.detr.is_dist_avail_and_initialized", return_value=True),
            patch("torch.distributed.all_reduce") as mock_allreduce,
        ):
            criterion(outputs, targets)
        mock_allreduce.assert_called_once()

    def test_detr_no_allreduce_single_gpu(self):
        criterion, outputs, targets = self._make_detr_criterion_and_inputs()
        with (
            patch("terratorch.models.detr.detr.is_dist_avail_and_initialized", return_value=False),
            patch("torch.distributed.all_reduce") as mock_allreduce,
        ):
            criterion(outputs, targets)
        mock_allreduce.assert_not_called()

    @pytest.mark.skipif(not _has_msdeform, reason="MultiScaleDeformableAttention not installed")
    def test_deformable_detr_calls_allreduce_when_distributed(self):
        from terratorch.models.detr.deformable_detr import SetCriterion  # noqa: PLC0415
        from terratorch.models.detr.matcher import HungarianMatcher  # noqa: PLC0415

        matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
        criterion = SetCriterion(
            num_classes=10,
            matcher=matcher,
            weight_dict={"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2},
            losses=["labels", "boxes", "cardinality"],
            focal_alpha=0.25,
        )
        outputs = {
            "pred_logits": torch.randn(2, 20, 10),
            "pred_boxes": torch.rand(2, 20, 4).sigmoid(),
        }
        targets = [
            {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
            {"labels": torch.tensor([3]), "boxes": torch.rand(1, 4)},
        ]
        with (
            patch(
                "terratorch.models.detr.deformable_detr.is_dist_avail_and_initialized",
                return_value=True,
            ),
            patch("torch.distributed.all_reduce") as mock_allreduce,
        ):
            criterion(outputs, targets)
        mock_allreduce.assert_called_once()


# ---------------------------------------------------------------------------
# DETR auxiliary loss tests
# ---------------------------------------------------------------------------


class TestDETRAuxLoss(unittest.TestCase):
    """Test DETR auxiliary loss support."""

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
            framework_num_decoder_layers=3,
            framework_dim_feedforward=128,
            framework_aux_loss=True,
            **kwargs,
        )

    def test_train_has_aux_loss_keys(self):
        self.model.train()
        images = torch.randn(2, 3, 128, 128)
        targets = _dummy_targets(2, 128, 128)
        output = self.model(images, targets)
        losses = output.output
        # 3 decoder layers → aux layers 0, 1 (first two intermediate)
        for i in range(2):
            assert f"loss_ce_{i}" in losses, f"loss_ce_{i} missing"
            assert f"loss_bbox_{i}" in losses, f"loss_bbox_{i} missing"
            assert f"loss_giou_{i}" in losses, f"loss_giou_{i} missing"
        gc.collect()

    def test_all_aux_losses_have_grad(self):
        self.model.train()
        images = torch.randn(2, 3, 128, 128)
        targets = _dummy_targets(2, 128, 128)
        output = self.model(images, targets)
        losses = output.output
        for k, v in losses.items():
            assert v.requires_grad, f"{k} should require grad"
        gc.collect()

    def test_weight_dict_extension(self):
        detr = self.model.torchvision_model
        wd = detr.criterion.weight_dict
        # Base keys
        assert "loss_ce" in wd
        assert "loss_bbox" in wd
        assert "loss_giou" in wd
        # Aux keys for 2 intermediate layers
        for i in range(2):
            assert f"loss_ce_{i}" in wd
            assert f"loss_bbox_{i}" in wd
            assert f"loss_giou_{i}" in wd

    def test_eval_not_affected(self):
        self.model.eval()
        images = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            output = self.model(images)
        preds = output.output
        assert isinstance(preds, list)
        assert len(preds) == 2
        gc.collect()


# ---------------------------------------------------------------------------
# Deformable DETR auxiliary loss tests
# ---------------------------------------------------------------------------


@requires_msdeform
class TestDeformableDETRAuxLoss(unittest.TestCase):
    """Test Deformable DETR auxiliary loss support."""

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
            framework_num_decoder_layers=3,
            framework_dim_feedforward=128,
            framework_aux_loss=True,
            **kwargs,
        )

    def test_train_has_aux_loss_keys(self):
        self.model.train()
        images = torch.randn(2, 3, 128, 128)
        targets = _dummy_targets(2, 128, 128)
        output = self.model(images, targets)
        losses = output.output
        for i in range(2):
            assert f"loss_ce_{i}" in losses, f"loss_ce_{i} missing"
            assert f"loss_bbox_{i}" in losses, f"loss_bbox_{i} missing"
            assert f"loss_giou_{i}" in losses, f"loss_giou_{i} missing"
        gc.collect()

    def test_all_aux_losses_have_grad(self):
        self.model.train()
        images = torch.randn(2, 3, 128, 128)
        targets = _dummy_targets(2, 128, 128)
        output = self.model(images, targets)
        losses = output.output
        for k, v in losses.items():
            assert v.requires_grad, f"{k} should require grad"
        gc.collect()

    def test_eval_not_affected(self):
        self.model.eval()
        images = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            output = self.model(images)
        preds = output.output
        assert isinstance(preds, list)
        assert len(preds) == 2
        gc.collect()


# ---------------------------------------------------------------------------
# Deformable DETR extra feature levels tests
# ---------------------------------------------------------------------------


@requires_msdeform
class TestDeformableDETRExtraLevels(unittest.TestCase):
    """Test extra feature level synthesis for Deformable DETR."""

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
            framework_num_feature_levels=6,
            framework_aux_loss=False,
            **kwargs,
        )
        cls.detr = cls.model.torchvision_model

    def test_input_proj_count(self):
        assert len(self.detr.input_proj) == 6

    def test_extra_proj_is_stride2_conv(self):
        # Backbone has 4 levels, so input_proj[4] and [5] are extra (stride-2 3x3)
        for idx in (4, 5):
            conv = self.detr.input_proj[idx][0]
            assert conv.kernel_size == (3, 3), f"input_proj[{idx}] should be 3x3"
            assert conv.stride == (2, 2), f"input_proj[{idx}] should have stride 2"

    def test_train_forward(self):
        self.model.train()
        images = torch.randn(2, 3, 128, 128)
        targets = _dummy_targets(2, 128, 128)
        output = self.model(images, targets)
        losses = output.output
        assert isinstance(losses, dict)
        assert "loss_ce" in losses
        gc.collect()

    def test_eval_forward(self):
        self.model.eval()
        images = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            output = self.model(images)
        preds = output.output
        assert isinstance(preds, list)
        assert len(preds) == 2
        gc.collect()


# ---------------------------------------------------------------------------
# Component parity tests
# ---------------------------------------------------------------------------


class TestComponentParity(unittest.TestCase):
    """Test individual DETR components for correctness."""

    def test_position_encoding_deterministic(self):
        from terratorch.models.detr.position_encoding import PositionEmbeddingSine  # noqa: PLC0415

        pe = PositionEmbeddingSine(64, normalize=True)
        x = torch.randn(2, 128, 16, 16)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2), "Position encoding should be deterministic"

    def test_position_encoding_with_mask_equals_without(self):
        from terratorch.models.detr.position_encoding import PositionEmbeddingSine  # noqa: PLC0415

        pe = PositionEmbeddingSine(64, normalize=True)
        x = torch.randn(2, 128, 16, 16)
        out_no_mask = pe(x)
        all_false_mask = torch.zeros(2, 16, 16, dtype=torch.bool)
        out_with_mask = pe(x, all_false_mask)
        assert torch.allclose(out_no_mask, out_with_mask), "pe(x) should equal pe(x, all_false_mask)"

    def test_hungarian_matcher_valid_indices(self):
        from terratorch.models.detr.matcher import HungarianMatcher  # noqa: PLC0415

        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        outputs = {
            "pred_logits": torch.randn(2, 20, 11),
            "pred_boxes": torch.rand(2, 20, 4).sigmoid(),
        }
        targets = [
            {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
            {"labels": torch.tensor([3]), "boxes": torch.rand(1, 4)},
        ]
        indices = matcher(outputs, targets)
        assert len(indices) == 2
        # First batch has 2 targets
        assert len(indices[0][0]) == 2
        assert len(indices[0][1]) == 2
        # Second batch has 1 target
        assert len(indices[1][0]) == 1
        assert len(indices[1][1]) == 1

    def test_transformer_output_shape_intermediate(self):
        from terratorch.models.detr.transformer import Transformer  # noqa: PLC0415

        t = Transformer(d_model=64, nhead=4, num_encoder_layers=1, num_decoder_layers=3, return_intermediate_dec=True)
        src = torch.randn(2, 64, 8, 8)
        mask = torch.zeros(2, 8, 8, dtype=torch.bool)
        query_embed = torch.randn(10, 64)
        pos_embed = torch.randn(2, 64, 8, 8)
        hs, _memory = t(src, mask, query_embed, pos_embed)
        # return_intermediate_dec=True → [num_layers, B, Q, D]
        assert hs.shape == (3, 2, 10, 64)

    def test_transformer_output_shape_no_intermediate(self):
        from terratorch.models.detr.transformer import Transformer  # noqa: PLC0415

        t = Transformer(d_model=64, nhead=4, num_encoder_layers=1, num_decoder_layers=3, return_intermediate_dec=False)
        src = torch.randn(2, 64, 8, 8)
        mask = torch.zeros(2, 8, 8, dtype=torch.bool)
        query_embed = torch.randn(10, 64)
        pos_embed = torch.randn(2, 64, 8, 8)
        hs, _memory = t(src, mask, query_embed, pos_embed)
        # return_intermediate_dec=False → [1, B, Q, D]
        assert hs.shape == (1, 2, 10, 64)

    def test_setcriterion_detr_loss_keys(self):
        from terratorch.models.detr.detr import SetCriterion  # noqa: PLC0415
        from terratorch.models.detr.matcher import HungarianMatcher  # noqa: PLC0415

        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        criterion = SetCriterion(
            num_classes=10,
            matcher=matcher,
            weight_dict={"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2},
            eos_coef=0.1,
            losses=["labels", "boxes", "cardinality"],
        )
        outputs = {
            "pred_logits": torch.randn(2, 20, 11),
            "pred_boxes": torch.rand(2, 20, 4).sigmoid(),
        }
        targets = [
            {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
            {"labels": torch.tensor([3]), "boxes": torch.rand(1, 4)},
        ]
        losses = criterion(outputs, targets)
        assert "loss_ce" in losses
        assert "loss_bbox" in losses
        assert "loss_giou" in losses
        assert "cardinality_error" in losses

    def test_setcriterion_detr_with_aux_outputs(self):
        from terratorch.models.detr.detr import SetCriterion  # noqa: PLC0415
        from terratorch.models.detr.matcher import HungarianMatcher  # noqa: PLC0415

        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        criterion = SetCriterion(
            num_classes=10,
            matcher=matcher,
            weight_dict={"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2},
            eos_coef=0.1,
            losses=["labels", "boxes", "cardinality"],
        )
        outputs = {
            "pred_logits": torch.randn(2, 20, 11),
            "pred_boxes": torch.rand(2, 20, 4).sigmoid(),
            "aux_outputs": [
                {"pred_logits": torch.randn(2, 20, 11), "pred_boxes": torch.rand(2, 20, 4).sigmoid()},
                {"pred_logits": torch.randn(2, 20, 11), "pred_boxes": torch.rand(2, 20, 4).sigmoid()},
            ],
        }
        targets = [
            {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
            {"labels": torch.tensor([3]), "boxes": torch.rand(1, 4)},
        ]
        losses = criterion(outputs, targets)
        # Should have aux loss keys for layers 0 and 1
        assert "loss_ce_0" in losses
        assert "loss_bbox_0" in losses
        assert "loss_giou_0" in losses
        assert "loss_ce_1" in losses
        assert "loss_bbox_1" in losses
        assert "loss_giou_1" in losses

    @pytest.mark.skipif(not _has_msdeform, reason="MultiScaleDeformableAttention not installed")
    def test_setcriterion_deformable_loss_keys(self):
        from terratorch.models.detr.deformable_detr import SetCriterion  # noqa: PLC0415
        from terratorch.models.detr.matcher import HungarianMatcher  # noqa: PLC0415

        matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
        criterion = SetCriterion(
            num_classes=10,
            matcher=matcher,
            weight_dict={"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2},
            losses=["labels", "boxes", "cardinality"],
            focal_alpha=0.25,
        )
        outputs = {
            "pred_logits": torch.randn(2, 20, 10),
            "pred_boxes": torch.rand(2, 20, 4).sigmoid(),
        }
        targets = [
            {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
            {"labels": torch.tensor([3]), "boxes": torch.rand(1, 4)},
        ]
        losses = criterion(outputs, targets)
        assert "loss_ce" in losses
        assert "loss_bbox" in losses
        assert "loss_giou" in losses


# ---------------------------------------------------------------------------
# End-to-end wrapper parity tests
# ---------------------------------------------------------------------------


class TestDETRWrapperParity(unittest.TestCase):
    """End-to-end wrapper tests for determinism and gradient flow."""

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
            framework_num_decoder_layers=2,
            framework_dim_feedforward=128,
            framework_aux_loss=True,
            **kwargs,
        )

    def test_eval_deterministic(self):
        self.model.eval()
        images = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out1 = self.model(images).output
            out2 = self.model(images).output
        for p1, p2 in zip(out1, out2, strict=False):
            assert torch.allclose(p1["scores"], p2["scores"]), "Eval should be deterministic"
            assert torch.allclose(p1["boxes"], p2["boxes"]), "Eval should be deterministic"
        gc.collect()

    def test_aux_loss_backward(self):
        self.model.train()
        images = torch.randn(2, 3, 128, 128)
        targets = _dummy_targets(2, 128, 128)
        output = self.model(images, targets)
        losses = output.output
        total_loss = sum(losses.values())
        total_loss.backward()
        # Check gradients exist on DETR-specific params (transformer, heads, etc.)
        # Backbone/neck params may not all receive gradients since DETR only uses
        # the last feature map.
        detr = self.model.torchvision_model
        detr_prefixes = ("transformer", "class_embed", "bbox_embed", "query_embed", "input_proj")
        for name, param in detr.named_parameters():
            if param.requires_grad and name.startswith(detr_prefixes):
                assert param.grad is not None, f"{name} should have gradient after backward"
        gc.collect()


if __name__ == "__main__":
    unittest.main()
