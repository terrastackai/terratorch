# Copyright contributors to the Terratorch project

"""TerraTorch wrapper classes for DETR and Deformable DETR.

These compose the reference DETR/Deformable DETR components with a BackboneWrapper
from ObjectDetectionModelFactory. They handle:
- Extracting features from backbone
- Building positional encodings
- Running the transformer encoder-decoder
- Computing losses (training) or postprocessing predictions (eval)
"""

import math
from collections import OrderedDict

import torch
from torch import Tensor, nn
from torchvision.ops import box_convert

from terratorch.models.detr.detr import MLP as DETR_MLP
from terratorch.models.detr.detr import PostProcess as DETRPostProcess
from terratorch.models.detr.detr import SetCriterion as DETRSetCriterion
from terratorch.models.detr.matcher import HungarianMatcher
from terratorch.models.detr.position_encoding import PositionEmbeddingSine
from terratorch.models.detr.transformer import Transformer


class TerraTorchDETR(nn.Module):
    """Wraps reference DETR for TerraTorch's ObjectDetectionModelFactory.

    Accepts a BackboneWrapper, builds Transformer/SetCriterion internally.
    Forward: images [B,C,H,W] + optional targets -> loss dict (train) or prediction list (eval)
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        in_channels: int = 3,  # noqa: ARG002
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_queries: int = 100,
        eos_coef: float = 0.1,
        aux_loss: bool = False,  # noqa: FBT001, FBT002
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.aux_loss = aux_loss

        backbone_out_channels = backbone.out_channels
        self.input_proj = nn.Conv2d(backbone_out_channels, d_model, kernel_size=1)
        self.position_embedding = PositionEmbeddingSine(d_model // 2, normalize=True)

        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            return_intermediate_dec=aux_loss,
        )

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = DETR_MLP(d_model, d_model, 4, 3)

        # Loss
        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        base_weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
        weight_dict = dict(base_weight_dict)
        if aux_loss:
            for i in range(num_decoder_layers - 1):
                weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
        losses = ["labels", "boxes", "cardinality"]
        self.criterion = DETRSetCriterion(
            num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses
        )

        self.postprocessor = DETRPostProcess()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        """Return auxiliary outputs for intermediate decoder layers."""
        return [
            {"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1], strict=False)
        ]

    def forward(self, images: Tensor, targets: list[dict] | None = None) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Forward pass.

        Args:
            images: [B, C, H, W] input images.
            targets: Training targets, list of dicts with 'boxes' (xyxy abs) and 'labels'.

        Returns:
            Training: dict of losses.
            Eval: list of dicts with 'boxes' (xyxy), 'scores', 'labels'.
        """
        _bs, _, img_h, img_w = images.shape

        # Extract features from backbone
        features = self.backbone(images)
        if isinstance(features, OrderedDict):
            feature_list = list(features.values())
        else:
            feature_list = features
        # Use last feature map
        src = feature_list[-1]  # [B, C_backbone, H', W']

        # Project to d_model
        src = self.input_proj(src)  # [B, d_model, H', W']
        pos = self.position_embedding(src)  # [B, d_model, H', W']

        # Create mask (all False = no padding)
        mask = torch.zeros(src.shape[0], src.shape[2], src.shape[3], dtype=torch.bool, device=src.device)

        # Run transformer
        hs = self.transformer(src, mask, self.query_embed.weight, pos)
        # hs is (decoder_output [num_layers, B, num_queries, d_model], memory)
        hs = hs[0]  # [num_layers, B, num_queries, d_model]

        # Prediction heads (applied to all decoder layers)
        outputs_class = self.class_embed(hs)  # [num_layers, B, num_queries, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [num_layers, B, num_queries, 4]

        # Use last decoder layer output
        outputs = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            outputs["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            if targets is None:
                msg = "targets must be provided during training"
                raise ValueError(msg)
            # Convert targets from xyxy absolute to cxcywh normalized for loss computation
            processed_targets = _convert_targets_for_loss(targets, img_h, img_w, images.device)
            loss_dict = self.criterion(outputs, processed_targets)
            # Weight losses (only include actual losses, not logging-only metrics)
            weighted = {}
            for k, v in loss_dict.items():
                if k in self.criterion.weight_dict:
                    weighted[k] = v * self.criterion.weight_dict[k]
            return weighted

        # Eval: postprocess to list of dicts
        target_sizes = torch.tensor([[img_h, img_w]], device=images.device).repeat(images.shape[0], 1)
        return self.postprocessor(outputs, target_sizes)


class TerraTorchDeformableDETR(nn.Module):
    """Wraps reference Deformable DETR for TerraTorch's ObjectDetectionModelFactory.

    Accepts a BackboneWrapper, builds DeformableTransformer/SetCriterion internally.
    Forward: images [B,C,H,W] + optional targets -> loss dict (train) or prediction list (eval)
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        in_channels: int = 3,  # noqa: ARG002
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_queries: int = 300,
        n_points: int = 4,
        eos_coef: float = 0.1,  # noqa: ARG002
        aux_loss: bool = True,  # noqa: FBT001, FBT002
        num_feature_levels: int | None = None,
    ):
        super().__init__()
        # Lazy import to avoid requiring CUDA extension when only DETR is used
        from terratorch.models.detr.deformable_detr import MLP as DeformMLP  # noqa: N811, PLC0415
        from terratorch.models.detr.deformable_detr import PostProcess as DeformPostProcess  # noqa: PLC0415
        from terratorch.models.detr.deformable_detr import SetCriterion as DeformSetCriterion  # noqa: PLC0415
        from terratorch.models.detr.deformable_transformer import DeformableTransformer  # noqa: PLC0415

        self.backbone = backbone
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        self.aux_loss = aux_loss

        n_backbone_outs = len(backbone.channel_list)
        n_levels = num_feature_levels if num_feature_levels is not None else n_backbone_outs
        self.num_feature_levels = n_levels

        # Per-level input projections (1x1 for backbone levels)
        input_proj_list = [
            nn.Sequential(
                nn.Conv2d(ch, d_model, kernel_size=1),
                nn.GroupNorm(32, d_model),
            )
            for ch in backbone.channel_list
        ]
        # Stride-2 3x3 projections for extra levels beyond backbone outputs
        in_ch = backbone.channel_list[-1]
        for _ in range(n_levels - n_backbone_outs):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, d_model),
                )
            )
            in_ch = d_model
        self.input_proj = nn.ModuleList(input_proj_list)

        # Sinusoidal positional encoding (shared across levels)
        self.position_embedding = PositionEmbeddingSine(d_model // 2, normalize=True)

        # Deformable Transformer
        self.transformer = DeformableTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=n_levels,
            dec_n_points=n_points,
            enc_n_points=n_points,
        )

        # Object queries (pos + content, split inside transformer)
        self.query_embed = nn.Embedding(num_queries, d_model * 2)

        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = DeformMLP(d_model, d_model, 4, 3)

        # Init weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # Shared heads across decoder layers (no box refinement)
        num_pred = self.transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None

        # Loss - Deformable DETR uses focal loss
        matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
        base_weight_dict = {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}
        weight_dict = dict(base_weight_dict)
        if aux_loss:
            for i in range(num_decoder_layers - 1):
                weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
        losses = ["labels", "boxes", "cardinality"]
        self.criterion = DeformSetCriterion(
            num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses, focal_alpha=0.25
        )

        self._postprocessor = DeformPostProcess()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        """Return auxiliary outputs for intermediate decoder layers."""
        return [
            {"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1], strict=False)
        ]

    def forward(self, images: Tensor, targets: list[dict] | None = None) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Forward pass.

        Args:
            images: [B, C, H, W] input images.
            targets: Training targets, list of dicts with 'boxes' (xyxy abs) and 'labels'.

        Returns:
            Training: dict of losses.
            Eval: list of dicts with 'boxes' (xyxy), 'scores', 'labels'.
        """
        from terratorch.models.detr.deformable_transformer import _inverse_sigmoid  # noqa: PLC0415

        bs, _, img_h, img_w = images.shape

        # Extract multi-scale features from backbone
        features = self.backbone(images)
        if isinstance(features, OrderedDict):
            feature_list = list(features.values())
        else:
            feature_list = features

        # Project each backbone level and collect spatial info
        srcs = []
        masks = []
        pos_embeds = []
        for lvl, feat in enumerate(feature_list):
            src = self.input_proj[lvl](feat)  # [bs, d_model, h_l, w_l]
            mask = torch.zeros(src.shape[0], src.shape[2], src.shape[3], dtype=torch.bool, device=src.device)
            pos = self.position_embedding(src)  # [bs, d_model, h_l, w_l]
            srcs.append(src)
            masks.append(mask)
            pos_embeds.append(pos)

        # Generate extra feature levels beyond backbone outputs
        if self.num_feature_levels > len(feature_list):
            _len_srcs = len(feature_list)
            for lvl in range(_len_srcs, self.num_feature_levels):
                if lvl == _len_srcs:
                    src = self.input_proj[lvl](feature_list[-1])
                else:
                    src = self.input_proj[lvl](srcs[-1])
                mask = torch.zeros(src.shape[0], src.shape[2], src.shape[3], dtype=torch.bool, device=src.device)
                pos = self.position_embedding(src, mask)
                srcs.append(src)
                masks.append(mask)
                pos_embeds.append(pos)

        # Run transformer
        hs, init_reference, inter_references, _, _ = self.transformer(srcs, masks, pos_embeds, self.query_embed.weight)

        # Prediction heads
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = _inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:  # noqa: PLR2004
                tmp += reference
            else:
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            if targets is None:
                msg = "targets must be provided during training"
                raise ValueError(msg)
            # Convert targets from xyxy absolute to cxcywh normalized for loss computation
            processed_targets = _convert_targets_for_loss(targets, img_h, img_w, images.device)
            loss_dict = self.criterion(out, processed_targets)
            # Weight losses (only include actual losses, not logging-only metrics)
            weighted = {}
            for k, v in loss_dict.items():
                if k in self.criterion.weight_dict:
                    weighted[k] = v * self.criterion.weight_dict[k]
            return weighted

        # Eval: postprocess to list of dicts
        target_sizes = torch.tensor([[img_h, img_w]], device=images.device).repeat(bs, 1)
        return self._postprocessor(out, target_sizes)


def _convert_targets_for_loss(targets: list[dict], img_h: int, img_w: int, device: torch.device) -> list[dict]:
    """Convert detection targets from xyxy absolute to cxcywh normalized.

    The reference DETR loss expects targets with:
    - 'labels': class labels tensor
    - 'boxes': bounding boxes in cxcywh format, normalized to [0, 1]

    Args:
        targets: list of dicts with 'boxes' in xyxy absolute and 'labels'.
        img_h: image height.
        img_w: image width.
        device: target device.

    Returns:
        Converted targets list.
    """
    processed = []
    for t in targets:
        boxes = t["boxes"].float().to(device)
        # Normalize to [0, 1]
        boxes_norm = boxes.clone()
        boxes_norm[:, 0::2] /= img_w
        boxes_norm[:, 1::2] /= img_h
        # Convert from xyxy to cxcywh
        boxes_cxcywh = box_convert(boxes_norm, "xyxy", "cxcywh")
        processed.append({"labels": t["labels"].to(device), "boxes": boxes_cxcywh})
    return processed
