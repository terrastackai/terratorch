import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric


class BoundaryMeanIoU(Metric):
    """Boundary mIoU for multiclass segmentation.

    Computes IoU on n-pixel-wide boundary bands of prediction and target for each class,
    then aggregates (macro/micro). `ignore_index` is ignored in both pred/target.

    Metric based on https://arxiv.org/abs/2103.16562:
    Cheng, B., Girshick, R., Dollár, P., Berg, A. C., & Kirillov, A. (2021). Boundary IoU: Improving object-centric
    image segmentation evaluation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
    """

    full_state_update = False

    def __init__(
        self,
        num_classes: int,
        thickness: int = 2,  # boundary band half-width in pixels
        ignore_index: int | None = None,
        average: str = "macro",  # "macro" or "micro"
        include_background: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if average not in {"macro", "micro"}:
            raise ValueError("average must be 'macro' or 'micro'")

        self.num_classes = num_classes
        self.thickness = thickness
        self.ignore_index = ignore_index
        self.average = average
        self.include_background = include_background

        # accumulators across batches
        self.add_state("intersections", default=torch.zeros(num_classes, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("unions", default=torch.zeros(num_classes, dtype=torch.int), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        preds: (N, C, H, W) logits/probs OR (N, H, W) class indices
        target: (N, H, W) class indices
        """
        if preds.dim() == 4:
            # logits -> hard labels
            preds_idx = preds.argmax(dim=1)
        elif preds.dim() == 3:
            preds_idx = preds
        else:
            raise ValueError("preds must be (N,C,H,W) or (N,H,W)")

        if target.dim() != 3:
            raise ValueError("target must be (N,H,W)")

        k = 2 * self.thickness + 1

        # mask out ignore_index everywhere
        ignore_mask = torch.zeros_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            ignore_mask = target == self.ignore_index

        # optionally skip background class
        start_cls = 0 if self.include_background else 1

        for c in range(start_cls, self.num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue

            # binary masks for class c
            pred_c = (preds_idx == c).float().unsqueeze(1)  # (N,1,H,W)
            target_c = (target == c).float().unsqueeze(1)

            # compute boundary bands via morphological gradient on binary maps
            dil_pred = F.max_pool2d(pred_c, kernel_size=k, stride=1, padding=self.thickness)
            ero_pred = 1.0 - F.max_pool2d(1.0 - pred_c, kernel_size=k, stride=1, padding=self.thickness)
            bnd_pred = (dil_pred - ero_pred).clamp_min(0.0) > 0.5  # (N,1,H,W) -> bool

            dil_target = F.max_pool2d(target_c, kernel_size=k, stride=1, padding=self.thickness)
            ero_target = 1.0 - F.max_pool2d(1.0 - target_c, kernel_size=k, stride=1, padding=self.thickness)
            bnd_target = (dil_target - ero_target).clamp_min(0.0) > 0.5

            # Apply ignore mask
            bnd_pred = bnd_pred & ~ignore_mask
            bnd_target = bnd_target & ~ignore_mask

            # IoU on boundary bands
            inter = (bnd_pred & bnd_target).sum()
            union = (bnd_pred | bnd_target).sum()

            # Accumulate
            self.intersections[c] += inter
            self.unions[c] += union

    def compute(self) -> Tensor:
        eps = 1e-9
        valid = self.unions > 0  # classes that had any boundary pixels at all

        # exclude classes with no boundary (union==0) from macro average
        if self.average == "macro":
            denom = valid.sum()
            iou_per_class = self.intersections[valid] / (self.unions[valid] + eps)
            return iou_per_class.mean() if denom > 0 else torch.tensor(0.0)
        else:  # micro
            inter_sum = self.intersections[valid].sum()
            union_sum = self.unions[valid].sum() + eps
            return inter_sum / union_sum
