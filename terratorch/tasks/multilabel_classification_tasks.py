from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchmetrics.wrappers import ClasswiseWrapper

from terratorch.models.model import ModelOutput
from terratorch.tasks import ClassificationTask
from terratorch.tasks.loss_handler import CombinedLoss, LossHandler


# from geobench
def _balanced_binary_cross_entropy_with_logits(outputs: Tensor, targets: Tensor) -> Tensor:
    """Compute balance binary cross entropy for multi-label classification.

    Args:
        outputs: model outputs
        targets: targets to compute binary cross entropy on
    """
    classes = targets.shape[-1]
    outputs = outputs.view(-1, classes)
    targets = targets.view(-1, classes).float()
    loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    loss = loss[targets == 0].mean() + loss[targets == 1].mean()
    return loss


def init_loss(loss: str, ignore_index: int = None, class_weights: list = None) -> nn.Module:
    if loss == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss == "balanced_bce":
        return _balanced_binary_cross_entropy_with_logits
    elif loss == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
    elif loss == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss == "jaccard":
        return JaccardLoss(mode="multiclass")
    elif loss == "focal":
        return FocalLoss(mode="multiclass", normalized=True)
    else:
        raise ValueError(
            f"Loss type '{loss}' is not valid. Only 'bce', 'balanced_bce', 'ce', 'bce', 'jaccard', or "
            f"'focal' supported."
        )


class MultiLabelClassificationTask(ClassificationTask):
    def configure_losses(self) -> None:
        loss = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]

        class_weights = (
            torch.Tensor(self.hparams["class_weights"]) if self.hparams["class_weights"] is not None else None
        )

        if isinstance(loss, str):
            # Single loss
            self.criterion = init_loss(loss, ignore_index=ignore_index, class_weights=class_weights)
        elif isinstance(loss, nn.Module):
            # Custom loss
            self.criterion = loss
        elif isinstance(loss, list):
            # List of losses with equal weights
            losses = {loss: init_loss(loss, ignore_index=ignore_index, class_weights=class_weights) for loss in loss}
            self.criterion = CombinedLoss(losses=losses)
        elif isinstance(loss, dict):
            # Equal weighting of losses
            loss, weight = list(loss.keys()), list(loss.values())
            losses = {loss: init_loss(loss, ignore_index=ignore_index, class_weights=class_weights) for loss in loss}
            self.criterion = CombinedLoss(losses=losses, weight=weight)
        else:
            raise ValueError(
                f"The loss type {loss} isn't supported. Provide loss as string, list, or dict[name, weights]."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["model_args"]["num_classes"]
        ignore_index: int = self.hparams["ignore_index"]
        class_names = self.hparams["class_names"]
        metrics = MetricCollection(
            {
                "Multilabel_Accuracy": MultilabelAccuracy(
                    num_labels=num_classes, ignore_index=ignore_index, average="macro"
                ),
                "Multilabel_Accuracy_Micro": MultilabelAccuracy(
                    num_labels=num_classes, ignore_index=ignore_index, average="micro"
                ),
                "Multilabel_F1_Score": MultilabelF1Score(
                    num_labels=num_classes, ignore_index=ignore_index, average="macro"
                ),
                "Multilabel_Precision": MultilabelPrecision(
                    num_labels=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Multilabel_Recall": MultilabelRecall(
                    num_labels=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Multilabel_AUROC": MultilabelAUROC(
                    num_labels=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Class_Accuracy": ClasswiseWrapper(
                    MultilabelAccuracy(
                        num_labels=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                    prefix="Class_Accuracy_",
                ),
                "Class_F1": ClasswiseWrapper(
                    MultilabelF1Score(
                        num_labels=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                    prefix="Class_F1_",
                ),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        if self.hparams["test_dataloaders_names"] is not None:
            self.test_metrics = nn.ModuleList(
                [metrics.clone(prefix=f"test/{dl_name}/") for dl_name in self.hparams["test_dataloaders_names"]]
            )
        else:
            self.test_metrics = nn.ModuleList([metrics.clone(prefix="test/")])

    @staticmethod
    def to_multilabel_prediction(y: ModelOutput) -> Tensor:
        y_hat = y.output
        return torch.sigmoid(y_hat)

    def training_step(self, batch: object, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        x = batch["image"]
        y = batch["label"].to(torch.float32)
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}

        model_output: ModelOutput = self(x, **rest)
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat = self.to_multilabel_prediction(model_output)
        self.train_metrics.update(y_hat, y.to(torch.int32))

        return loss["loss"]

    def validation_step(self, batch: object, batch_idx: int, dataloader_idx: int = 0) -> None:
        x = batch["image"]
        y = batch["label"].to(torch.float32)
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat = self.to_multilabel_prediction(model_output)
        self.val_metrics.update(y_hat, y.to(torch.int32))

        if self._do_plot_samples(batch_idx):
            batch["prediction"] = y_hat
            self.plot_sample(batch, batch_idx)

    def test_step(self, batch: object, batch_idx: int, dataloader_idx: int = 0) -> None:
        x = batch["image"]
        y = batch["label"].to(torch.float32)
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        if dataloader_idx >= len(self.test_loss_handler):
            msg = "You are returning more than one test dataloader but not defining enough test_dataloaders_names."
            raise ValueError(msg)
        loss = self.test_loss_handler[dataloader_idx].compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.test_loss_handler[dataloader_idx].log_loss(
            partial(self.log, add_dataloader_idx=False),  # We don't need the dataloader idx as prefixes are different
            loss_dict=loss,
            batch_size=y.shape[0],
        )
        y_hat = self.to_multilabel_prediction(model_output)
        self.test_metrics[dataloader_idx].update(y_hat, y.to(torch.int32))

        if self._do_plot_samples(batch_idx):
            batch["prediction"] = y_hat
            self.plot_sample(batch, batch_idx)

    def predict_step(self, batch: object, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        file_names = batch["filename"] if "filename" in batch else None
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output = self(x, **rest)
        y_hat = self.to_multilabel_prediction(model_output)
        return y_hat, file_names
