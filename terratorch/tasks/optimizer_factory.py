from typing import Any

import lightning
import torch.optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def LambdaFns(milestone: int, warmup_type: str = "linear_warmup"):
    def linear_warmup(current_step: int):
        return float((current_step + 1) / milestone)

    # add cosine warmup fn

    if warmup_type == "linear_warmup":
        return linear_warmup


def optimizer_factory(
    optimizer: str,
    lr: float,
    params_to_be_optimized: Any,
    optimizer_hparams: dict | None = None,
    scheduler: str | None = None,
    monitor: str = "val_loss",
    scheduler_hparams: dict | None = None,
) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
    """Instantiates an optimizer with a scheduler

    Args:
        optimizer (str): Name of optimizer class from torch.optim.
        lr (float): Learning rate
        params_to_be_optimized (Any): Parameters to be optimized
        optimizer_hparams (dict | None, optional): Parameters to be passed to the optimizer constructor.
            Defaults to None.
        scheduler (str | None, optional): Name of scheduler class from torch.optim.lr_scheduler. Defaults to None.
        monitor (str, optional): Quantity for the scheduler to monitor. Defaults to "val_loss".
        scheduler_hparams (dict | None, optional): Parameters to be passed to the scheduler constructor.
            Defaults to None.

    Returns:
        lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig
    """
    optimizer_hparams = optimizer_hparams if optimizer_hparams else {}
    optimizer = getattr(torch.optim, optimizer)
    optimizer = optimizer(params_to_be_optimized, lr=lr, **optimizer_hparams)

    if scheduler is None:
        return {"optimizer": optimizer}

    scheduler_hparams = scheduler_hparams if scheduler_hparams else {}
    interval = scheduler_hparams.get("interval", "epoch")
    scheduler_hparams_no_interval = {k: v for k, v in scheduler_hparams.items() if k != "interval"}

    # unpack sequential schedule
    if scheduler == "SequentialLR":
        assert "schedulers" in scheduler_hparams_no_interval, "Please provide scheduler for SequentialLR"
        num_schedulers = len(scheduler_hparams_no_interval["schedulers"])
        if num_schedulers > 1:
            # if using more than scheduler, milestones must be defined
            assert "milestones" in scheduler_hparams_no_interval, "Please provide milestones for SequentialLR"
            expected_milestones = num_schedulers - 1
            assert len(scheduler_hparams_no_interval["milestones"]) == expected_milestones, (
                "Please provide 1 milestone for each transition"
            )
            assert scheduler_hparams_no_interval["milestones"][0] >= 1, "The first milestone must be greater than 0"
            if expected_milestones > 1:
                check_progression = [
                    expected_milestones[i] < expected_milestones[i + 1] for i in range(expected_milestones - 1)
                ]
                assert all(check_progression), "Each milestone must be greater than the previous one"

        schedule_sequence = []
        for i, (key, value) in enumerate(scheduler_hparams_no_interval["schedulers"].items()):
            if key == "LambdaLR":
                lr_lambda = value.get("lr_lambda", "linear_warmup")
                milestone = scheduler_hparams_no_interval["milestones"][i]
                value["lr_lambda"] = LambdaFns(milestone, lr_lambda)
            nested_scheduler = getattr(torch.optim.lr_scheduler, key)
            nested_scheduler = nested_scheduler(optimizer, **value)
            schedule_sequence.append(nested_scheduler)
        scheduler_hparams_no_interval["schedulers"] = schedule_sequence
    scheduler = getattr(torch.optim.lr_scheduler, scheduler)
    scheduler = scheduler(optimizer, **scheduler_hparams_no_interval)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "monitor": monitor, "interval": interval},
    }
