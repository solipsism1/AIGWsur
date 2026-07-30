"""Learning-rate warmup and scheduler helpers for training."""

from __future__ import annotations

from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)


class LinearWarmupController:
    """Linearly ramp the learning rate from zero to the configured base LR."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int = 0):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps or 0))
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_step = 0

    def is_active(self) -> bool:
        return self.warmup_steps > 0 and self.last_step < self.warmup_steps

    def sync_optimizer(self):
        """Apply the LR implied by the current warmup state."""
        if self.warmup_steps <= 0:
            return
        if self.last_step >= self.warmup_steps:
            return

        scale = self.last_step / max(self.warmup_steps, 1)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale

    def step(self):
        """Advance warmup by one optimizer update."""
        if self.warmup_steps <= 0 or self.last_step >= self.warmup_steps:
            return

        self.last_step += 1
        scale = self.last_step / self.warmup_steps
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale

    def state_dict(self) -> dict:
        return {
            "warmup_steps": self.warmup_steps,
            "base_lrs": list(self.base_lrs),
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict: Optional[dict]):
        if not state_dict:
            return

        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.base_lrs = list(state_dict.get("base_lrs", self.base_lrs))
        self.last_step = int(state_dict.get("last_step", 0))
        self.sync_optimizer()


def build_scheduler(optimizer: Optimizer, train_cfg: dict):
    """Build an epoch-level LR scheduler from config."""
    scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
    scheduler_min_lr = float(train_cfg.get("scheduler_min_lr", 0.0))

    if scheduler_name in {"none", "off"}:
        return None, "none"

    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(train_cfg["max_epochs"])),
            eta_min=scheduler_min_lr,
        )
        return scheduler, scheduler_name

    if scheduler_name == "cosine_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, int(train_cfg.get("scheduler_restart_epochs", 250))),
            T_mult=max(1, int(train_cfg.get("scheduler_restart_mult", 2))),
            eta_min=scheduler_min_lr,
        )
        return scheduler, scheduler_name

    if scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_cfg.get("scheduler_factor", 0.5)),
            patience=max(1, int(train_cfg.get("scheduler_patience", 10))),
            threshold=float(train_cfg.get("scheduler_threshold", 1e-3)),
            min_lr=scheduler_min_lr,
        )
        return scheduler, scheduler_name

    if scheduler_name == "cosine_restart_best":
        # CosineAnnealingLR over one restart period. The training loop handles
        # reloading best model weights and rebuilding this scheduler every
        # restart_interval epochs.
        t_max = max(1, int(train_cfg.get("restart_interval", 500)))
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=scheduler_min_lr,
        )
        return scheduler, scheduler_name

    raise ValueError(
        "training.scheduler must be one of: none, cosine, cosine_restarts, plateau, cosine_restart_best"
    )


def step_scheduler(scheduler, scheduler_name: str, metric: Optional[float] = None):
    """Advance the epoch scheduler using the right calling convention."""
    if scheduler is None or scheduler_name == "none":
        return

    if scheduler_name == "plateau":
        if metric is None:
            raise ValueError("ReduceLROnPlateau requires a validation metric")
        scheduler.step(metric)
        return

    scheduler.step()
