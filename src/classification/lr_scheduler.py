"""Linear-warmup + cosine-annealing learning-rate scheduler.

A self-contained re-implementation of
``pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR`` so that
the deprecated ``lightning-bolts`` package is not required as a runtime
dependency.

Behaviour
---------
``lr`` starts at ``warmup_start_lr`` and rises linearly to each parameter
group's base ``lr`` over the first ``warmup_epochs`` epochs, then
cosine-anneals down to ``eta_min`` over the remaining
``max_epochs - warmup_epochs`` epochs. Past ``max_epochs`` the LR is
clamped to ``eta_min`` (rather than restarting the cosine cycle as the
underlying ``CosineAnnealingLR`` would).

The trajectory is mathematically equivalent to ``pl_bolts``'s
implementation; the formulation here computes ``lr(epoch)`` directly
rather than recursively, which is easier to read.

Quick smoke test
----------------
$ python -m src.classification.lr_scheduler
"""
from __future__ import annotations

import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """Linear warmup → cosine annealing.

    Args:
        optimizer: Wrapped optimiser.
        warmup_epochs: Number of epochs in the linear-warmup phase.
            Must satisfy ``0 <= warmup_epochs <= max_epochs``.
        max_epochs: Total number of scheduled epochs (warmup + cosine).
        warmup_start_lr: LR at epoch 0. Default: ``0.0``.
        eta_min: Minimum LR reached at the end of cosine annealing.
            Default: ``0.0``.
        last_epoch: The index of the last epoch. Default: ``-1`` (start
            of training; matches the standard PyTorch convention).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}.")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}.")
        if warmup_epochs > max_epochs:
            raise ValueError(
                f"warmup_epochs ({warmup_epochs}) must be <= "
                f"max_epochs ({max_epochs})."
            )

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )

        epoch = self.last_epoch

        # ----- Phase 1: linear warmup -----
        if epoch < self.warmup_epochs:
            if self.warmup_epochs <= 1:
                # Degenerate case: no warmup or single-epoch warmup
                # — jump straight to base_lr.
                return list(self.base_lrs)
            return [
                self.warmup_start_lr
                + (base_lr - self.warmup_start_lr)
                * epoch
                / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        # ----- Phase 2: cosine annealing -----
        cosine_epochs = self.max_epochs - self.warmup_epochs
        if cosine_epochs == 0:
            return [self.eta_min for _ in self.base_lrs]

        progress = (epoch - self.warmup_epochs) / cosine_epochs
        progress = min(progress, 1.0)  # clamp past max_epochs
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    from torch.optim import Adam

    dummy = torch.nn.Linear(10, 1)
    opt = Adam(dummy.parameters(), lr=1e-3)
    sched = LinearWarmupCosineAnnealingLR(
        opt,
        warmup_epochs=5,
        max_epochs=100,
        warmup_start_lr=0.0,
        eta_min=1e-5,
    )

    # Note: the LR for the *first* training epoch is whatever the
    # scheduler computed at construction time (last_epoch=0).
    # Subsequent epochs use the value after each .step() call.
    lrs = [sched.get_last_lr()[0]]
    for _ in range(99):
        opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    print("warmup_epochs=5, max_epochs=100, base_lr=1e-3, eta_min=1e-5\n")
    markers = [0, 1, 4, 5, 6, 50, 99]
    for m in markers:
        print(f"  epoch {m:>3d}:  lr = {lrs[m]:.6e}")
    print("\nExpected: lr rises 0 → 1e-3 over epochs 0-4, then cosines down to ~1e-5.")