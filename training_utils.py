"""Optimisation guards and best-checkpoint bookkeeping shared by the vSBM trainers.

Two distinct failure modes motivate this module.

*Non-finite updates.* A single non-finite loss or gradient propagates into the
Adam moments and from there into every parameter, so the run keeps going and
silently reports NaN metrics instead of failing loudly. ``safe_optimizer_step``
skips such an update and counts it.

*Mid-run decoder divergence.* At large learning rates the edge decoder can
diverge after several healthy epochs: the held-out AUC falls to $0.5$ and never
recovers, so the last-epoch parameters are not the parameters the validation
metric selected. ``BestCheckpoint`` retains the state attaining the best
validation metric so that the reported model is the selected one.

Global-norm clipping happens inside the same helper. ``resolve_grad_clip`` maps
a non-positive threshold to ``None``, meaning "compute the norm but never
rescale", so the finiteness guard stays active even with clipping disabled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def resolve_grad_clip(threshold: float | None) -> float | None:
    """Return the clipping threshold, or ``None`` when a non-positive value disables it."""
    if threshold is None or threshold <= 0.0:
        return None
    return float(threshold)


def safe_optimizer_step(
    loss: torch.Tensor,
    params: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    grad_clip: float | None,
) -> bool:
    """Backward, clip to ``grad_clip`` in global norm, then step.

    Returns ``False`` (leaving the parameters untouched) when the loss or the
    gradient is non-finite.
    """
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return False

    optimizer.zero_grad()
    loss.backward()
    # An infinite threshold still computes the norm and never rescales, which is
    # what keeps the guard below meaningful when clipping is switched off.
    max_norm = float("inf") if grad_clip is None else grad_clip
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return False

    optimizer.step()
    return True


class BestCheckpoint:
    """Track the parameter state attaining the highest value of a validation metric."""

    def __init__(self) -> None:
        self.metric: float = -float("inf")
        self.epoch: int = -1
        self._state: dict[str, torch.Tensor] | None = None

    @property
    def is_set(self) -> bool:
        return self._state is not None

    @torch.no_grad()
    def update(self, metric: float, epoch: int, tensors: Mapping[str, torch.Tensor]) -> bool:
        """Snapshot ``tensors`` if ``metric`` improves on the best seen so far."""
        if self._state is not None and not metric > self.metric:
            return False
        self.metric = float(metric)
        self.epoch = int(epoch)
        self._state = {name: t.detach().clone() for name, t in tensors.items()}
        return True

    @torch.no_grad()
    def restore(self, tensors: Mapping[str, torch.Tensor]) -> bool:
        """Copy the retained state back into ``tensors`` in place."""
        if self._state is None:
            return False
        for name, t in tensors.items():
            t.copy_(self._state[name])
        return True


def checkpoint_metrics(checkpoint: BestCheckpoint, last_epoch: int) -> dict[str, object]:
    """Provenance of the parameters that produced the saved assignments."""
    restored = checkpoint.is_set
    return {
        "checkpoint": "best_val_metric" if restored else "last_epoch",
        "assignment_source": "best_val_metric" if restored else "last_epoch",
        "best_epoch": checkpoint.epoch if restored else last_epoch,
        "last_epoch": last_epoch,
    }


def prefixed(metrics: Mapping[str, float] | None, prefix: str) -> dict[str, float]:
    """Re-key a metrics mapping, e.g. ``hungarian`` -> ``last_gt_hungarian``."""
    if not metrics:
        return {}
    return {f"{prefix}{k}": v for k, v in metrics.items()}
