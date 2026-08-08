"""Optimisation guards and best-checkpoint bookkeeping shared by the vSBM trainers.

Three distinct failure modes motivate this module.

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
Clipping only delays that divergence, however: it bounds the step, not the
target the step is walking towards.

*Unattainable Bernoulli optima.* With hard $0/1$ targets the maximiser of
$y\\log\\sigma(\\eta)+(1-y)\\log(1-\\sigma(\\eta))$ is $\\eta=\\pm\\infty$, so the
decoder can only keep improving its training objective by growing $|\\eta|$
without bound. ``smooth_binary_targets`` replaces $y$ by

$$ \\tilde y = (1-\\epsilon)\\,y + \\epsilon\\,p_0, $$

whose optimum is the *finite* logit $\\operatorname{logit}(\\tilde y)$; the
decoder therefore has somewhere to stop. The prior $p_0$ is selected by
``LABEL_SMOOTHING_TARGETS``:

``base_rate``
    $p_0$ is the empirical edge density. Then
    $\\mathbb{E}[\\tilde y] = (1-\\epsilon)p_0+\\epsilon p_0 = p_0$, i.e. the
    marginal is preserved exactly. This is the default.
``uniform``
    $p_0=1/2$, the textbook form, which for a graph of density $10^{-4}$ raises
    every negative target to $\\epsilon/2$ — three orders of magnitude above the
    true base rate — and biases the decoder towards predicting edges everywhere.
    Retained only so the two forms can be compared.

Smoothing is a Bernoulli device and is deliberately inert for the count
likelihoods; see ``resolve_label_smoothing``.
"""

from __future__ import annotations

import math
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


LABEL_SMOOTHING_TARGETS = ("base_rate", "uniform")


def resolve_label_smoothing(eps: float | None, likelihood: str) -> float:
    """Return the smoothing strength actually in force under ``likelihood``.

    Label smoothing interpolates a *binary* target towards a prior. Poisson and
    negative-binomial targets are synapse counts, for which that interpolation
    has no meaning: shrinking a count towards a probability would corrupt the
    sufficient statistic of the rate. Rather than invent an analogue, the flag
    is a no-op there and announces itself once.
    """
    eps = 0.0 if eps is None else float(eps)
    if eps < 0.0 or eps >= 1.0:
        raise ValueError(f"label smoothing must lie in [0, 1), got {eps}")
    if eps == 0.0:
        return 0.0
    if likelihood != "bernoulli":
        print(
            f"[label-smoothing] IGNORED: --label-smoothing {eps} is a Bernoulli "
            f"device and this run uses likelihood={likelihood}, whose targets are "
            "counts. Training proceeds unsmoothed."
        )
        return 0.0
    return eps


def smoothing_prior(target: str, base_rate: float) -> float:
    """The value $p_0$ that smoothed targets are pulled towards."""
    if target == "base_rate":
        return float(base_rate)
    if target == "uniform":
        return 0.5
    raise ValueError(
        f"unknown label-smoothing target {target!r}; expected one of {LABEL_SMOOTHING_TARGETS}"
    )


def smooth_binary_targets(
    y: torch.Tensor,
    eps: float,
    base_rate: float,
    target: str = "base_rate",
) -> torch.Tensor:
    """Return $(1-\\epsilon)y + \\epsilon p_0$, or ``y`` itself when disabled.

    The identity short-circuit at $\\epsilon=0$ is deliberate: the unsmoothed
    training path must remain the one that produced every result collected so
    far, not an arithmetically equivalent rewrite of it.
    """
    if eps <= 0.0:
        return y
    return y * (1.0 - eps) + eps * smoothing_prior(target, base_rate)


def smoothed_logit_bounds(
    eps: float,
    base_rate: float,
    target: str = "base_rate",
) -> tuple[float, float]:
    """Optimal logits $(\\eta^-,\\eta^+)$ for a smoothed negative and positive.

    These are the ceiling the smoothing imposes: with hard targets the pair is
    $(-\\infty,+\\infty)$, which is exactly what the diverging decoder chases.
    """
    if eps <= 0.0:
        return (-math.inf, math.inf)
    p0 = smoothing_prior(target, base_rate)
    t_neg = eps * p0
    t_pos = (1.0 - eps) + eps * p0
    return (math.log(t_neg / (1.0 - t_neg)), math.log(t_pos / (1.0 - t_pos)))


def label_smoothing_line(eps: float, base_rate: float, target: str) -> str:
    """One-line description of the smoothing in force, for the run header."""
    if eps <= 0.0:
        return "label_smoothing=0 (hard 0/1 targets; optimal logit is unbounded)"
    lo, hi = smoothed_logit_bounds(eps, base_rate, target)
    return (
        f"label_smoothing={eps} target={target} p0={smoothing_prior(target, base_rate):.3e} "
        f"=> optimal logits bounded to [{lo:.3f}, {hi:.3f}]"
    )


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
