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

*Unbounded block logits.* Smoothing bounds the *optimum*; it does not bound the
*iterate*. The block predictor $\\eta_{kk'} = u_k^\\top v_{k'} + b$ is bilinear
in unconstrained embeddings, so nothing stops $\\lVert u_k\\rVert$ from growing:
at $d=256$, $\\mathrm{lr}=0.1$ the observed $\\max|\\eta|$ runs $15 \\to 125 \\to
1129 \\to 3030$ over four epochs, at which point float32 $\\sigma(\\eta)$ is
exactly $0$ or $1$, every gradient is exactly zero and the parameters freeze.
``unit_rows`` and ``BlockScale`` replace the bilinear form by a scaled cosine
similarity,

$$ \\eta_{kk'} = s\\,\\hat u_k^\\top \\hat v_{k'} + b,
   \\qquad \\lVert\\hat u_k\\rVert = \\lVert\\hat v_{k'}\\rVert = 1, $$

for which $|\\eta_{kk'} - b| \\le s$ holds *structurally* rather than as
something the optimiser has to be persuaded of. Normalisation happens in the
forward pass, so gradients flow through it and no post-hoc projection of the
parameters is required.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn


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


U_NORMS = ("none", "unit")
U_SCALES = ("fixed", "learned", "per_row")

# The bias carries the base rate, $b = \log(1.5\times10^{-4}) \approx -8.8$, and
# the scale is the entire budget the decoder has for departing from it. A block
# of density 0.1 sits at $\eta \approx -2.2$, i.e. $+6.6$ above $b$; the densest
# blocks observed under BFS sampling need rather less. Eight logits of headroom
# therefore spans the range the data actually occupy while capping $|\eta|$ near
# $17$ -- an order of magnitude below where float32 $\sigma$ saturates, and two
# orders below the values the unconstrained decoder reached.
DEFAULT_U_SCALE_INIT = 8.0


def unit_rows(u: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise $u/\\lVert u\\rVert$, differentiable and finite at $u=0$.

    The guard lives *inside* the square root rather than as a clamp on the norm:
    $u/\\max(\\lVert u\\rVert,\\varepsilon)$ is finite at the origin but its
    gradient is not, whereas $u/\\sqrt{\\lVert u\\rVert^2+\\varepsilon}$ is smooth
    everywhere and still satisfies $\\lVert\\hat u\\rVert \\le 1$, which is what
    the logit bound rests on.
    """
    return u * torch.rsqrt(u.pow(2).sum(dim=-1, keepdim=True) + eps)


class BlockScale:
    """The positive scale $s$ multiplying unit-normalised cluster embeddings.

    ``fixed``
        $s$ is a constant. There is no parameter, hence nothing to diverge, and
        $|\\eta_{kk'} - b| \\le s$ holds for the whole run.
    ``learned``
        A single scalar, carried as $\\log s$ so that $s>0$ by construction. The
        bound becomes $|\\eta_{kk'} - b| \\le s_T$ for whatever $s_T$ the run
        ends at, which is only useful if $s$ itself stays bounded -- hence
        ``mean_value`` is logged every epoch.
    ``per_row``
        $K$ scalars, one per left cluster. This is a strictly weaker guarantee:
        the ceiling is $\\max_k s_k$, so a single runaway row suffices to
        reintroduce the saturation this construction exists to prevent.

    The scale multiplies the left factor only, so $\\eta_{kk'} = s_k\\,\\hat
    u_k^\\top\\hat v_{k'}$; for the scalar modes that is identical to scaling the
    product.
    """

    def __init__(
        self,
        mode: str,
        init: float,
        k: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> None:
        if mode not in U_SCALES:
            raise ValueError(f"unknown u-scale mode {mode!r}; expected one of {U_SCALES}")
        if not init > 0.0:
            raise ValueError(f"u-scale-init must be positive, got {init}")
        self.mode = mode
        self.init = float(init)
        self.log_scale: nn.Parameter | None = None
        if mode == "fixed":
            return
        rows = 1 if mode == "learned" else k
        self.log_scale = nn.Parameter(
            torch.full((rows, 1), math.log(self.init), dtype=dtype, device=device)
        )

    @property
    def is_learnable(self) -> bool:
        return self.log_scale is not None

    @property
    def parameters(self) -> list[nn.Parameter]:
        return [] if self.log_scale is None else [self.log_scale]

    def factor(self) -> torch.Tensor | float:
        """Broadcastable multiplier of shape ``(1, 1)`` / ``(K, 1)``, or a constant."""
        return self.init if self.log_scale is None else torch.exp(self.log_scale)

    @torch.no_grad()
    def values(self) -> torch.Tensor:
        if self.log_scale is None:
            return torch.tensor([self.init])
        return torch.exp(self.log_scale.detach()).flatten().cpu()

    def mean_value(self) -> float:
        return float(self.values().mean().item())

    def max_value(self) -> float:
        return float(self.values().max().item())

    def line(self) -> str:
        if not self.is_learnable:
            return f"u_scale={self.init:.4f}"
        vals = self.values()
        if vals.numel() == 1:
            return f"u_scale={float(vals.item()):.4f}"
        return (
            f"u_scale_mean={float(vals.mean().item()):.4f} "
            f"u_scale_min={float(vals.min().item()):.4f} "
            f"u_scale_max={float(vals.max().item()):.4f}"
        )


def make_block_scale(
    u_norm: str,
    mode: str,
    init: float,
    k: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> BlockScale | None:
    """A ``BlockScale`` under ``--u-norm unit``, and ``None`` otherwise.

    Returning ``None`` rather than a unit scale is deliberate: under
    ``--u-norm none`` the decoder must execute the *same* arithmetic it always
    has, not an algebraically equivalent rewrite of it.
    """
    if u_norm not in U_NORMS:
        raise ValueError(f"unknown u-norm {u_norm!r}; expected one of {U_NORMS}")
    if u_norm == "none":
        return None
    return BlockScale(mode, init, k, dtype, device)


def scaled_block_embeddings(
    u_left: torch.Tensor,
    u_right: torch.Tensor,
    scale: BlockScale | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Effective cluster embeddings entering the bilinear decoder.

    With ``scale=None`` the inputs are returned untouched -- the unconstrained
    path is bit-for-bit the historical one.
    """
    if scale is None:
        return u_left, u_right
    return unit_rows(u_left) * scale.factor(), unit_rows(u_right)


def u_norm_line(u_norm: str, scale: BlockScale | None, bias_init: float) -> str:
    """One-line description of the block-logit constraint, for the run header."""
    if scale is None:
        return f"u_norm=none (block logits unbounded; bias init {bias_init:.3f})"
    s = scale.max_value()
    return (
        f"u_norm=unit u_scale={scale.mode} init={scale.init} "
        f"=> |eta - b| <= {s:.3f}, eta in [{bias_init - s:.3f}, {bias_init + s:.3f}] "
        f"at the initial bias"
    )


def decoder_line(epoch: int, bias: float, scale: BlockScale | None) -> str:
    """Per-epoch state of the two terms that can still grow without bound."""
    tail = "" if scale is None else f" {scale.line()}"
    return f"[decoder] epoch={epoch} bias={bias:.4f}{tail}"


def block_scale_metrics(scale: BlockScale | None, prefix: str = "") -> dict[str, float | None]:
    """Realised scale, reported at whichever parameter state is current."""
    if scale is None:
        return {f"{prefix}u_scale_value": None, f"{prefix}u_scale_value_max": None}
    return {
        f"{prefix}u_scale_value": scale.mean_value(),
        f"{prefix}u_scale_value_max": scale.max_value(),
    }


def add_u_norm_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach ``--u-norm`` / ``--u-scale`` / ``--u-scale-init`` to a trainer."""
    parser.add_argument(
        "--u-norm",
        choices=U_NORMS,
        default="none",
        help=(
            "Constraint on the cluster embeddings entering the block logit. "
            "'none' (default) leaves them unconstrained, reproducing every run "
            "made so far; 'unit' normalises each row in the forward pass, which "
            "bounds |eta - bias| by the scale below"
        ),
    )
    parser.add_argument(
        "--u-scale",
        choices=U_SCALES,
        default="fixed",
        help=(
            "Parameterisation of the scale applied under --u-norm unit: a "
            "constant ('fixed'), one learned scalar ('learned', carried as its "
            "log so it stays positive), or K learned per-cluster scales "
            "('per_row', whose ceiling is only max_k s_k). Inert under --u-norm none"
        ),
    )
    parser.add_argument(
        "--u-scale-init",
        type=float,
        default=DEFAULT_U_SCALE_INIT,
        help=(
            "Fixed value, or initialisation of the learned scale. The default "
            "gives the decoder ~8 logits of travel either side of the base-rate "
            "bias, enough to express block densities from far below the base "
            "rate up to ~0.3, while keeping |eta| an order of magnitude short of "
            "float32 sigmoid saturation"
        ),
    )
    return parser


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
