"""Held-out splits for unsupervised model selection.

We never use ground-truth neuron types for hyperparameter selection. Instead:

* **LV / GNN-vSBM:** hold out a fixed set of directed pairs (positives + negatives)
  and score models by mean Bernoulli log-likelihood on that set
  (the likelihood term of the ELBO).
* **PCA:** hold out a fixed set of node rows and score by mean squared
  reconstruction error on those rows (lower is better; we store the negated
  value so that higher ``val_metric`` is always better across methods).

Count likelihoods (Poisson / negative binomial) produce log-likelihoods on a
scale that is not comparable with the Bernoulli one, so the selection metric
remains the held-out **Bernoulli** log-likelihood for every variant. The helpers
below convert a fitted count model into the implied probability of a present
edge, $P(y>0)=1-e^{-\\lambda}$ for Poisson and $P(y>0)=1-(r/(r+\\mu))^{r}$ for
the negative binomial, so that the binary held-out labels can be scored
directly. The native count log-likelihood is reported separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.stats import rankdata


def make_heldout_pairs(
    adj: csr_matrix,
    n_pos: int = 50_000,
    n_neg: int | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Sample a fixed held-out set of directed pairs for link likelihood."""
    if n_neg is None:
        n_neg = n_pos
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    coo = adj.tocoo()
    n_edges = int(coo.nnz)
    n_pos = min(n_pos, n_edges)

    pos_idx = rng.choice(n_edges, size=n_pos, replace=False)
    pos_src = coo.row[pos_idx].astype(np.int64)
    pos_tgt = coo.col[pos_idx].astype(np.int64)
    edge_set = {(int(s), int(t)) for s, t in zip(coo.row.tolist(), coo.col.tolist(), strict=True)}

    neg_src_list: list[np.ndarray] = []
    neg_tgt_list: list[np.ndarray] = []
    remaining = n_neg
    while remaining > 0:
        cand_src = rng.integers(0, n, size=max(remaining * 4, 4096), dtype=np.int64)
        cand_tgt = rng.integers(0, n, size=max(remaining * 4, 4096), dtype=np.int64)
        keep_s: list[int] = []
        keep_t: list[int] = []
        for s, t in zip(cand_src.tolist(), cand_tgt.tolist(), strict=True):
            if s == t or (s, t) in edge_set:
                continue
            keep_s.append(s)
            keep_t.append(t)
            if len(keep_s) >= remaining:
                break
        if not keep_s:
            continue
        neg_src_list.append(np.asarray(keep_s, dtype=np.int64))
        neg_tgt_list.append(np.asarray(keep_t, dtype=np.int64))
        remaining -= len(keep_s)

    neg_src = np.concatenate(neg_src_list)[:n_neg]
    neg_tgt = np.concatenate(neg_tgt_list)[:n_neg]

    src = np.concatenate([pos_src, neg_src])
    tgt = np.concatenate([pos_tgt, neg_tgt])
    y = np.concatenate([np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)])
    return {"src": src, "tgt": tgt, "y": y}


def make_heldout_rows(n: int, fraction: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(n * fraction)))
    return np.sort(rng.choice(n, size=n_val, replace=False).astype(np.int64))


def save_heldout(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    np.savez_compressed(path, **arrays)
    meta = {k: int(v.shape[0]) for k, v in arrays.items()}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def load_heldout(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def remove_heldout_positives(
    adj: csr_matrix,
    src: np.ndarray,
    tgt: np.ndarray,
    y: np.ndarray,
) -> csr_matrix:
    """Copy of ``adj`` with held-out positive edges removed (for GNN message passing)."""
    out = adj.tolil(copy=True)
    pos = y > 0.5
    for s, t in zip(src[pos].tolist(), tgt[pos].tolist(), strict=True):
        out[s, t] = 0
    out = out.tocsr()
    out.eliminate_zeros()
    return out


def heldout_pair_mask(
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    held_src: np.ndarray,
    held_tgt: np.ndarray,
) -> np.ndarray:
    """Return a (B_i, B_j) boolean mask of held-out pairs inside a dense block."""
    i_pos = {int(n): k for k, n in enumerate(idx_i.tolist())}
    j_pos = {int(n): k for k, n in enumerate(idx_j.tolist())}
    in_i = np.isin(held_src, idx_i, assume_unique=False)
    in_j = np.isin(held_tgt, idx_j, assume_unique=False)
    active = np.nonzero(in_i & in_j)[0]
    mask = np.zeros((len(idx_i), len(idx_j)), dtype=bool)
    for k in active.tolist():
        a = i_pos.get(int(held_src[k]))
        b = j_pos.get(int(held_tgt[k]))
        if a is not None and b is not None:
            mask[a, b] = True
    return mask


@torch.no_grad()
def mean_bernoulli_ll(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Mean Bernoulli log-likelihood (higher is better)."""
    return float((-F.binary_cross_entropy_with_logits(logits, y, reduction="mean")).item())


LIKELIHOODS = ("bernoulli", "poisson", "nb")
ETA_MIN = -30.0
ETA_MAX = 10.0


def native_ll_name(likelihood: str) -> str:
    """Metrics-file name for the native held-out log-likelihood of a likelihood."""
    return f"heldout_{likelihood}_ll"


def clamp_eta(eta: torch.Tensor) -> torch.Tensor:
    """Keep log-rates in a range where ``exp`` cannot overflow float32."""
    return torch.clamp(eta, min=ETA_MIN, max=ETA_MAX)


def lookup_pair_counts(adj: csr_matrix, src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Synapse counts of the given directed pairs from the *unbinarized* adjacency.

    Held-out pairs are masked out of the training objective, so reading their
    weights back is a label lookup rather than leakage.
    """
    return np.asarray(adj[src, tgt]).ravel().astype(np.float32)


def poisson_log_pmf(y: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """$\\log p(y\\mid\\lambda=e^{\\eta})$ for the Poisson."""
    eta = clamp_eta(eta)
    return y * eta - torch.exp(eta) - torch.lgamma(y + 1.0)


def nb_log_pmf(y: torch.Tensor, eta: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
    """$\\log p(y\\mid\\mu=e^{\\eta}, r=e^{\\log r})$ for the negative binomial."""
    eta = clamp_eta(eta)
    r = torch.exp(log_r)
    log_denom = torch.logaddexp(log_r, eta)
    return (
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * (log_r - log_denom)
        + y * (eta - log_denom)
    )


def poisson_edge_logit(eta: torch.Tensor) -> torch.Tensor:
    """Logit of $P(y>0)=1-e^{-\\lambda}$ implied by a Poisson log-rate."""
    lam = torch.exp(clamp_eta(eta))
    log_p = torch.log(torch.clamp(-torch.expm1(-lam), min=1e-12))
    return log_p + lam


def nb_edge_logit(eta: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
    """Logit of $P(y>0)=1-(r/(r+\\mu))^{r}$ implied by a negative binomial."""
    eta = clamp_eta(eta)
    r = torch.exp(log_r)
    log_q0 = torch.clamp(r * (log_r - torch.logaddexp(log_r, eta)), max=-1e-7)
    log_p = torch.log(torch.clamp(-torch.expm1(log_q0), min=1e-12))
    return log_p - log_q0


@torch.no_grad()
def mean_auc(scores: torch.Tensor, y: torch.Tensor) -> float:
    """ROC AUC of ``scores`` against binary ``y`` via the Mann-Whitney statistic.

    Unlike the Bernoulli log-likelihood this is invariant to any monotone
    recalibration, so it separates genuine ranking of edges above non-edges from
    a base rate inflated by training on dense subgraph blocks.
    """
    s = scores.flatten().detach().cpu().numpy().astype(np.float64)
    pos = y.flatten().detach().cpu().numpy() > 0.5
    n_pos = int(pos.sum())
    n_neg = int(pos.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Midranks matter: an untrained model predicts a near-constant score, and the
    # held-out file lists every positive before every negative, so breaking ties
    # by index would report AUC near 0 instead of 0.5.
    ranks = rankdata(s, method="average")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def write_metrics(path: Path | str, metrics: dict) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2))
