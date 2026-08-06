"""Held-out splits for unsupervised model selection.

We never use ground-truth neuron types for hyperparameter selection. Instead:

* **LV / GNN-vSBM:** hold out a fixed set of directed pairs (positives + negatives)
  and score models by mean Bernoulli log-likelihood on that set
  (the likelihood term of the ELBO).
* **PCA:** hold out a fixed set of node rows and score by mean squared
  reconstruction error on those rows (lower is better; we store the negated
  value so that higher ``val_metric`` is always better across methods).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix


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


def write_metrics(path: Path | str, metrics: dict) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2))
