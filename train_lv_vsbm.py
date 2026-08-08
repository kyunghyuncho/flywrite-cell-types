"""Low-rank variational SBM baseline with held-out likelihood for model selection.

Minibatches are square induced subgraphs drawn by
[`subgraph_sampler.SubgraphBatchSampler`](subgraph_sampler.py): a ``bfs_frac``
fraction of each node budget comes from breadth-first expansion over $A+A^\\top$
and the remainder is uniform, so the block contains orders of magnitude more
edges than an independently permuted rectangular block while the base rate stays
calibrated.

The expected complete log-likelihood is aggregated exactly on the $K\\times K$
cluster-pair grid. With ``--likelihood {poisson,nb}`` the adjacency is *not*
binarized and the same linear predictor $\\eta_{kk'}$ is read as a log-rate;
model selection nonetheless remains the held-out **Bernoulli** log-likelihood so
that ``val_metric`` is comparable across likelihoods.

The reported model is the one attaining the best ``val_metric`` over epochs, not
the last one: at large learning rates the decoder diverges mid-run and the final
parameters score at chance. Ground-truth agreement is recorded at *both* the
restored best checkpoint (``gt_*``) and the final epoch (``last_gt_*``) so the
cost of that divergence can be read off directly; neither quantity participates
in selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz
from torch import nn
from tqdm import tqdm

from evaluate_clustering import score_assignment_dict
from heldout import (
    LIKELIHOODS,
    clamp_eta,
    heldout_pair_mask,
    load_heldout,
    lookup_pair_counts,
    mean_auc,
    mean_bernoulli_ll,
    native_ll_name,
    nb_edge_logit,
    nb_log_pmf,
    poisson_edge_logit,
    poisson_log_pmf,
    write_metrics,
)
from index_mapping import load_mapping
from subgraph_sampler import SubgraphBatchSampler
from training_utils import (
    BestCheckpoint,
    checkpoint_metrics,
    prefixed,
    resolve_grad_clip,
    safe_optimizer_step,
)


class Snapshot(NamedTuple):
    """Everything reported about one parameter state (best checkpoint or last epoch)."""

    assignments: np.ndarray
    scores: np.ndarray
    assignment_dict: dict
    val_ll: float
    native_ll: float
    val_auc: float
    gt: dict[str, float] | None


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def log_clamp(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def load_adjacency(path: str, likelihood: str) -> tuple[csr_matrix, csr_matrix]:
    """Return ``(counts, training adjacency)``; the latter is binary only for Bernoulli."""
    counts = load_npz(path).tocsr().astype(np.float32)
    counts.eliminate_zeros()
    if likelihood != "bernoulli":
        return counts, counts
    binary = counts.copy()
    binary.data = (binary.data > 0).astype(np.float32)
    binary.eliminate_zeros()
    return counts, binary


def edge_prob_kk(likelihood: str, eta: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
    """$K\\times K$ probability of a present edge implied by the fitted likelihood."""
    if likelihood == "bernoulli":
        return torch.sigmoid(eta)
    if likelihood == "poisson":
        return torch.sigmoid(poisson_edge_logit(eta))
    return torch.sigmoid(nb_edge_logit(eta, log_r))


def expected_block_ll(
    likelihood: str,
    q: torch.Tensor,
    block: torch.Tensor,
    keep: torch.Tensor,
    eta: torch.Tensor,
    log_r: torch.Tensor,
) -> torch.Tensor:
    """$\\sum_{ij}\\mathbb{E}_{q}[\\log p(y_{ij})]$ aggregated over the cluster-pair grid.

    Terms that are constant in the parameters ($-\\log y_{ij}!$) are dropped.
    """
    if likelihood == "bernoulli":
        w_edge = q.T @ (block * keep) @ q
        w_non = q.T @ ((1.0 - block) * keep) @ q
        prob = torch.sigmoid(eta)
        return (w_edge * log_clamp(prob)).sum() + (w_non * log_clamp(1.0 - prob)).sum()

    eta = clamp_eta(eta)
    w_count = q.T @ (block * keep) @ q
    w_all = q.T @ keep @ q
    if likelihood == "poisson":
        return (w_count * eta).sum() - (w_all * torch.exp(eta)).sum()

    r = torch.exp(log_r)
    log_denom = torch.logaddexp(log_r, eta)
    grid = (w_all * (r * (log_r - log_denom))).sum() + (w_count * (eta - log_denom)).sum()
    # lgamma(y+r) - lgamma(r) does not depend on (k,k') but does depend on r.
    direct = ((torch.lgamma(block + r) - torch.lgamma(r)) * keep).sum()
    return grid + direct


@torch.no_grad()
def eval_heldout(
    q_logits: torch.Tensor,
    u_left: torch.Tensor,
    u_right: torch.Tensor,
    bias: torch.Tensor,
    log_r: torch.Tensor,
    held: dict[str, np.ndarray],
    counts: np.ndarray | None,
    likelihood: str,
    device: str,
    dtype: torch.dtype,
    chunk: int = 8192,
) -> tuple[float, float, float]:
    """Held-out (Bernoulli LL, native LL, AUC) under $\\mathbb{E}_{q_i q_j}$."""
    alpha = torch.softmax(q_logits, dim=-1)
    eta = u_left @ u_right.T + bias
    prob_kk = edge_prob_kk(likelihood, eta, log_r)
    src = held["src"]
    tgt = held["tgt"]
    y = torch.tensor(held["y"], dtype=dtype, device=device)
    lls = []
    edge_logits = []
    for start in range(0, len(src), chunk):
        sl = slice(start, start + chunk)
        ai = alpha[src[sl]]
        aj = alpha[tgt[sl]]
        p = ((ai @ prob_kk) * aj).sum(dim=-1).clamp(1e-6, 1 - 1e-6)
        logits = torch.log(p) - torch.log1p(-p)
        lls.append(mean_bernoulli_ll(logits, y[sl]))
        edge_logits.append(logits)
    bern_ll = float(np.mean(lls))
    auc = mean_auc(torch.cat(edge_logits), y)
    if likelihood == "bernoulli" or counts is None:
        return bern_ll, bern_ll, auc

    # E_q[log p(y)] = sum_{kk'} alpha_i(k) alpha_j(k') log p(y | eta_{kk'}); the
    # grid depends on y, so group the held-out pairs by their (small) count.
    total = 0.0
    for value in np.unique(counts):
        sel = np.nonzero(counts == value)[0]
        y_val = torch.tensor(float(value), dtype=dtype, device=device)
        if likelihood == "poisson":
            lp_kk = poisson_log_pmf(y_val, eta)
        else:
            lp_kk = nb_log_pmf(y_val, eta, log_r)
        for start in range(0, len(sel), chunk):
            sub = sel[start : start + chunk]
            ai = alpha[src[sub]]
            aj = alpha[tgt[sub]]
            total += float(((ai @ lp_kk) * aj).sum(dim=-1).sum().item())
    return bern_ll, total / max(len(counts), 1), auc


def train(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    dtype = torch.float32
    torch.manual_seed(args.seed)

    adj_counts, adj = load_adjacency(args.adjacency, args.likelihood)
    n = adj.shape[0]
    nnz = int(adj.nnz)
    print(f"Loaded A with shape {adj.shape}, nnz={nnz}, device={device}")
    print(
        f"LV: d={args.d} lr={args.lr} likelihood={args.likelihood} "
        f"bfs_frac={args.bfs_frac} bfs_seeds={args.bfs_seeds}"
    )

    held = load_heldout(Path(args.heldout_pairs))
    held_src_np = held["src"]
    held_tgt_np = held["tgt"]
    held_counts = (
        None
        if args.likelihood == "bernoulli"
        else lookup_pair_counts(adj_counts, held_src_np, held_tgt_np)
    )

    u_left = nn.Parameter(
        (1.0 / np.sqrt(args.k * args.d)) * torch.randn(args.k, args.d, dtype=dtype, device=device)
    )
    u_right = nn.Parameter(
        (1.0 / np.sqrt(args.k * args.d)) * torch.randn(args.k, args.d, dtype=dtype, device=device)
    )
    bias = nn.Parameter(
        torch.log(torch.tensor([max(float(adj.mean()), 1e-8)], dtype=dtype, device=device))
    )
    q_logits = nn.Parameter((1.0 / args.k) * torch.randn(n, args.k, dtype=dtype, device=device))
    log_r = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))

    params = [u_left, u_right, bias, q_logits]
    if args.likelihood == "nb":
        params.append(log_r)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    grad_clip = resolve_grad_clip(args.grad_clip)
    state = {
        "u_left": u_left,
        "u_right": u_right,
        "bias": bias,
        "q_logits": q_logits,
        "log_r": log_r,
    }
    sampler = SubgraphBatchSampler(
        adj,
        batch_size=args.minibatch,
        bfs_frac=args.bfs_frac,
        n_seeds=args.bfs_seeds,
        seed=args.seed,
    )
    checkpoint = BestCheckpoint()
    cum_pos_obs = 0
    total_updates = 0
    n_skipped = 0
    last_epoch = -1
    # A coverage-defined epoch is longer at higher bfs_frac, so an update budget is
    # what makes compute comparable across bfs_frac.
    epochs = args.epochs if args.target_updates is None else 10**9

    try:
        for epoch in range(epochs):
            if args.target_updates is not None and total_updates >= args.target_updates:
                break
            loss_sum = 0.0
            n_scored = 0
            n_batches = 0
            epoch_pos_obs = 0

            for idx in tqdm(sampler.epoch(), desc=f"lv epoch {epoch}"):
                if args.max_updates is not None and n_batches >= args.max_updates:
                    break
                if args.target_updates is not None and total_updates >= args.target_updates:
                    break
                block = torch.tensor(adj[idx][:, idx].toarray(), dtype=dtype, device=device)
                mask = torch.tensor(
                    heldout_pair_mask(idx, idx, held_src_np, held_tgt_np), device=device
                )
                keep = ~mask
                # The graph has no self-loops; do not train on the diagonal.
                keep.fill_diagonal_(False)
                keep = keep.to(dtype)
                n_keep = keep.sum()
                if float(n_keep.item()) == 0.0:
                    continue
                n_batches += 1
                total_updates += 1
                epoch_pos_obs += int(((block > 0).to(dtype) * keep).sum().item())

                idx_t = torch.from_numpy(idx).to(device)
                q = torch.softmax(q_logits[idx_t], dim=-1)
                eta = u_left @ u_right.T + bias
                obj = expected_block_ll(args.likelihood, q, block, keep, eta, log_r)
                obj = obj - (q * log_clamp(q)).sum(1).mean()
                # Normalize roughly by #kept pairs so loss scale is stable.
                obj = obj * (idx.size * idx.size) / n_keep.clamp(min=1)
                loss = -obj

                if not safe_optimizer_step(loss, params, optimizer, grad_clip):
                    n_skipped += 1
                    continue
                loss_sum += float(loss.item())
                n_scored += 1

            mean_loss = loss_sum / max(n_scored, 1)
            cum_pos_obs += epoch_pos_obs
            last_epoch = epoch
            print(sampler.coverage_line(epoch, n_batches, epoch_pos_obs, cum_pos_obs, nnz))
            val_ll, _, val_auc = eval_heldout(
                q_logits, u_left, u_right, bias, log_r, held, None, args.likelihood, device, dtype
            )
            checkpoint.update(val_ll, epoch, state)
            n_used = int(torch.unique(torch.argmax(q_logits, dim=-1)).numel())
            print(
                f"loss={mean_loss:.4f} val_ll={val_ll:.6f} val_auc={val_auc:.4f} "
                f"best_val_ll={checkpoint.metric:.6f}@{checkpoint.epoch} "
                f"clusters={n_used}/{args.k} "
                f"updates={total_updates} skipped={n_skipped}"
            )
    except KeyboardInterrupt:
        print("Training interrupted.")

    mapping = load_mapping(args.mapping)
    gt_path = None if args.no_gt_eval else args.gt

    def snapshot() -> Snapshot:
        with torch.no_grad():
            assignments = torch.argmax(q_logits, dim=-1).cpu().numpy()
            scores = torch.max(torch.softmax(q_logits, dim=-1), dim=-1).values.cpu().numpy()
            val_ll, native_ll, val_auc = eval_heldout(
                q_logits,
                u_left,
                u_right,
                bias,
                log_r,
                held,
                held_counts,
                args.likelihood,
                device,
                dtype,
            )
        pred = {mapping[i]: int(assignments[i]) for i in range(len(assignments))}
        return Snapshot(
            assignments,
            scores,
            pred,
            val_ll,
            native_ll,
            val_auc,
            score_assignment_dict(pred, gt_path),
        )

    # Evaluate the final parameters before restoring, so the cost of any mid-run
    # divergence is measurable rather than inferred.
    last = snapshot()
    restored = checkpoint.epoch != last_epoch and checkpoint.restore(state)
    best = snapshot() if restored else last

    prefix = Path(args.out_prefix)
    np.save(f"{prefix}_assignments.npy", best.assignments)
    np.save(f"{prefix}_scores.npy", best.scores)
    np.save(f"{prefix}_assignment_dict.npy", best.assignment_dict)
    if best is not last:
        np.save(f"{prefix}_last_assignment_dict.npy", last.assignment_dict)
    metrics = {
        "method": "lv_vsbm",
        "val_metric": best.val_ll,
        "val_metric_name": "heldout_bernoulli_ll",
        "val_metric_higher_is_better": True,
        "val_native_ll": best.native_ll,
        "val_native_ll_name": native_ll_name(args.likelihood),
        "val_auc": best.val_auc,
        "last_val_metric": last.val_ll,
        "last_val_native_ll": last.native_ll,
        "last_val_auc": last.val_auc,
        **checkpoint_metrics(checkpoint, last_epoch),
        **prefixed(best.gt, "gt_"),
        **prefixed(last.gt, "last_gt_"),
        "seed": args.seed,
        "k": args.k,
        "d": args.d,
        "lr": args.lr,
        "epochs": args.epochs,
        "likelihood": args.likelihood,
        "bfs_frac": args.bfs_frac,
        "bfs_seeds": args.bfs_seeds,
        "grad_clip": grad_clip,
        "edges_seen_x_nnz": cum_pos_obs / max(nnz, 1),
        "total_updates": total_updates,
        "skipped_updates": n_skipped,
        "target_updates": args.target_updates,
        "nb_r": float(torch.exp(log_r).item()) if args.likelihood == "nb" else None,
        "n_pred_clusters": int(len(np.unique(best.assignments))),
        "last_n_pred_clusters": int(len(np.unique(last.assignments))),
    }
    write_metrics(f"{prefix}_metrics.json", metrics)
    print(
        f"Saved {prefix}_* from the {metrics['checkpoint']} model "
        f"(epoch {metrics['best_epoch']}); val_ll={best.val_ll:.6f} "
        f"native_ll={best.native_ll:.6f} val_auc={best.val_auc:.4f} "
        f"| last val_ll={last.val_ll:.6f} val_auc={last.val_auc:.4f}"
    )
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adjacency", default="sparse_connectivity_matrix.npz")
    p.add_argument("--mapping", default="root_id_to_index_mapping.json")
    p.add_argument("--heldout-pairs", default="heldout_pairs.npz")
    p.add_argument("--out-prefix", default="lv")
    p.add_argument("--k", type=int, default=729)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max-updates", type=int, default=None, help="Cap on updates per epoch")
    p.add_argument(
        "--target-updates",
        type=int,
        default=None,
        help="Total update budget, overriding --epochs; matches compute across bfs_frac",
    )
    p.add_argument("--lr", type=float, default=1e-1)
    p.add_argument(
        "--bfs-frac",
        type=float,
        default=0.5,
        help="Fraction of each minibatch grown by BFS; 0 recovers uniform sampling",
    )
    p.add_argument("--bfs-seeds", type=int, default=4, help="BFS seeds per expansion round")
    p.add_argument(
        "--likelihood",
        choices=LIKELIHOODS,
        default="bernoulli",
        help="Edge likelihood; count models use unbinarized synapse counts",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Global gradient-norm clip applied before each step; 0 or less disables it",
    )
    p.add_argument(
        "--gt",
        default="root_id_type_dict.pkl",
        help="Ground-truth types, scored as a diagnostic only (never used for selection)",
    )
    p.add_argument(
        "--no-gt-eval",
        action="store_true",
        help="Skip the diagnostic ground-truth scoring of the best and last models",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.seed is None:
        args.seed = int(datetime.now().timestamp())
    train(args)


if __name__ == "__main__":
    main()
