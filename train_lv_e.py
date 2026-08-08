"""Low-rank vSBM with node residuals e_i (no GNN).

Edge logits:

    logit_ij = (alpha_i U_s) · (alpha_j U_t) + b + e_i · e_j

``e_i`` captures within-type / degree-like residual connectivity. Weight decay
is applied **only** to ``e`` (Adam param group) so residuals stay small and do
not absorb cluster identity. Clusters are still read from ``argmax(alpha)``.

Minibatches are square induced subgraphs from
[`subgraph_sampler.SubgraphBatchSampler`](subgraph_sampler.py). Unlike the
aggregated LV objective this trainer scores pairs directly, so ``--likelihood
{poisson,nb}`` simply reinterprets ``logit_ij`` as a log-rate over unbinarized
synapse counts. Selection stays on the held-out Bernoulli log-likelihood.

As in [`train_lv_vsbm.py`](train_lv_vsbm.py), the reported model is the epoch
attaining the best ``val_metric`` rather than the last one, and ground-truth
agreement is recorded at both that checkpoint (``gt_*``) and the final epoch
(``last_gt_*``) as a diagnostic that never feeds selection. ``--label-smoothing``
likewise behaves as it does there: it caps the optimal Bernoulli logit at a
finite value, applies to the **training loss only**, and is inert under the
count likelihoods.

``--u-norm unit`` bounds the *cluster* part of the logit by normalising the
prototypes in the forward pass, giving $|(\\alpha_i U_s)\\cdot(\\alpha_j U_t)|
\\le s$. Note that here it is only a partial guarantee: the residual term
$e_i\\cdot e_j$ is left unconstrained, and is held in check by ``--e-wd``
instead. ``--u-norm none`` (the default) changes nothing.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, load_npz
from torch import nn
from tqdm import tqdm

from evaluate_clustering import score_assignment_dict
from heldout import (
    LIKELIHOODS,
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
    LABEL_SMOOTHING_TARGETS,
    BestCheckpoint,
    BlockScale,
    add_u_norm_arguments,
    block_scale_metrics,
    checkpoint_metrics,
    decoder_line,
    label_smoothing_line,
    make_block_scale,
    prefixed,
    resolve_grad_clip,
    resolve_label_smoothing,
    safe_optimizer_step,
    scaled_block_embeddings,
    smooth_binary_targets,
    u_norm_line,
)


class Snapshot(NamedTuple):
    """Everything reported about one parameter state (best checkpoint or last epoch)."""

    assignments: np.ndarray
    scores: np.ndarray
    assignment_dict: dict
    val_ll: float
    native_ll: float
    val_auc: float
    e_rms: float
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


def pair_log_lik(
    likelihood: str, logits: torch.Tensor, target: torch.Tensor, log_r: torch.Tensor
) -> torch.Tensor:
    """Per-pair log-likelihood; ``logits`` is a logit (Bernoulli) or a log-rate."""
    if likelihood == "bernoulli":
        return -F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if likelihood == "poisson":
        return poisson_log_pmf(target, logits)
    return nb_log_pmf(target, logits, log_r)


def pair_edge_logit(likelihood: str, logits: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
    """Logit of $P(y>0)$ implied by the fitted likelihood."""
    if likelihood == "bernoulli":
        return logits
    if likelihood == "poisson":
        return poisson_edge_logit(logits)
    return nb_edge_logit(logits, log_r)


@torch.no_grad()
def eval_heldout(
    q_logits: torch.Tensor,
    u_left: torch.Tensor,
    u_right: torch.Tensor,
    bias: torch.Tensor,
    e: torch.Tensor,
    log_r: torch.Tensor,
    held: dict[str, np.ndarray],
    counts: np.ndarray | None,
    likelihood: str,
    device: str,
    dtype: torch.dtype,
    chunk: int = 8192,
    scale: BlockScale | None = None,
) -> tuple[float, float, float]:
    """Held-out (Bernoulli LL, native LL, AUC) for the fitted model.

    INVARIANT: this function scores the *true* unsmoothed $0/1$ labels
    ``held["y"]``. Label smoothing is a property of the training loss alone and
    must never reach this path -- note that it takes no smoothing argument, and
    do not add one. ``val_metric`` and ``val_auc`` are therefore directly
    comparable across every run ever made, smoothed or not.
    """
    alpha = torch.softmax(q_logits, dim=-1)
    eff_left, eff_right = scaled_block_embeddings(u_left, u_right, scale)
    src = held["src"]
    tgt = held["tgt"]
    y = torch.tensor(held["y"], dtype=dtype, device=device)
    y_count = None if counts is None else torch.tensor(counts, dtype=dtype, device=device)
    lls = []
    native = []
    edge_logits = []
    for start in range(0, len(src), chunk):
        sl = slice(start, start + chunk)
        ai = alpha[src[sl]]
        aj = alpha[tgt[sl]]
        lv = ((ai @ eff_left) * (aj @ eff_right)).sum(dim=-1)
        resid = (e[src[sl]] * e[tgt[sl]]).sum(dim=-1)
        logits = lv + bias + resid
        edge_logit = pair_edge_logit(likelihood, logits, log_r)
        lls.append(mean_bernoulli_ll(edge_logit, y[sl]))
        edge_logits.append(edge_logit)
        if y_count is not None:
            native.append(float(pair_log_lik(likelihood, logits, y_count[sl], log_r).mean().item()))
    bern_ll = float(np.mean(lls))
    auc = mean_auc(torch.cat(edge_logits), y)
    return bern_ll, float(np.mean(native)) if native else bern_ll, auc


def train(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    dtype = torch.float32
    torch.manual_seed(args.seed)

    adj_counts, adj = load_adjacency(args.adjacency, args.likelihood)
    n = adj.shape[0]
    nnz = int(adj.nnz)
    base_rate = max(float(adj.mean()), 1e-8)
    label_smoothing = resolve_label_smoothing(args.label_smoothing, args.likelihood)
    print(f"Loaded A with shape {adj.shape}, nnz={nnz}, device={device}")
    print(
        f"LV+e: d={args.d} d_e={args.d_e} e_wd={args.e_wd} lr={args.lr} "
        f"likelihood={args.likelihood} bfs_frac={args.bfs_frac} bfs_seeds={args.bfs_seeds}"
    )
    print(label_smoothing_line(label_smoothing, base_rate, args.label_smoothing_target))

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
    bias = nn.Parameter(torch.log(torch.tensor([base_rate], dtype=dtype, device=device)))
    q_logits = nn.Parameter((1.0 / args.k) * torch.randn(n, args.k, dtype=dtype, device=device))
    # Small init so residuals start near inactive.
    e = nn.Parameter(
        (0.01 / np.sqrt(args.d_e)) * torch.randn(n, args.d_e, dtype=dtype, device=device)
    )
    log_r = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
    scale = make_block_scale(args.u_norm, args.u_scale, args.u_scale_init, args.k, dtype, device)
    print(u_norm_line(args.u_norm, scale, float(bias.item())))

    dense_params = [u_left, u_right, bias, q_logits]
    if args.likelihood == "nb":
        dense_params.append(log_r)
    dense_params += [] if scale is None else scale.parameters
    optimizer = torch.optim.Adam(
        [
            {"params": dense_params, "weight_decay": 0.0},
            {"params": [e], "weight_decay": args.e_wd},
        ],
        lr=args.lr,
    )
    clip_params = [*dense_params, e]
    grad_clip = resolve_grad_clip(args.grad_clip)
    state = {
        "u_left": u_left,
        "u_right": u_right,
        "bias": bias,
        "q_logits": q_logits,
        "e": e,
        "log_r": log_r,
    }
    if scale is not None and scale.log_scale is not None:
        state["log_scale"] = scale.log_scale
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
    max_abs_eta = 0.0
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
            epoch_max_abs_eta = 0.0

            for idx in tqdm(sampler.epoch(), desc=f"lv_e epoch {epoch}"):
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
                eff_left, eff_right = scaled_block_embeddings(u_left, u_right, scale)
                lv = (q @ eff_left) @ (q @ eff_right).T
                resid = e[idx_t] @ e[idx_t].T
                logits = lv + bias + resid
                epoch_max_abs_eta = max(
                    epoch_max_abs_eta, float(logits.detach().abs().max().item())
                )

                # Training targets only; the held-out labels are never smoothed.
                target = smooth_binary_targets(
                    block, label_smoothing, base_rate, args.label_smoothing_target
                )
                pair_ll = pair_log_lik(args.likelihood, logits, target, log_r)
                ll = ((pair_ll * keep).sum() / n_keep.clamp(min=1.0)) * float(idx.size * idx.size)
                entropy = -(q * log_clamp(q)).sum(1).mean()
                loss = -(ll + entropy)

                if not safe_optimizer_step(loss, clip_params, optimizer, grad_clip):
                    n_skipped += 1
                    continue
                loss_sum += float(loss.item())
                n_scored += 1

            mean_loss = loss_sum / max(n_scored, 1)
            cum_pos_obs += epoch_pos_obs
            max_abs_eta = max(max_abs_eta, epoch_max_abs_eta)
            last_epoch = epoch
            print(sampler.coverage_line(epoch, n_batches, epoch_pos_obs, cum_pos_obs, nnz))
            val_ll, _, val_auc = eval_heldout(
                q_logits,
                u_left,
                u_right,
                bias,
                e,
                log_r,
                held,
                None,
                args.likelihood,
                device,
                dtype,
                scale=scale,
            )
            checkpoint.update(val_ll, epoch, state)
            with torch.no_grad():
                n_used = int(torch.unique(torch.argmax(q_logits, dim=-1)).numel())
                e_rms = float(torch.sqrt(torch.mean(e * e)).item())
            print(decoder_line(epoch, float(bias.item()), scale))
            print(
                f"loss={mean_loss:.4f} val_ll={val_ll:.6f} val_auc={val_auc:.4f} "
                f"best_val_ll={checkpoint.metric:.6f}@{checkpoint.epoch} "
                f"clusters={n_used}/{args.k} max_abs_logit={epoch_max_abs_eta:.3f} "
                f"e_rms={e_rms:.5f} updates={total_updates} skipped={n_skipped}"
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
                e,
                log_r,
                held,
                held_counts,
                args.likelihood,
                device,
                dtype,
                scale=scale,
            )
            e_rms = float(torch.sqrt(torch.mean(e * e)).item())
        pred = {mapping[i]: int(assignments[i]) for i in range(len(assignments))}
        return Snapshot(
            assignments,
            scores,
            pred,
            val_ll,
            native_ll,
            val_auc,
            e_rms,
            score_assignment_dict(pred, gt_path),
        )

    # Evaluate the final parameters before restoring, so the cost of any mid-run
    # divergence is measurable rather than inferred.
    last = snapshot()
    last_scale = block_scale_metrics(scale, "last_")
    last_bias = float(bias.item())
    restored = checkpoint.epoch != last_epoch and checkpoint.restore(state)
    best = snapshot() if restored else last

    prefix = Path(args.out_prefix)
    np.save(f"{prefix}_assignments.npy", best.assignments)
    np.save(f"{prefix}_scores.npy", best.scores)
    np.save(f"{prefix}_e.npy", e.detach().cpu().numpy())
    np.save(f"{prefix}_assignment_dict.npy", best.assignment_dict)
    if best is not last:
        np.save(f"{prefix}_last_assignment_dict.npy", last.assignment_dict)
    metrics = {
        "method": "lv_e",
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
        "d_e": args.d_e,
        "e_wd": args.e_wd,
        "e_rms": best.e_rms,
        "last_e_rms": last.e_rms,
        "lr": args.lr,
        "epochs": args.epochs,
        "likelihood": args.likelihood,
        "bfs_frac": args.bfs_frac,
        "bfs_seeds": args.bfs_seeds,
        "grad_clip": grad_clip,
        "label_smoothing": args.label_smoothing,
        "label_smoothing_target": args.label_smoothing_target,
        "label_smoothing_applied": label_smoothing > 0.0,
        "u_norm": args.u_norm,
        "u_scale": args.u_scale,
        "u_scale_init": args.u_scale_init,
        **block_scale_metrics(scale),
        **last_scale,
        "decoder_bias": float(bias.item()),
        "last_decoder_bias": last_bias,
        "train_base_rate": base_rate,
        "max_abs_train_logit": max_abs_eta,
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
        f"native_ll={best.native_ll:.6f} val_auc={best.val_auc:.4f} e_rms={best.e_rms:.5f} "
        f"| last val_ll={last.val_ll:.6f} val_auc={last.val_auc:.4f}"
    )
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adjacency", default="sparse_connectivity_matrix.npz")
    p.add_argument("--mapping", default="root_id_to_index_mapping.json")
    p.add_argument("--heldout-pairs", default="heldout_pairs.npz")
    p.add_argument("--out-prefix", default="lv_e")
    p.add_argument("--k", type=int, default=729)
    p.add_argument("--d", type=int, default=32, help="Block embedding dim")
    p.add_argument("--d-e", type=int, default=16, help="Node residual dim")
    p.add_argument("--e-wd", type=float, default=1e-2, help="Weight decay on e only")
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
        "--label-smoothing",
        type=float,
        default=0.0,
        help=(
            "Bernoulli target smoothing eps in [0, 1); 0 (default) trains on hard "
            "0/1 targets, whose optimal logit is unbounded. Applied to the training "
            "loss only, and ignored for the count likelihoods"
        ),
    )
    p.add_argument(
        "--label-smoothing-target",
        choices=LABEL_SMOOTHING_TARGETS,
        default="base_rate",
        help=(
            "Prior the targets are pulled towards: 'base_rate' preserves the edge "
            "marginal; 'uniform' (0.5) inflates negatives far above the true density"
        ),
    )
    add_u_norm_arguments(p)
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
