"""Low-rank vSBM with node residuals e_i (no GNN).

Edge logits:

    logit_ij = (alpha_i U_s) · (alpha_j U_t) + b + e_i · e_j

``e_i`` captures within-type / degree-like residual connectivity. Weight decay
is applied **only** to ``e`` (Adam param group) so residuals stay small and do
not absorb cluster identity. Clusters are still read from ``argmax(alpha)``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import load_npz
from torch import nn
from tqdm import tqdm

from heldout import (
    heldout_pair_mask,
    load_heldout,
    mean_bernoulli_ll,
    write_metrics,
)
from index_mapping import load_mapping


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


@torch.no_grad()
def eval_heldout_ll(
    q_logits: torch.Tensor,
    u_left: torch.Tensor,
    u_right: torch.Tensor,
    bias: torch.Tensor,
    e: torch.Tensor,
    held: dict[str, np.ndarray],
    device: str,
    dtype: torch.dtype,
    chunk: int = 8192,
) -> float:
    alpha = torch.softmax(q_logits, dim=-1)
    src = held["src"]
    tgt = held["tgt"]
    y = torch.tensor(held["y"], dtype=dtype, device=device)
    lls = []
    for start in range(0, len(src), chunk):
        sl = slice(start, start + chunk)
        ai = alpha[src[sl]]
        aj = alpha[tgt[sl]]
        lv = ((ai @ u_left) * (aj @ u_right)).sum(dim=-1)
        resid = (e[src[sl]] * e[tgt[sl]]).sum(dim=-1)
        logits = lv + bias + resid
        lls.append(mean_bernoulli_ll(logits, y[sl]))
    return float(np.mean(lls))


def train(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    dtype = torch.float32
    torch.manual_seed(args.seed)

    adj = load_npz(args.adjacency)
    adj.data = (adj.data > 0).astype(np.float32)
    adj.eliminate_zeros()
    n = adj.shape[0]
    print(f"Loaded A with shape {adj.shape}, nnz={adj.nnz}, device={device}")
    print(f"LV+e: d={args.d} d_e={args.d_e} e_wd={args.e_wd} lr={args.lr}")

    held = load_heldout(Path(args.heldout_pairs))
    held_src_np = held["src"]
    held_tgt_np = held["tgt"]

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
    # Small init so residuals start near inactive.
    e = nn.Parameter(
        (0.01 / np.sqrt(args.d_e)) * torch.randn(n, args.d_e, dtype=dtype, device=device)
    )

    optimizer = torch.optim.Adam(
        [
            {"params": [u_left, u_right, bias, q_logits], "weight_decay": 0.0},
            {"params": [e], "weight_decay": args.e_wd},
        ],
        lr=args.lr,
    )
    n_minibatches = max(n // args.minibatch, 1)
    best_val = -float("inf")

    try:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch}")
            indices_i = torch.randperm(n, device=device)
            indices_j = torch.randperm(n, device=device)
            final_loss = 0.0

            for step in tqdm(range(n_minibatches), desc=f"lv_e epoch {epoch}"):
                if args.max_updates is not None and step >= args.max_updates:
                    break
                idx_i = indices_i[step * args.minibatch : (step + 1) * args.minibatch]
                idx_j = indices_j[step * args.minibatch : (step + 1) * args.minibatch]
                if idx_i.numel() == 0 or idx_j.numel() == 0:
                    continue

                q_i = torch.softmax(q_logits[idx_i], dim=-1)
                q_j = torch.softmax(q_logits[idx_j], dim=-1)
                cc = torch.tensor(
                    adj[idx_i.cpu().numpy()][:, idx_j.cpu().numpy()].toarray(),
                    dtype=dtype,
                    device=device,
                )
                mask = torch.tensor(
                    heldout_pair_mask(
                        idx_i.cpu().numpy(),
                        idx_j.cpu().numpy(),
                        held_src_np,
                        held_tgt_np,
                    ),
                    device=device,
                )
                keep = (~mask).to(dtype)
                if not bool((keep > 0).any()):
                    continue

                lv = (q_i @ u_left) @ (q_j @ u_right).T
                resid = e[idx_i] @ e[idx_j].T
                logits = lv + bias + resid

                bce = (
                    F.binary_cross_entropy_with_logits(logits, cc, reduction="none") * keep
                ).sum() / keep.sum().clamp(min=1.0)
                # Scale like other trainers; entropy on soft assignments.
                ll = -bce * float(idx_i.numel() * idx_j.numel())
                entropy = -(q_i * log_clamp(q_i)).sum(1).mean() / 2.0
                entropy = entropy - (q_j * log_clamp(q_j)).sum(1).mean() / 2.0
                loss = -(ll + entropy)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([u_left, u_right, bias, q_logits, e], 1.0)
                optimizer.step()
                final_loss = float(loss.item())

            val_ll = eval_heldout_ll(
                q_logits, u_left, u_right, bias, e, held, device, dtype
            )
            best_val = max(best_val, val_ll)
            with torch.no_grad():
                n_used = int(torch.unique(torch.argmax(q_logits, dim=-1)).numel())
                e_rms = float(torch.sqrt(torch.mean(e * e)).item())
            print(
                f"loss={final_loss:.4f} val_ll={val_ll:.6f} "
                f"best_val_ll={best_val:.6f} clusters={n_used}/{args.k} "
                f"e_rms={e_rms:.5f}"
            )
    except KeyboardInterrupt:
        print("Training interrupted.")

    with torch.no_grad():
        assignments = torch.argmax(q_logits, dim=-1).cpu().numpy()
        scores = torch.max(torch.softmax(q_logits, dim=-1), dim=-1).values.cpu().numpy()
        val_ll = eval_heldout_ll(q_logits, u_left, u_right, bias, e, held, device, dtype)
        e_rms = float(torch.sqrt(torch.mean(e * e)).item())

    mapping = load_mapping(args.mapping)
    cluster_assignment_dict = {
        mapping[i]: int(assignments[i]) for i in range(len(assignments))
    }
    prefix = Path(args.out_prefix)
    np.save(f"{prefix}_assignments.npy", assignments)
    np.save(f"{prefix}_scores.npy", scores)
    np.save(f"{prefix}_e.npy", e.detach().cpu().numpy())
    np.save(f"{prefix}_assignment_dict.npy", cluster_assignment_dict)
    metrics = {
        "method": "lv_e",
        "val_metric": val_ll,
        "val_metric_name": "heldout_bernoulli_ll",
        "val_metric_higher_is_better": True,
        "seed": args.seed,
        "k": args.k,
        "d": args.d,
        "d_e": args.d_e,
        "e_wd": args.e_wd,
        "e_rms": e_rms,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_pred_clusters": int(len(np.unique(assignments))),
    }
    write_metrics(f"{prefix}_metrics.json", metrics)
    print(f"Saved {prefix}_* ; val_ll={val_ll:.6f} e_rms={e_rms:.5f}")
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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-updates", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-1)
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
