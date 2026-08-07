"""LV + residual GNN over node embeddings e_i (GNN_e).

Edge logits:

    logit_ij = (alpha_i U_s)·(alpha_j U_t) + b
               + gamma * (h_i^{src} · h_j^{tgt})

where ``h^{(0)} = e_i`` (not cluster features). Directed GNN layers + Jumping
Knowledge refine ``e``; clusters remain ``argmax(alpha)``. Adam applies weight
decay **only** to raw ``e`` so residuals stay small and do not absorb type
identity. ``gamma`` is initialized at 0 so training can fall back to pure LV.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, load_npz
from torch import nn
from tqdm import tqdm

from gnn_vsbm import (
    HOP_BALL_NOTE,
    DirectedGNNLayer,
    csr_to_torch_sparse,
    mask_minibatch_edges,
    row_normalize_sparse,
)
from heldout import (
    heldout_pair_mask,
    load_heldout,
    mean_bernoulli_ll,
    remove_heldout_positives,
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


class GNNeVSBM(nn.Module):
    """LV bilinear backbone + GNN residual over e_i."""

    def __init__(
        self,
        n: int,
        k: int,
        d: int,
        d_e: int,
        n_layers: int,
        edge_bias_init: float,
        dtype: torch.dtype,
        gamma_init: float = 0.0,
        e_init_scale: float = 0.01,
    ):
        super().__init__()
        self.n = n
        self.k = k
        self.d = d
        self.d_e = d_e
        self.n_layers = n_layers
        self.q_logits = nn.Parameter((1.0 / k) * torch.randn(n, k, dtype=dtype))
        self.u_src = nn.Parameter((1.0 / np.sqrt(k * d)) * torch.randn(k, d, dtype=dtype))
        self.u_tgt = nn.Parameter((1.0 / np.sqrt(k * d)) * torch.randn(k, d, dtype=dtype))
        # Node residuals: GNN input features.
        self.e = nn.Parameter((e_init_scale / np.sqrt(d_e)) * torch.randn(n, d_e, dtype=dtype))
        self.layers = nn.ModuleList([DirectedGNNLayer(d_e) for _ in range(n_layers)])
        self.jk_proj = nn.Linear(d_e * (n_layers + 1), d_e, bias=False)
        self.src_head = nn.Linear(d_e, d_e, bias=False)
        self.tgt_head = nn.Linear(d_e, d_e, bias=False)
        self.bias = nn.Parameter(torch.tensor([edge_bias_init], dtype=dtype))
        self.gamma = nn.Parameter(torch.tensor([gamma_init], dtype=dtype))

    def soft_assignments(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.q_logits if idx is None else self.q_logits[idx]
        return torch.softmax(logits, dim=-1)

    def encode(
        self,
        a_out: torch.Tensor,
        a_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.e
        hops = [h]
        for layer in self.layers:
            h = layer(h, a_out, a_in)
            hops.append(h)
        h_jk = self.jk_proj(torch.cat(hops, dim=-1))
        return self.src_head(h_jk), self.tgt_head(h_jk)

    def edge_logits(
        self,
        h_src: torch.Tensor,
        h_tgt: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        alpha_i = self.soft_assignments(idx_i)
        alpha_j = self.soft_assignments(idx_j)
        lv = (alpha_i @ self.u_src) @ (alpha_j @ self.u_tgt).T
        gnn = h_src[idx_i] @ h_tgt[idx_j].T
        return lv + self.bias + self.gamma * gnn

    def pair_logits(
        self,
        h_src: torch.Tensor,
        h_tgt: torch.Tensor,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        alpha_s = self.soft_assignments(src)
        alpha_t = self.soft_assignments(tgt)
        lv = ((alpha_s @ self.u_src) * (alpha_t @ self.u_tgt)).sum(dim=-1)
        gnn = (h_src[src] * h_tgt[tgt]).sum(dim=-1)
        return lv + self.bias + self.gamma * gnn


def load_binary_adjacency(path: Path) -> csr_matrix:
    adj = load_npz(path)
    adj.data = (adj.data > 0).astype(np.float32)
    adj.eliminate_zeros()
    return adj.tocsr()


@torch.no_grad()
def eval_heldout_ll(
    model: GNNeVSBM,
    a_out: torch.Tensor,
    a_in: torch.Tensor,
    held_src: torch.Tensor,
    held_tgt: torch.Tensor,
    held_y: torch.Tensor,
    chunk: int = 8192,
) -> float:
    h_src, h_tgt = model.encode(a_out, a_in)
    lls = []
    for start in range(0, held_src.numel(), chunk):
        sl = slice(start, start + chunk)
        logits = model.pair_logits(h_src, h_tgt, held_src[sl], held_tgt[sl])
        lls.append(mean_bernoulli_ll(logits, held_y[sl]))
    return float(np.mean(lls))


def train(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    dtype = torch.float32
    torch.manual_seed(args.seed)

    adj_csr = load_binary_adjacency(Path(args.adjacency))
    n = adj_csr.shape[0]
    print(f"Loaded A with shape {adj_csr.shape}, nnz={adj_csr.nnz}, device={device}")
    print(HOP_BALL_NOTE)
    print(
        f"GNN_e: L={args.layers} d={args.d} d_e={args.d_e} e_wd={args.e_wd} "
        f"lr={args.lr} gamma_init={args.gamma_init}"
    )

    held = load_heldout(Path(args.heldout_pairs))
    held_src_np = held["src"]
    held_tgt_np = held["tgt"]
    held_y = torch.tensor(held["y"], dtype=dtype, device=device)
    held_src = torch.tensor(held_src_np, dtype=torch.long, device=device)
    held_tgt = torch.tensor(held_tgt_np, dtype=torch.long, device=device)

    adj_mp = remove_heldout_positives(adj_csr, held_src_np, held_tgt_np, held["y"])
    a_bin = csr_to_torch_sparse(adj_mp, device=device, dtype=dtype)
    a_bin_t = torch.sparse_coo_tensor(
        a_bin.indices().flip(0),
        a_bin.values(),
        size=a_bin.size(),
        device=device,
    ).coalesce()
    a_out_full = row_normalize_sparse(a_bin)
    a_in_full = row_normalize_sparse(a_bin_t)

    edge_bias_init = float(np.log(max(float(adj_csr.mean()), 1e-8)))
    model = GNNeVSBM(
        n=n,
        k=args.k,
        d=args.d,
        d_e=args.d_e,
        n_layers=args.layers,
        edge_bias_init=edge_bias_init,
        dtype=dtype,
        gamma_init=args.gamma_init,
    ).to(device=device, dtype=dtype)

    # Weight decay only on raw e; GNN / LV params unconstrained.
    e_params = [model.e]
    other_params = [p for n, p in model.named_parameters() if n != "e"]
    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "weight_decay": 0.0},
            {"params": e_params, "weight_decay": args.e_wd},
        ],
        lr=args.lr,
    )

    minibatch = args.minibatch
    n_minibatches = max(n // minibatch, 1)
    best_val = -float("inf")

    try:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch}")
            indices_i = torch.randperm(n, device=device)
            indices_j = torch.randperm(n, device=device)
            final_loss = 0.0
            n_steps = 0

            for step in tqdm(range(n_minibatches), desc=f"gnn_e epoch {epoch}"):
                if args.max_updates is not None and step >= args.max_updates:
                    print(f"Reached max updates {args.max_updates}.")
                    break

                idx_i = indices_i[step * minibatch : (step + 1) * minibatch]
                idx_j = indices_j[step * minibatch : (step + 1) * minibatch]
                if idx_i.numel() == 0 or idx_j.numel() == 0:
                    continue

                batch_union = torch.unique(torch.cat([idx_i, idx_j], dim=0))
                if args.no_edge_mask or args.layers == 0:
                    a_out, a_in = a_out_full, a_in_full
                else:
                    a_out = row_normalize_sparse(mask_minibatch_edges(a_bin, batch_union))
                    a_in = row_normalize_sparse(mask_minibatch_edges(a_bin_t, batch_union))

                h_src, h_tgt = model.encode(a_out, a_in)
                logits = model.edge_logits(h_src, h_tgt, idx_i, idx_j)

                cc = torch.tensor(
                    adj_csr[idx_i.cpu().numpy()][:, idx_j.cpu().numpy()].toarray(),
                    dtype=dtype,
                    device=device,
                )
                held_mask = torch.tensor(
                    heldout_pair_mask(
                        idx_i.cpu().numpy(),
                        idx_j.cpu().numpy(),
                        held_src_np,
                        held_tgt_np,
                    ),
                    device=device,
                )
                keep = (~held_mask).to(dtype)
                denom = keep.sum().clamp(min=1.0)
                bce = (
                    F.binary_cross_entropy_with_logits(logits, cc, reduction="none") * keep
                ).sum() / denom
                ll = -bce * float(idx_i.numel() * idx_j.numel())

                alpha_i = model.soft_assignments(idx_i)
                alpha_j = model.soft_assignments(idx_j)
                entropy = -(alpha_i * log_clamp(alpha_i)).sum(1).mean()
                entropy = entropy - (alpha_j * log_clamp(alpha_j)).sum(1).mean()
                entropy = args.entropy_weight * entropy / 2.0

                loss = -(ll + entropy)
                if not torch.isfinite(loss):
                    print(f"Non-finite loss at epoch {epoch} step {step}; skipping.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                final_loss = float(loss.item())
                n_steps += 1

            val_ll = eval_heldout_ll(model, a_out_full, a_in_full, held_src, held_tgt, held_y)
            best_val = max(best_val, val_ll)
            with torch.no_grad():
                n_used = int(torch.unique(torch.argmax(model.q_logits, dim=-1)).numel())
                gamma = float(model.gamma.item())
                e_rms = float(torch.sqrt(torch.mean(model.e * model.e)).item())
            print(
                f"loss={final_loss:.4f} (steps={n_steps}) val_ll={val_ll:.6f} "
                f"best_val_ll={best_val:.6f} gamma={gamma:.4f} "
                f"e_rms={e_rms:.5f} clusters={n_used}/{args.k}"
            )
    except KeyboardInterrupt:
        print("Training interrupted.")

    return save_results(
        model,
        Path(args.mapping),
        Path(args.out_prefix),
        a_out_full,
        a_in_full,
        held_src,
        held_tgt,
        held_y,
        args,
    )


def save_results(
    model: GNNeVSBM,
    mapping_path: Path,
    out_prefix: Path,
    a_out: torch.Tensor,
    a_in: torch.Tensor,
    held_src: torch.Tensor,
    held_tgt: torch.Tensor,
    held_y: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    with torch.no_grad():
        assignments = torch.argmax(model.q_logits, dim=-1).cpu().numpy()
        scores = torch.max(torch.softmax(model.q_logits, dim=-1), dim=-1).values.cpu().numpy()
        e = model.e.detach().cpu().numpy()
        val_ll = eval_heldout_ll(model, a_out, a_in, held_src, held_tgt, held_y)
        gamma = float(model.gamma.item())
        e_rms = float(np.sqrt(np.mean(e * e)))

    mapping = load_mapping(str(mapping_path))
    cluster_assignment_dict = {mapping[i]: int(assignments[i]) for i in range(len(assignments))}

    np.save(f"{out_prefix}_assignments.npy", assignments)
    np.save(f"{out_prefix}_scores.npy", scores)
    np.save(f"{out_prefix}_e.npy", e)
    np.save(f"{out_prefix}_assignment_dict.npy", cluster_assignment_dict)
    metrics = {
        "method": "gnn_e",
        "val_metric": val_ll,
        "val_metric_name": "heldout_bernoulli_ll",
        "val_metric_higher_is_better": True,
        "seed": args.seed,
        "k": args.k,
        "d": args.d,
        "d_e": args.d_e,
        "e_wd": args.e_wd,
        "e_rms": e_rms,
        "layers": args.layers,
        "lr": args.lr,
        "gamma": gamma,
        "gamma_init": args.gamma_init,
        "entropy_weight": args.entropy_weight,
        "epochs": args.epochs,
        "n_pred_clusters": int(len(np.unique(assignments))),
    }
    write_metrics(f"{out_prefix}_metrics.json", metrics)
    print(f"Saved {out_prefix}_* ; val_ll={val_ll:.6f} gamma={gamma:.4f} e_rms={e_rms:.5f}")
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adjacency", default="sparse_connectivity_matrix.npz")
    p.add_argument("--mapping", default="root_id_to_index_mapping.json")
    p.add_argument("--heldout-pairs", default="heldout_pairs.npz")
    p.add_argument("--out-prefix", default="gnn_e")
    p.add_argument("--k", type=int, default=729)
    p.add_argument("--d", type=int, default=32, help="LV block embedding dim")
    p.add_argument("--d-e", type=int, default=16, help="Node residual / GNN feature dim")
    p.add_argument("--e-wd", type=float, default=1e-2, help="Weight decay on e only")
    p.add_argument("--layers", type=int, default=2, help="GNN depth L; 0 = JK(e) only")
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-updates", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--gamma-init", type=float, default=0.0)
    p.add_argument("--entropy-weight", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-edge-mask", action="store_true")
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
