"""GNN-augmented variational stochastic block model (GNN-vSBM).

Extends the low-rank vSBM in ``hidden_markov_graph.py``. Soft assignments
``alpha = softmax(beta)`` feed a low-rank bilinear edge model, and a directed
GNN supplies a *residual* multi-hop correction:

    logits_ij = (alpha_i U_s)·(alpha_j U_t) + b
                + gamma * (h_i^{src} · h_j^{tgt})

Hop features are combined with Jumping Knowledge (concat of
``h^{(0)},…,h^{(L)}`` then a linear map) so deeper layers expand the
receptive field without erasing 0-hop cluster identity. On this connectome,
undirected 2-hop balls already cover ~3k nodes (median) and 3-hop ~35k, so
oversmoothing—not insufficient neighborhood size—is the main risk.
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

from heldout import (
    heldout_pair_mask,
    load_heldout,
    mean_bernoulli_ll,
    remove_heldout_positives,
    write_metrics,
)
from index_mapping import load_mapping

# Empirically measured undirected hop-ball sizes (excl. self) on FlyWire A.
HOP_BALL_NOTE = (
    "Receptive field (undirected hop ball, median excl. self): "
    "L=1 ~18, L=2 ~3k, L=3 ~37k, L=4 ~105k nodes."
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def log_clamp(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def csr_to_torch_sparse(mat: csr_matrix, device: str, dtype: torch.dtype) -> torch.Tensor:
    coo = mat.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=dtype, device=device)
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device).coalesce()


def row_normalize_sparse(adj: torch.Tensor) -> torch.Tensor:
    """Row-normalize a coalesced sparse adjacency (out-degree normalization)."""
    indices = adj.indices()
    values = adj.values()
    row = indices[0]
    deg = torch.zeros(adj.size(0), dtype=values.dtype, device=values.device)
    deg.index_add_(0, row, values)
    inv_deg = 1.0 / deg.clamp(min=1.0)
    norm_values = values * inv_deg[row]
    return torch.sparse_coo_tensor(
        indices, norm_values, size=adj.size(), device=adj.device
    ).coalesce()


def mask_minibatch_edges(adj: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """Zero directed edges whose both endpoints lie in ``batch_idx``.

    This prevents the GNN from reading the minibatch edges that the decoder is
    asked to reconstruct. Empirically this removes ≪1% of edges for typical
    minibatch sizes, so multi-hop neighborhoods remain intact.
    """
    indices = adj.indices()
    values = adj.values()
    n = adj.size(0)
    in_batch = torch.zeros(n, dtype=torch.bool, device=adj.device)
    in_batch[batch_idx] = True
    keep = ~(in_batch[indices[0]] & in_batch[indices[1]])
    if bool(keep.all()):
        return adj
    return torch.sparse_coo_tensor(
        indices[:, keep],
        values[keep],
        size=adj.size(),
        device=adj.device,
    ).coalesce()


class DirectedGNNLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w_self = nn.Linear(d, d, bias=False)
        self.w_in = nn.Linear(d, d, bias=False)
        self.w_out = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.act = nn.GELU()

    def forward(
        self,
        h: torch.Tensor,
        a_out: torch.Tensor,
        a_in: torch.Tensor,
    ) -> torch.Tensor:
        msg_out = torch.sparse.mm(a_out, h)
        msg_in = torch.sparse.mm(a_in, h)
        update = self.w_self(h) + self.w_out(msg_out) + self.w_in(msg_in)
        return self.act(self.norm(h + update))


class GNNvSBM(nn.Module):
    """Mean-field vSBM with residual multi-hop GNN correction + JK."""

    def __init__(
        self,
        n: int,
        k: int,
        d: int,
        n_layers: int,
        edge_bias_init: float,
        dtype: torch.dtype,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        self.n = n
        self.k = k
        self.d = d
        self.n_layers = n_layers
        self.q_logits = nn.Parameter((1.0 / k) * torch.randn(n, k, dtype=dtype))
        # Shared projection for GNN node features (0-hop).
        self.u = nn.Parameter((1.0 / np.sqrt(k * d)) * torch.randn(k, d, dtype=dtype))
        # Low-rank LV prototypes (bilinear decoder backbone).
        self.u_src = nn.Parameter((1.0 / np.sqrt(k * d)) * torch.randn(k, d, dtype=dtype))
        self.u_tgt = nn.Parameter((1.0 / np.sqrt(k * d)) * torch.randn(k, d, dtype=dtype))
        self.layers = nn.ModuleList([DirectedGNNLayer(d) for _ in range(n_layers)])
        # Jumping Knowledge: concat h0..hL then project back to d.
        self.jk_proj = nn.Linear(d * (n_layers + 1), d, bias=False)
        self.src_head = nn.Linear(d, d, bias=False)
        self.tgt_head = nn.Linear(d, d, bias=False)
        self.bias = nn.Parameter(torch.tensor([edge_bias_init], dtype=dtype))
        # Residual mix; init near 0 so training can fall back to pure LV.
        self.gamma = nn.Parameter(torch.tensor([gamma_init], dtype=dtype))

    def soft_assignments(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.q_logits if idx is None else self.q_logits[idx]
        return torch.softmax(logits, dim=-1)

    def encode(
        self,
        a_out: torch.Tensor,
        a_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.soft_assignments()
        h = alpha @ self.u
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
        # LV bilinear: (α_i U_s) (α_j U_t)^T
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
        """Per-pair logits for held-out evaluation (not a dense block)."""
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


def train(args: argparse.Namespace) -> dict:
    device = args.device or pick_device()
    dtype = torch.float32
    torch.manual_seed(args.seed)

    adj_csr = load_binary_adjacency(Path(args.adjacency))
    n = adj_csr.shape[0]
    print(f"Loaded A with shape {adj_csr.shape}, nnz={adj_csr.nnz}, device={device}")
    print(HOP_BALL_NOTE)
    print(
        f"GNN layers L={args.layers} (JK over hops 0..{args.layers}), gamma_init={args.gamma_init}"
    )

    held = load_heldout(Path(args.heldout_pairs))
    held_src_np = held["src"]
    held_tgt_np = held["tgt"]
    held_y = torch.tensor(held["y"], dtype=dtype, device=device)
    held_src = torch.tensor(held_src_np, dtype=torch.long, device=device)
    held_tgt = torch.tensor(held_tgt_np, dtype=torch.long, device=device)

    # Message passing graph: remove held-out positive edges to avoid leakage.
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
    model = GNNvSBM(
        n=n,
        k=args.k,
        d=args.d,
        n_layers=args.layers,
        edge_bias_init=edge_bias_init,
        dtype=dtype,
        gamma_init=args.gamma_init,
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    minibatch = args.minibatch
    n_minibatches = max(n // minibatch, 1)
    best_val = -float("inf")

    try:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch}")
            indices_i = torch.randperm(n, device=device)
            indices_j = torch.randperm(n, device=device)
            initial_loss = 0.0
            final_loss = 0.0
            n_steps = 0

            for step in tqdm(range(n_minibatches), desc=f"epoch {epoch}"):
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

                if step == 0:
                    initial_loss = float(loss.item())
                final_loss = float(loss.item())
                n_steps += 1

            val_ll = eval_heldout_ll_gnn(model, a_out_full, a_in_full, held_src, held_tgt, held_y)
            best_val = max(best_val, val_ll)
            with torch.no_grad():
                n_used = int(torch.unique(torch.argmax(model.q_logits, dim=-1)).numel())
                gamma = float(model.gamma.item())
            print(
                f"Initial loss: {initial_loss}, Final loss: {final_loss} "
                f"(steps={n_steps}) val_ll={val_ll:.6f} best_val_ll={best_val:.6f} "
                f"gamma={gamma:.4f} clusters={n_used}/{args.k}"
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


@torch.no_grad()
def eval_heldout_ll_gnn(
    model: GNNvSBM,
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


def save_results(
    model: GNNvSBM,
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
        u = model.u.detach().cpu().numpy()
        val_ll = eval_heldout_ll_gnn(model, a_out, a_in, held_src, held_tgt, held_y)
        gamma = float(model.gamma.item())

    mapping = load_mapping(str(mapping_path))
    cluster_assignment_dict = {mapping[i]: int(assignments[i]) for i in range(len(assignments))}

    np.save(f"{out_prefix}_assignments.npy", assignments)
    np.save(f"{out_prefix}_scores.npy", scores)
    np.save(f"{out_prefix}_U.npy", u)
    np.save(f"{out_prefix}_assignment_dict.npy", cluster_assignment_dict)
    metrics = {
        "method": "gnn_vsbm",
        "val_metric": val_ll,
        "val_metric_name": "heldout_bernoulli_ll",
        "val_metric_higher_is_better": True,
        "seed": args.seed,
        "k": args.k,
        "d": args.d,
        "layers": args.layers,
        "lr": args.lr,
        "gamma": gamma,
        "gamma_init": args.gamma_init,
        "entropy_weight": args.entropy_weight,
        "epochs": args.epochs,
        "n_pred_clusters": int(len(np.unique(assignments))),
    }
    write_metrics(f"{out_prefix}_metrics.json", metrics)
    print(f"Saved {out_prefix}_* ; val_ll={val_ll:.6f} gamma={gamma:.4f}")
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adjacency", default="sparse_connectivity_matrix.npz")
    p.add_argument("--mapping", default="root_id_to_index_mapping.json")
    p.add_argument("--heldout-pairs", default="heldout_pairs.npz")
    p.add_argument("--out-prefix", default="gnn")
    p.add_argument("--k", type=int, default=729)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--layers", type=int, default=2, help="GNN depth L; 0 = LV + JK(h0) only")
    p.add_argument("--minibatch", type=int, default=2500)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-updates", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--gamma-init", type=float, default=0.0, help="Init for residual GNN mix")
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
