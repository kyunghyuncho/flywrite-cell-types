"""GNN-augmented variational stochastic block model (GNN-vSBM).

Extends the low-rank vSBM in ``hidden_markov_graph.py`` by replacing the
single-hop edge probability

    p(e_ij = 1 | z_i, z_j) = sigmoid(u_s^{z_i} · u_t^{z_j} + b)

with a directed graph neural net that conditions on multi-hop soft cluster
assignments. Soft assignments alpha = softmax(beta) are projected into a
d-dimensional embedding space, propagated over the (masked) observed graph,
and decoded into directed edge probabilities. All parameters are trained by
maximizing a minibatch estimate of the variational lower bound.
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

from index_mapping import load_mapping


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
    asked to reconstruct.
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
    """Mean-field variational SBM with a directed multi-hop GNN decoder."""

    def __init__(
        self,
        n: int,
        k: int,
        d: int,
        n_layers: int,
        edge_bias_init: float,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.n = n
        self.k = k
        self.d = d
        self.q_logits = nn.Parameter((1.0 / k) * torch.randn(n, k, dtype=dtype))
        self.u = nn.Parameter((1.0 / np.sqrt(k * d)) * torch.randn(k, d, dtype=dtype))
        self.layers = nn.ModuleList([DirectedGNNLayer(d) for _ in range(n_layers)])
        self.src_head = nn.Linear(d, d, bias=False)
        self.tgt_head = nn.Linear(d, d, bias=False)
        self.bias = nn.Parameter(torch.tensor([edge_bias_init], dtype=dtype))

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
        for layer in self.layers:
            h = layer(h, a_out, a_in)
        return self.src_head(h), self.tgt_head(h)

    def edge_logits(
        self,
        h_src: torch.Tensor,
        h_tgt: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        return (h_src[idx_i] @ h_tgt[idx_j].T) + self.bias


def load_binary_adjacency(path: Path) -> csr_matrix:
    adj = load_npz(path)
    adj.data = (adj.data > 0).astype(np.float32)
    adj.eliminate_zeros()
    return adj.tocsr()


def train(args: argparse.Namespace) -> None:
    device = args.device or pick_device()
    # Prefer float32 everywhere for sparse GNN stability (esp. on MPS).
    dtype = torch.float32
    torch.manual_seed(args.seed)

    adj_csr = load_binary_adjacency(Path(args.adjacency))
    n = adj_csr.shape[0]
    print(f"Loaded A with shape {adj_csr.shape}, nnz={adj_csr.nnz}, device={device}")

    a_bin = csr_to_torch_sparse(adj_csr, device=device, dtype=dtype)
    a_bin_t = torch.sparse_coo_tensor(
        a_bin.indices().flip(0),
        a_bin.values(),
        size=a_bin.size(),
        device=device,
    ).coalesce()
    # Pre-normalize once; minibatch masking is applied on the binary tensors.
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
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    minibatch = args.minibatch
    n_minibatches = max(n // minibatch, 1)

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
                if args.no_edge_mask:
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

                # Mean BCE-with-logits is numerically stable; scale by B^2 so the
                # magnitude remains comparable to a summed Bernoulli ELBO term.
                bce = F.binary_cross_entropy_with_logits(logits, cc, reduction="mean")
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

            print(f"Initial loss: {initial_loss}, Final loss: {final_loss} (steps={n_steps})")
            with torch.no_grad():
                sample = torch.argmax(model.q_logits[:10], dim=-1).cpu().tolist()
                n_used = int(torch.unique(torch.argmax(model.q_logits, dim=-1)).numel())
                print(f"Sample hard assignments (nodes 0-9): {sample}")
                print(f"Unique hard clusters in use: {n_used}/{args.k}")
    except KeyboardInterrupt:
        print("Training interrupted.")

    save_results(model, Path(args.mapping), Path(args.out_prefix))


def save_results(model: GNNvSBM, mapping_path: Path, out_prefix: Path) -> None:
    with torch.no_grad():
        assignments = torch.argmax(model.q_logits, dim=-1).cpu().numpy()
        scores = torch.max(torch.softmax(model.q_logits, dim=-1), dim=-1).values.cpu().numpy()
        u = model.u.detach().cpu().numpy()

    mapping = load_mapping(str(mapping_path))
    cluster_assignment_dict = {mapping[i]: int(assignments[i]) for i in range(len(assignments))}

    np.save(f"{out_prefix}_assignments.npy", assignments)
    np.save(f"{out_prefix}_scores.npy", scores)
    np.save(f"{out_prefix}_U.npy", u)
    np.save(f"{out_prefix}_assignment_dict_729.npy", cluster_assignment_dict)
    torch.save(
        {
            "q_logits": model.q_logits.detach().cpu(),
            "u": model.u.detach().cpu(),
            "bias": model.bias.detach().cpu(),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        f"{out_prefix}_checkpoint.pt",
    )
    print(
        "Saved "
        f"{out_prefix}_assignments.npy, "
        f"{out_prefix}_scores.npy, "
        f"{out_prefix}_U.npy, "
        f"{out_prefix}_assignment_dict_729.npy, "
        f"{out_prefix}_checkpoint.pt"
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adjacency", default="sparse_connectivity_matrix.npz")
    p.add_argument("--mapping", default="root_id_to_index_mapping.json")
    p.add_argument("--out-prefix", default="gnn")
    p.add_argument("--k", type=int, default=729)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=2500)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-updates", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-2)
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
