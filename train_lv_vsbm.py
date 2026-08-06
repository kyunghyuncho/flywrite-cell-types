"""Low-rank variational SBM baseline (single-hop), CLI-friendly.

This is the same model as ``hidden_markov_graph.py``:
    p(e_ij=1 | z_i, z_j) = sigmoid(u_s^{z_i} · u_t^{z_j} + b)
trained by minibatch maximization of the mean-field ELBO.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import load_npz
from torch import nn
from tqdm import tqdm

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


def train(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    dtype = torch.float32
    torch.manual_seed(args.seed)

    adj = load_npz(args.adjacency)
    adj.data = (adj.data > 0).astype(np.float32)
    adj.eliminate_zeros()
    n = adj.shape[0]
    print(f"Loaded A with shape {adj.shape}, nnz={adj.nnz}, device={device}")

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

    optimizer = torch.optim.Adam([u_left, u_right, bias, q_logits], lr=args.lr)
    n_minibatches = max(n // args.minibatch, 1)

    try:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch}")
            indices_i = torch.randperm(n, device=device)
            indices_j = torch.randperm(n, device=device)
            initial_loss = 0.0
            final_loss = 0.0

            for step in tqdm(range(n_minibatches), desc=f"lv epoch {epoch}"):
                if args.max_updates is not None and step >= args.max_updates:
                    print(f"Reached max updates {args.max_updates}.")
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

                edge_kk = q_i.T @ (cc @ q_j)
                non_edge_kk = q_i.T @ ((1.0 - cc) @ q_j)
                e_prob = torch.sigmoid(u_left @ u_right.T + bias)

                obj = (edge_kk * log_clamp(e_prob)).sum() + (
                    non_edge_kk * log_clamp(1.0 - e_prob)
                ).sum()
                obj = obj - (q_i * log_clamp(q_i)).sum(1).mean() / 2.0
                obj = obj - (q_j * log_clamp(q_j)).sum(1).mean() / 2.0
                loss = -obj

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([u_left, u_right, bias, q_logits], 1.0)
                optimizer.step()

                if step == 0:
                    initial_loss = float(loss.item())
                final_loss = float(loss.item())

            print(f"Initial loss: {initial_loss}, Final loss: {final_loss}")
            with torch.no_grad():
                n_used = int(torch.unique(torch.argmax(q_logits, dim=-1)).numel())
                print(f"Unique hard clusters in use: {n_used}/{args.k}")
    except KeyboardInterrupt:
        print("Training interrupted.")

    with torch.no_grad():
        assignments = torch.argmax(q_logits, dim=-1).cpu().numpy()
        scores = torch.max(torch.softmax(q_logits, dim=-1), dim=-1).values.cpu().numpy()

    mapping = load_mapping(args.mapping)
    cluster_assignment_dict = {mapping[i]: int(assignments[i]) for i in range(len(assignments))}
    prefix = Path(args.out_prefix)
    np.save(f"{prefix}_assignments.npy", assignments)
    np.save(f"{prefix}_scores.npy", scores)
    np.save(f"{prefix}_U_left.npy", u_left.detach().cpu().numpy())
    np.save(f"{prefix}_U_right.npy", u_right.detach().cpu().numpy())
    np.save(f"{prefix}_assignment_dict.npy", cluster_assignment_dict)
    print(f"Saved results with prefix {prefix}_*")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adjacency", default="sparse_connectivity_matrix.npz")
    p.add_argument("--mapping", default="root_id_to_index_mapping.json")
    p.add_argument("--out-prefix", default="lv")
    p.add_argument("--k", type=int, default=729)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--minibatch", type=int, default=2500)
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
