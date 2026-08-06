"""PCA + k-means baseline with held-out reconstruction for model selection."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import load_npz

from heldout import load_heldout, write_metrics
from index_mapping import load_mapping
from sparse_graph_pca import kmeans_clustering, orthogonalize


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def stochastic_pca_heldout(
    w_csr,
    n_components: int,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    batch_size: int = 512,
    lr: float = 0.01,
    max_iter: int = 10_000,
    device: str = "cuda",
):
    """SGD PCA trained only on ``train_rows``; report negated val MSE (higher better)."""
    n, _m = w_csr.shape
    d = n_components
    u = torch.randn(n, d, device=device, requires_grad=True)
    b = torch.tensor([float(w_csr.mean())], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([u, b], lr=lr)

    def row_recon_diff(indices: np.ndarray) -> torch.Tensor:
        w_batch = torch.tensor(w_csr[indices].toarray(), dtype=torch.float32, device=device)
        # Same reconstruction as sparse_graph_pca.stochastic_pca: U U^T W + b.
        ut_w = torch.mm(w_batch, u)
        w_hat = torch.mm(ut_w, u.T)
        return w_hat - w_batch + b

    for it in range(max_iter):
        batch = np.random.choice(train_rows, size=min(batch_size, len(train_rows)), replace=False)
        diff = row_recon_diff(batch)
        loss = torch.norm(diff, p="fro") ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % 500 == 0:
            with torch.no_grad():
                val = float(torch.mean(row_recon_diff(val_rows) ** 2).item())
            print(f"PCA iter {it}: train_fro={float(loss.item()):.4f} val_mse={val:.6f}")

    with torch.no_grad():
        val_mse = float(torch.mean(row_recon_diff(val_rows) ** 2).item())
    return u.detach(), b.detach(), val_mse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adjacency", default="sparse_connectivity_matrix.npz")
    p.add_argument("--mapping", default="root_id_to_index_mapping.json")
    p.add_argument("--heldout-rows", default="heldout_rows.npz")
    p.add_argument("--out-prefix", default="pca")
    p.add_argument("--k", type=int, default=729)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-iter", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.seed is None:
        args.seed = int(datetime.now().timestamp())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    adj = load_npz(args.adjacency)
    n = adj.shape[0]
    print(f"Loaded A with shape {adj.shape}, nnz={adj.nnz}, device={device}")

    held = load_heldout(Path(args.heldout_rows))
    val_rows = held["rows"]
    train_rows = np.setdiff1d(np.arange(n), val_rows, assume_unique=False)

    u, _b, val_mse = stochastic_pca_heldout(
        adj,
        args.d,
        train_rows=train_rows,
        val_rows=val_rows,
        batch_size=args.batch_size,
        lr=args.lr,
        max_iter=args.max_iter,
        device=device,
    )
    u_orth = orthogonalize(u)
    centers, labels, min_distances = kmeans_clustering(u_orth, args.k, device=device)

    mapping = load_mapping(args.mapping)
    labels_np = labels.cpu().numpy()
    cluster_assignment_dict = {mapping[i]: int(labels_np[i]) for i in range(len(labels_np))}

    prefix = args.out_prefix
    torch.save(u_orth.cpu(), f"{prefix}_U_orth.pt")
    torch.save(centers.cpu(), f"{prefix}_cluster_centers.pt")
    torch.save(labels.cpu(), f"{prefix}_labels.pt")
    torch.save(min_distances.cpu(), f"{prefix}_min_distances.pt")
    np.save(f"{prefix}_assignment_dict.npy", cluster_assignment_dict)

    # Higher is better across methods: store negated reconstruction MSE.
    val_metric = -val_mse
    metrics = {
        "method": "pca_kmeans",
        "val_metric": val_metric,
        "val_metric_name": "neg_heldout_row_mse",
        "val_metric_higher_is_better": True,
        "val_mse": val_mse,
        "seed": args.seed,
        "k": args.k,
        "d": args.d,
        "lr": args.lr,
        "max_iter": args.max_iter,
        "n_pred_clusters": int(len(np.unique(labels_np))),
    }
    write_metrics(f"{prefix}_metrics.json", metrics)
    print(f"Saved {prefix}_* ; val_mse={val_mse:.6f} val_metric={val_metric:.6f}")


if __name__ == "__main__":
    main()
