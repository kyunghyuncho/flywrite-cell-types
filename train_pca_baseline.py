"""PCA + k-means baseline with CLI arguments."""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import torch
from scipy.sparse import load_npz

from index_mapping import load_mapping
from sparse_graph_pca import kmeans_clustering, orthogonalize, stochastic_pca


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adjacency", default="sparse_connectivity_matrix.npz")
    p.add_argument("--mapping", default="root_id_to_index_mapping.json")
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
    print(f"Loaded A with shape {adj.shape}, nnz={adj.nnz}, device={device}")

    u, _b = stochastic_pca(
        adj,
        args.d,
        batch_size=args.batch_size,
        lr=args.lr,
        max_iter=args.max_iter,
        tol=1e-6,
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
    print(f"Saved results with prefix {prefix}_*")


if __name__ == "__main__":
    main()
