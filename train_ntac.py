"""Unseeded NTAC baseline (Schwartzman et al., Nat Commun 2026).

Wraps the official ``ntac`` package
(https://github.com/BenJourdan/ntac) so we can evaluate approximate equitable
partitioning under the same FlyWire protocol as LV.

Unsupervised selection metric: negated mean Jaccard cost of the best
(lowest-cost) partition returned by unseeded NTAC (higher is better).
Clusters are arbitrary labels; GT Hungarian / ARI / NMI are computed later
by ``run_experiments.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz

from heldout import write_metrics
from index_mapping import load_mapping


def _patch_ntac_cuda(force_cpu: bool) -> None:
    """Make NTAC's CUDA probe return False instead of raising on machines without GPU toolchains."""
    import ntac.unseeded.utilgpu as utilgpu

    orig_validate = utilgpu.validate_cuda_toolchain

    def safe_validate() -> bool:
        if force_cpu:
            return False
        try:
            return bool(orig_validate())
        except Exception:
            return False

    utilgpu.validate_cuda_toolchain = safe_validate
    utilgpu.is_cuda_available = lambda: (
        (not force_cpu) and utilgpu.cuda.is_available() and safe_validate()
    )


def _permute_csr(adj: sp.csr_matrix, perm: np.ndarray) -> sp.csr_matrix:
    """Reorder vertices by ``perm`` (new_index -> old_index)."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    coo = adj.tocoo()
    return sp.coo_matrix(
        (coo.data, (inv[coo.row], inv[coo.col])),
        shape=adj.shape,
    ).tocsr()


def _compact_nonisolated(adj: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """Drop degree-zero vertices and return the surviving original indices.

    ``ntac.unseeded.convert.problem_from_data`` builds its vertex-name list from
    edge endpoints alone (``np.unique`` over the COO rows and columns) and then
    indexes that list with the *original* matrix indices, ``vertex_names[u]``.
    The two agree only when every vertex carries at least one edge: an isolate
    shortens the name list without shifting the indices, so an endpoint index
    beyond the truncated list raises ``IndexError``. The same mismatch also
    leaves ``problem.numv`` smaller than the ``n``-by-``n`` matrix it stores in
    ``problem.A_``, so passing isolates through is unsound even when the index
    happens to stay in range.

    The induced visual subgraph triggers this: inducing on visual neurons drops
    every cross-region edge, which strands a handful of vertices with no
    surviving in-graph partner.
    """
    row_nnz = np.diff(adj.indptr)
    col_nnz = np.diff(adj.tocsc().indptr)
    active_indices = np.flatnonzero((row_nnz + col_nnz) > 0)
    compact = adj[active_indices][:, active_indices].tocsr()
    return compact, active_indices


def train(args: argparse.Namespace) -> dict:
    if args.device == "cpu":
        force_cpu = True
    elif args.device == "cuda":
        force_cpu = False
    else:
        force_cpu = not _torch_cuda_available()
    _patch_ntac_cuda(force_cpu=force_cpu)

    from ntac import GraphData
    from ntac.unseeded import convert
    from ntac.unseeded.unseeded import solve_unseeded

    adj = load_npz(args.adjacency)
    adj.data = np.asarray(adj.data, dtype=np.float32)
    # NTAC expects positive edge weights (synapse counts). Our matrix may be binary.
    adj.data = np.maximum(adj.data, 0.0)
    adj.eliminate_zeros()
    adj = adj.tocsr()
    n_total = adj.shape[0]
    adj, active_indices = _compact_nonisolated(adj)
    n = adj.shape[0]
    n_isolates = n_total - n
    if n == 0:
        raise ValueError("NTAC requires at least one non-isolated vertex")
    print(
        f"NTAC unseeded: n={n_total} active={n} isolates={n_isolates} "
        f"nnz={adj.nnz} max_k={args.max_k} "
        f"R={args.max_iterations} T={args.frac_seeds} seed={args.seed} "
        f"device={'cpu' if force_cpu else 'cuda-if-available'}"
    )

    rng = np.random.default_rng(args.seed)
    # Seed diversity: permute vertices so greedy center growth sees a different order.
    perm = rng.permutation(n)
    adj_p = _permute_csr(adj, perm)
    a_csr = sp.csr_array(adj_p)
    dummy_labels = np.array(["0"] * n, dtype=object)
    data = GraphData(a_csr, labels=dummy_labels)

    problem = convert.problem_from_data(data)
    problem.set_device("cpu" if force_cpu else "default")
    # NTAC only offloads the weighted-Jaccard distance kernel, and it silently
    # reverts to CPU when the numba CUDA toolchain fails to link. Report the
    # kernel that was actually selected so a fallback cannot masquerade as a GPU
    # run in the sweep log.
    kernel = getattr(problem.all_distances_func, "__name__", repr(problem.all_distances_func))
    print(f"NTAC distance kernel: {kernel}", flush=True)

    best_partition, _last, history = solve_unseeded(
        problem,
        max_k=args.max_k,
        center_size=args.center_size,
        info_step=args.info_step,
        max_iterations=args.max_iterations,
        frac_seeds=args.frac_seeds,
        chunk_size=args.chunk_size,
    )
    labels_perm = np.asarray(best_partition.labels(), dtype=np.int64)
    # Map back to original vertex order.
    active_assignments = np.empty(n, dtype=np.int64)
    active_assignments[perm] = labels_perm
    # Isolates must still appear in the exported dict so that NTAC is scored on
    # exactly the vertex set LV and PCA are scored on. Equitable partitioning has
    # no evidence about them, so they go to their own residual label rather than
    # contaminating a real NTAC cluster's ground-truth match.
    isolate_label = int(active_assignments.max()) + 1 if n else 0
    assignments = np.full(n_total, isolate_label, dtype=np.int64)
    assignments[active_indices] = active_assignments
    jac = float(history[-1][1])
    n_clusters = int(len(np.unique(assignments)))
    # Higher is better across methods.
    val_metric = -jac

    mapping = load_mapping(args.mapping)
    cluster_assignment_dict = {mapping[i]: int(assignments[i]) for i in range(len(assignments))}
    prefix = Path(args.out_prefix)
    np.save(f"{prefix}_assignments.npy", assignments)
    np.save(f"{prefix}_assignment_dict.npy", cluster_assignment_dict)
    metrics = {
        "method": "ntac",
        "val_metric": val_metric,
        "val_metric_name": "neg_mean_jaccard_cost",
        "val_metric_higher_is_better": True,
        "jaccard_cost": jac,
        "seed": args.seed,
        "k": args.max_k,
        "max_k": args.max_k,
        "max_iterations": args.max_iterations,
        "frac_seeds": args.frac_seeds,
        "center_size": args.center_size,
        "n_vertices": n_total,
        "n_active_vertices": n,
        "n_isolates": n_isolates,
        "n_pred_clusters": n_clusters,
        "history_len": len(history),
    }
    write_metrics(f"{prefix}_metrics.json", metrics)
    print(
        f"Saved {prefix}_* ; jac={jac:.6f} val_metric={val_metric:.6f} "
        f"clusters={n_clusters}/{args.max_k}"
    )
    return metrics


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adjacency", default="sparse_connectivity_matrix.npz")
    p.add_argument("--mapping", default="root_id_to_index_mapping.json")
    p.add_argument("--out-prefix", default="ntac")
    p.add_argument("--max-k", type=int, default=729, help="Target / max number of clusters")
    p.add_argument(
        "--max-iterations",
        type=int,
        default=12,
        help="Seeded NTAC iterations R per unseeded step (paper default 12)",
    )
    p.add_argument(
        "--frac-seeds",
        type=float,
        default=0.1,
        help="Fraction T of nodes considered for next seed (paper default 0.1)",
    )
    p.add_argument("--center-size", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=6000)
    p.add_argument("--info-step", type=int, default=25)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="'cpu' forces CPU; else use CUDA if available")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.seed is None:
        args.seed = int(datetime.now().timestamp())
    train(args)


if __name__ == "__main__":
    main()
