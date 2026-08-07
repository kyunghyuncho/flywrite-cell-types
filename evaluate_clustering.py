"""Evaluate cluster assignments against FlyWire visual neuron types.

Uses the Hungarian algorithm on the confusion matrix (same protocol as
``cluster_similarity_test.ipynb``) and also reports ARI / NMI.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def load_assignment_dict(path: Path) -> dict[Any, Any]:
    obj = np.load(path, allow_pickle=True)
    if hasattr(obj, "item"):
        obj = obj.item()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(obj)}")
    return obj


def load_ground_truth(path: Path) -> dict[Any, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def encode_labels(
    values: list[Any],
) -> tuple[np.ndarray, dict[Any, int]]:
    unique = list(dict.fromkeys(values))
    mapping = {v: i for i, v in enumerate(unique)}
    return np.array([mapping[v] for v in values], dtype=np.int64), mapping


def confusion_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    x_map = {v: i for i, v in enumerate(unique_x)}
    y_map = {v: i for i, v in enumerate(unique_y)}
    mat = np.zeros((len(unique_x), len(unique_y)), dtype=np.float64)
    for xi, yi in zip(x, y, strict=True):
        mat[x_map[xi], y_map[yi]] += 1.0
    return mat


def hungarian_score(assignment_a: np.ndarray, assignment_b: np.ndarray) -> float:
    conf = confusion_matrix(assignment_a, assignment_b)
    row_ind, col_ind = linear_sum_assignment(conf, maximize=True)
    return float(conf[row_ind, col_ind].sum())


def align_assignments(
    pred: dict[Any, Any],
    gt: dict[Any, Any],
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    shared = sorted(set(pred.keys()).intersection(gt.keys()))
    if not shared:
        raise ValueError("No shared root IDs between prediction and ground truth.")
    gt_labels, _ = encode_labels([gt[k] for k in shared])
    pred_labels, _ = encode_labels([pred[k] for k in shared])
    return gt_labels, pred_labels, shared


def evaluate_pair(pred: dict[Any, Any], gt: dict[Any, Any]) -> dict[str, float]:
    gt_labels, pred_labels, shared = align_assignments(pred, gt)
    return {
        "n_shared": float(len(shared)),
        "n_gt_clusters": float(len(np.unique(gt_labels))),
        "n_pred_clusters": float(len(np.unique(pred_labels))),
        "hungarian": hungarian_score(gt_labels, pred_labels),
        "ari": float(adjusted_rand_score(gt_labels, pred_labels)),
        "nmi": float(normalized_mutual_info_score(gt_labels, pred_labels)),
    }


def random_assignment_baseline(
    gt_labels: np.ndarray,
    k: int = 729,
    n_seeds: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    """Assign each neuron uniformly at random to one of ``k`` clusters.

    Returns mean ± std over ``n_seeds`` of Hungarian / ARI / NMI, plus the
    theoretical ceilings (perfect recovery of the GT partition on this set).
    """
    n = int(gt_labels.shape[0])
    rng = np.random.default_rng(seed)
    hun: list[float] = []
    ari: list[float] = []
    nmi: list[float] = []
    for _ in range(n_seeds):
        pred = rng.integers(0, k, size=n, dtype=np.int64)
        hun.append(hungarian_score(gt_labels, pred))
        ari.append(float(adjusted_rand_score(gt_labels, pred)))
        nmi.append(float(normalized_mutual_info_score(gt_labels, pred)))
    hun_a = np.asarray(hun, dtype=float)
    ari_a = np.asarray(ari, dtype=float)
    nmi_a = np.asarray(nmi, dtype=float)
    ddof = 1 if n_seeds > 1 else 0
    return {
        "n_shared": float(n),
        "k": float(k),
        "n_seeds": float(n_seeds),
        "hungarian_mean": float(hun_a.mean()),
        "hungarian_std": float(hun_a.std(ddof=ddof)),
        "ari_mean": float(ari_a.mean()),
        "ari_std": float(ari_a.std(ddof=ddof)),
        "nmi_mean": float(nmi_a.mean()),
        "nmi_std": float(nmi_a.std(ddof=ddof)),
        # Perfect recovery of the GT labels on this shared set.
        "hungarian_max": float(n),
        "ari_max": 1.0,
        "nmi_max": 1.0,
    }


def random_baselines(
    gt_labels: np.ndarray,
    seed: int = 0,
    k: int = 729,
    n_seeds: int = 20,
) -> dict[str, float]:
    """Back-compat wrapper plus the multi-seed random-assignment baseline."""
    rng = np.random.default_rng(seed)
    optimistic = rng.permutation(gt_labels)
    pessimistic = rng.integers(0, len(np.unique(gt_labels)), size=gt_labels.shape[0])
    out = {
        "random_optimistic_hungarian": hungarian_score(gt_labels, optimistic),
        "random_pessimistic_hungarian": hungarian_score(gt_labels, pessimistic),
    }
    out.update(random_assignment_baseline(gt_labels, k=k, n_seeds=n_seeds, seed=seed))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("root_id_type_dict.pkl"),
        help="Pickle of root_id -> type string",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        nargs="+",
        required=True,
        help="One or more .npy assignment dicts (root_id -> cluster id)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=729, help="Clusters for random baseline")
    parser.add_argument("--n-random-seeds", type=int, default=20)
    args = parser.parse_args()

    gt = load_ground_truth(args.gt)
    print(f"Ground truth: {len(gt)} neurons, {len(set(gt.values()))} types")

    first_gt_labels = None
    for pred_path in args.pred:
        pred = load_assignment_dict(pred_path)
        metrics = evaluate_pair(pred, gt)
        if first_gt_labels is None:
            first_gt_labels, _, _ = align_assignments(pred, gt)
        print(f"\n=== {pred_path} ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    if first_gt_labels is not None:
        baselines = random_baselines(
            first_gt_labels, seed=args.seed, k=args.k, n_seeds=args.n_random_seeds
        )
        print("\n=== random baselines (uniform over K clusters) ===")
        for key in (
            "hungarian_mean",
            "hungarian_std",
            "ari_mean",
            "nmi_mean",
            "hungarian_max",
        ):
            print(f"  {key}: {baselines[key]}")
        print(
            f"  (also optimistic={baselines['random_optimistic_hungarian']:.1f}, "
            f"pessimistic={baselines['random_pessimistic_hungarian']:.1f})"
        )


if __name__ == "__main__":
    main()
