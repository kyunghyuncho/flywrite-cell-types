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


def random_baselines(
    gt_labels: np.ndarray,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    optimistic = rng.permutation(gt_labels)
    pessimistic = rng.integers(0, len(np.unique(gt_labels)), size=gt_labels.shape[0])
    return {
        "random_optimistic_hungarian": hungarian_score(gt_labels, optimistic),
        "random_pessimistic_hungarian": hungarian_score(gt_labels, pessimistic),
    }


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
        baselines = random_baselines(first_gt_labels, seed=args.seed)
        print("\n=== random baselines ===")
        for k, v in baselines.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
