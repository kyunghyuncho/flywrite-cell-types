"""Re-score existing cluster assignments against the full-brain sink ground truth.

The published tables evaluate a partition only on the neurons that carry a
FlyWire visual type. That protocol conditions on knowing which neurons are
visual, which no unsupervised method is given. This script rebuilds the ground
truth over every node of the connectome graph, giving each unlabelled node the
single sink label ``__nonvisual__``, and re-runs the Hungarian / ARI / NMI
protocol on assignment dictionaries that already exist on disk. No model is
retrained. The classic visual-only numbers are reported alongside, on the same
predictions, so the two protocols can be compared row by row.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from evaluate_clustering import (
    NONVISUAL_LABEL,
    build_nonvisual_sink_gt,
    evaluate_pair,
    evaluate_pair_full_graph,
    load_assignment_dict,
    load_graph_root_ids,
    load_ground_truth,
    majority_class_baseline,
)

FIELDNAMES = [
    "run",
    "method",
    "seed",
    "n_nodes",
    "n_covered",
    "n_missing",
    "sink_hungarian",
    "sink_fraction",
    "sink_ari",
    "sink_nmi",
    "sink_n_pred_clusters",
    "visual_hungarian",
    "visual_fraction",
    "visual_ari",
    "visual_nmi",
    "visual_n_shared",
    "visual_n_pred_clusters",
]


def infer_method(name: str) -> str:
    lowered = name.lower()
    for key in ("ntac", "pca", "lv"):
        if key in lowered:
            return key
    return "unknown"


def infer_seed(name: str) -> str:
    match = re.search(r"seed(\d+)", name)
    return match.group(1) if match else ""


def split_oracle_prediction(sink_gt: dict[int, Any]) -> dict[int, int]:
    """The partition that knows only which neurons are visual.

    A method fitted on the visual subgraph alone is handed exactly this
    information for free, so its full-graph score has to be read against this
    reference rather than against the single-cluster floor.
    """
    return {k: int(v != NONVISUAL_LABEL) for k, v in sink_gt.items()}


def score_pred(
    run: str,
    pred: dict[Any, Any],
    sink_gt: dict[int, Any],
    visual_gt: dict[Any, Any],
) -> dict[str, Any]:
    sink = evaluate_pair_full_graph(pred, sink_gt)
    visual = evaluate_pair(pred, visual_gt)
    return {
        "run": run,
        "method": infer_method(run),
        "seed": infer_seed(run),
        "n_nodes": int(sink["n_nodes"]),
        "n_covered": int(sink["n_covered"]),
        "n_missing": int(sink["n_missing"]),
        "sink_hungarian": sink["hungarian"],
        "sink_fraction": sink["hungarian_fraction"],
        "sink_ari": sink["ari"],
        "sink_nmi": sink["nmi"],
        "sink_n_pred_clusters": int(sink["n_pred_clusters"]),
        "visual_hungarian": visual["hungarian"],
        "visual_fraction": visual["hungarian"] / visual["n_shared"],
        "visual_ari": visual["ari"],
        "visual_nmi": visual["nmi"],
        "visual_n_shared": int(visual["n_shared"]),
        "visual_n_pred_clusters": int(visual["n_pred_clusters"]),
    }


def score_one(
    pred_path: Path,
    sink_gt: dict[int, Any],
    visual_gt: dict[Any, Any],
) -> dict[str, Any]:
    run = pred_path.name.replace("_assignment_dict.npy", "")
    return score_pred(run, load_assignment_dict(pred_path), sink_gt, visual_gt)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("root_id_type_dict.pkl"),
        help="Pickle of root_id -> visual type string (729 types)",
    )
    parser.add_argument(
        "--index-mapping",
        type=Path,
        default=Path("root_id_to_index_mapping.json"),
        help="JSON of root_id -> row index; defines the graph node set",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        nargs="+",
        required=True,
        help="Assignment dicts (.npy of root_id -> cluster id) to re-score",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reeval_artifacts/nonvisual_sink_summary.csv"),
        help="Destination CSV (kept out of version control)",
    )
    args = parser.parse_args()

    visual_gt = load_ground_truth(args.gt)
    graph_root_ids = load_graph_root_ids(args.index_mapping)
    sink_gt = build_nonvisual_sink_gt(visual_gt, graph_root_ids)
    n_sink = sum(1 for v in sink_gt.values() if v == NONVISUAL_LABEL)
    print(
        f"Graph nodes: {len(sink_gt)}; visual-typed: {len(sink_gt) - n_sink}; "
        f"sink ({NONVISUAL_LABEL}): {n_sink}"
    )
    floor = majority_class_baseline(sink_gt)
    print(
        f"Single-cluster floor on the sink ground truth: "
        f"Hungarian {floor['hungarian']:.0f} ({floor['hungarian_fraction']:.1%})"
    )

    rows = [
        score_pred(
            "baseline_visual_split_oracle",
            split_oracle_prediction(sink_gt),
            sink_gt,
            visual_gt,
        )
    ]
    rows += [score_one(p, sink_gt, visual_gt) for p in sorted(args.pred)]
    for row in rows:
        print(
            f"{row['run']}: sink H={row['sink_hungarian']:.0f} "
            f"({row['sink_fraction']:.1%}) ARI={row['sink_ari']:.3f} "
            f"NMI={row['sink_nmi']:.3f} | visual H={row['visual_hungarian']:.0f} "
            f"({row['visual_fraction']:.1%}) ARI={row['visual_ari']:.3f} "
            f"NMI={row['visual_nmi']:.3f}"
        )
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
