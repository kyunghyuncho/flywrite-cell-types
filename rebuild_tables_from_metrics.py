"""Rebuild a sweep's ``hp_*`` tables from the per-run metrics it left behind.

``run_experiments.py`` accumulates HP rows in memory and writes ``hp_results.*``
and ``hp_best.json`` only after the whole grid finishes. A sweep stopped part way
-- because it diverged, ran out of time, or was cut short once its trend was
clear -- therefore leaves every completed run's ``<name>_metrics.json`` on disk
with no table tying them together, and ``merge_sweep_results.py`` has nothing to
consume.

This reconstructs those tables from the metrics files alone, so a truncated sweep
is merged on the same footing as a complete one. Two differences from the live
path are deliberate:

* Ground-truth columns are copied from the metrics rather than recomputed. The
  trainers already score the selected assignment against the ground truth, and
  the assignment ``.npy`` files are large enough that they are usually left on
  the Studio; copying keeps the reconstruction runnable from the metrics alone.
* Hyperparameter columns are read back off each run, not re-derived from a grid
  specification. The grid that produced a partial sweep is not recoverable from
  its output, and inferring one risks labelling rows with a configuration they
  were not run at.

The reconstructed ``hp_best`` is the argmax of the *completed* runs, which is a
weaker statement than a finished sweep's -- it is the best of what ran, not the
best of what was asked for. ``partial`` records that on every row.

Usage::

    uv run python rebuild_tables_from_metrics.py --source <dir> --method lv
    uv run python rebuild_tables_from_metrics.py --source <dir> --method lv --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from run_experiments import DIAGNOSTIC_KEYS, u_norm_hyperparams, u_norm_of

# Per method, the columns that identify a point in the sweep grid. These mirror
# the ``hyperparams`` dicts assembled in ``run_experiments.hp_specs``.
HYPERPARAM_KEYS: dict[str, tuple[str, ...]] = {
    "lv": ("d", "lr", "bfs_frac", "likelihood", "label_smoothing"),
    "pca": ("d", "lr"),
    "ntac": ("d", "max_k", "max_iterations", "frac_seeds"),
}
GT_KEYS = ("n_shared", "n_gt_clusters", "n_pred_clusters", "hungarian", "ari", "nmi")


def hp_of(method: str, metrics: dict) -> dict[str, Any]:
    """The grid coordinates of one run, as its tables would have recorded them."""
    hp = {k: metrics[k] for k in HYPERPARAM_KEYS.get(method, ()) if k in metrics}
    if method == "lv":
        hp.update(u_norm_hyperparams(u_norm_of(metrics)))
    return hp


def row_of(method: str, name: str, metrics: dict) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": "hp",
        "name": name,
        "method": method,
        **hp_of(method, metrics),
        "val_metric": metrics["val_metric"],
        "val_metric_name": metrics["val_metric_name"],
        "val_native_ll": metrics.get("val_native_ll"),
        "val_native_ll_name": metrics.get("val_native_ll_name"),
        "n_pred_clusters": metrics.get("n_pred_clusters"),
    }
    row.update({f"gt_{k}": metrics[f"gt_{k}"] for k in GT_KEYS if f"gt_{k}" in metrics})
    row.update({k: metrics[k] for k in DIAGNOSTIC_KEYS if k in metrics})
    row.update({k: v for k, v in metrics.items() if k.startswith("last_gt_")})
    # Distinguishes a row recovered from a truncated sweep from one written by a
    # grid that ran to completion.
    row["partial"] = True
    return row


def best_hyperparams(method: str, best: dict) -> dict[str, Any]:
    """Mirror the selected-hyperparameter record ``run_experiments`` would write."""
    if method == "ntac":
        return {
            "d": int(best["d"]),
            "lr": float(best.get("lr", 0.0)),
            "max_k": int(best["max_k"]),
            "max_iterations": int(best["max_iterations"]),
            "frac_seeds": float(best["frac_seeds"]),
        }
    hp: dict[str, Any] = {"d": int(best["d"]), "lr": float(best["lr"])}
    if method == "lv":
        hp["bfs_frac"] = float(best["bfs_frac"])
        hp["likelihood"] = str(best["likelihood"])
        hp["label_smoothing"] = float(best.get("label_smoothing", 0.0))
        hp.update(u_norm_hyperparams(u_norm_of(best)))
        if best.get("target_updates"):
            hp["target_updates"] = int(best["target_updates"])
    return hp


def write(path: Path, rows: list[dict[str, Any]], dry_run: bool) -> None:
    print(f"  write {path} ({len(rows)} rows)")
    if dry_run:
        return
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, indent=2))
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="Directory of one sweep's *_metrics.json")
    p.add_argument(
        "--method",
        required=True,
        choices=["pca", "lv", "ntac"],
        help="Method these runs belong to",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"No such directory: {source}")

    paths = sorted(source.glob(f"hp_{args.method}_*_metrics.json"))
    if not paths:
        raise SystemExit(f"No hp_{args.method}_*_metrics.json under {source}")

    rows = [
        row_of(args.method, path.name.removesuffix("_metrics.json"), json.loads(path.read_text()))
        for path in paths
    ]
    rows.sort(key=lambda r: -r["val_metric"])
    print(f"Rebuilding {len(rows)} {args.method} row(s) from {source}")
    for r in rows:
        print(
            f"  {r['name']}: val={r['val_metric']:.6f} "
            f"gt_hungarian={r.get('gt_hungarian', float('nan')):.1f}"
        )

    best = rows[0]
    print(f"BEST {args.method} by val_metric: {best['name']} ({best['val_metric']:.6f})")
    hp_best = {
        args.method: {
            "name": best["name"],
            "val_metric": best["val_metric"],
            "hyperparams": best_hyperparams(args.method, best),
            "partial": True,
        }
    }

    write(source / "hp_results.json", rows, args.dry_run)
    write(source / "hp_results.csv", rows, args.dry_run)
    print(f"  write {source / 'hp_best.json'}")
    if not args.dry_run:
        (source / "hp_best.json").write_text(json.dumps(hp_best, indent=2))


if __name__ == "__main__":
    main()
