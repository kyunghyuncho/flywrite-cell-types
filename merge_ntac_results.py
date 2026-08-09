"""Merge the truncated NTAC-only sweep into the canonical result artefacts.

The NTAC sweep was interrupted before ``run_experiments.py`` could write its
finals artefacts, so only the HP records (``ntac_hp_results.{csv,json}``,
``ntac_hp_best.json``) and the per-seed assignment dictionaries were retrieved.
Ground-truth metrics for the finals are therefore *recomputed locally* from the
saved assignment dictionaries with the same ``evaluate_pair`` protocol used by
``run_experiments.py``; the unsupervised selection metrics are taken from the
surviving remote log (see ``FINAL_VAL_METRICS``).

The script is idempotent: pre-existing ``ntac`` rows are dropped before the
reconstructed ones are appended, and originals are backed up once to
``*.prentac.bak``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_clustering import evaluate_pair, load_assignment_dict, load_ground_truth

METHOD = "ntac"
BACKUP_SUFFIX = ".prentac.bak"

# Unsupervised val_metric per final seed. Seed 0 is numerically identical to the
# HP run (NTAC is deterministic given the graph; the seed only permutes vertex
# order), so its value is read from ``ntac_hp_results.json`` at full precision.
# Seed 1's ``*_metrics.json`` never reached us, so we use the six-decimal value
# printed in the remote log.
FINAL_VAL_METRICS: dict[int, float | None] = {0: None, 1: -0.482294}


def backup_once(path: Path, dry_run: bool) -> None:
    """Copy ``path`` to ``path.prentac.bak`` unless a backup already exists."""
    if not path.exists():
        return
    dest = path.with_name(path.name + BACKUP_SUFFIX)
    if dest.exists():
        print(f"backup exists, keeping {dest}")
        return
    print(f"backup {path} -> {dest}")
    if not dry_run:
        shutil.copy2(path, dest)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], dry_run: bool) -> None:
    """Write ``rows`` with the sorted union of keys, as ``run_experiments.py`` does."""
    fields = sorted({k for r in rows for k in r})
    print(f"write {path} ({len(rows)} rows, {len(fields)} columns)")
    if dry_run:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any, dry_run: bool) -> None:
    print(f"write {path}")
    if not dry_run:
        path.write_text(json.dumps(obj, indent=2))


def drop_method(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("method") != method]


def final_prefix(hp: dict[str, Any], seed: int) -> str:
    return f"final_{METHOD}_k{hp['max_k']}_R{hp['max_iterations']}_T{hp['frac_seeds']}_seed{seed}"


def build_final_rows(best: dict[str, Any], hp_row: dict[str, Any], gt_path: Path) -> list[dict]:
    """Reconstruct the finals rows ``run_experiments.py`` would have written."""
    gt = load_ground_truth(gt_path)
    hp = best["hyperparams"]
    rows: list[dict[str, Any]] = []
    for seed, val_metric in sorted(FINAL_VAL_METRICS.items()):
        prefix = final_prefix(hp, seed)
        pred_path = Path(f"{prefix}_assignment_dict.npy")
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        gt_metrics = evaluate_pair(load_assignment_dict(pred_path), gt)
        rows.append(
            {
                "phase": "final",
                "name": prefix,
                "method": METHOD,
                **hp,
                "seed": seed,
                "val_metric": float(hp_row["val_metric"]) if val_metric is None else val_metric,
                **{f"gt_{k}": v for k, v in gt_metrics.items()},
            }
        )
        print(
            f"FINAL {prefix}: val={rows[-1]['val_metric']:.6f} "
            f"gt_hungarian={rows[-1]['gt_hungarian']:.1f}"
        )
    return rows


def summary_entry(final_rows: list[dict[str, Any]], best: dict[str, Any]) -> dict[str, Any]:
    hun = np.array([float(r["gt_hungarian"]) for r in final_rows], dtype=float)
    ari = np.array([float(r["gt_ari"]) for r in final_rows], dtype=float)
    nmi = np.array([float(r["gt_nmi"]) for r in final_rows], dtype=float)
    n = len(final_rows)
    return {
        "method": METHOD,
        "n_seeds": n,
        "hungarian_mean": float(hun.mean()),
        "hungarian_std": float(hun.std(ddof=1)) if n > 1 else 0.0,
        "ari_mean": float(ari.mean()),
        "ari_std": float(ari.std(ddof=1)) if n > 1 else 0.0,
        "nmi_mean": float(nmi.mean()),
        "nmi_std": float(nmi.std(ddof=1)) if n > 1 else 0.0,
        "best_hyperparams": best["hyperparams"],
        "selection_val_metric": best["val_metric"],
    }


def flatten_summary(entry: dict[str, Any]) -> dict[str, Any]:
    row = {k: v for k, v in entry.items() if k != "best_hyperparams"}
    row.update({f"hp_{k}": v for k, v in entry["best_hyperparams"].items()})
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=Path, default=Path("root_id_type_dict.pkl"))
    p.add_argument("--ntac-hp-results", type=Path, default=Path("ntac_hp_results.json"))
    p.add_argument("--ntac-hp-best", type=Path, default=Path("ntac_hp_best.json"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ntac_hp_rows = json.loads(args.ntac_hp_results.read_text())
    ntac_best = json.loads(args.ntac_hp_best.read_text())[METHOD]
    hp_row = next(r for r in ntac_hp_rows if r["name"] == ntac_best["name"])

    final_rows = build_final_rows(ntac_best, hp_row, args.gt)
    entry = summary_entry(final_rows, ntac_best)
    print(
        f"SUMMARY {METHOD}: Hungarian "
        f"{entry['hungarian_mean']:.1f} ± {entry['hungarian_std']:.1f} (n_seeds={entry['n_seeds']})"
    )

    targets = [
        Path("hp_results.csv"),
        Path("hp_results.json"),
        Path("hp_best.json"),
        Path("final_results.csv"),
        Path("final_results.json"),
        Path("final_summary.csv"),
        Path("final_summary.json"),
    ]
    for path in targets:
        backup_once(path, args.dry_run)

    # HP phase.
    hp_csv = drop_method(read_csv_rows(Path("hp_results.csv")), METHOD) + ntac_hp_rows
    write_csv_rows(Path("hp_results.csv"), hp_csv, args.dry_run)
    hp_json_path = Path("hp_results.json")
    if hp_json_path.exists():
        merged = drop_method(json.loads(hp_json_path.read_text()), METHOD) + ntac_hp_rows
        write_json(hp_json_path, merged, args.dry_run)
    best_path = Path("hp_best.json")
    if best_path.exists():
        best_all = json.loads(best_path.read_text())
        best_all[METHOD] = ntac_best
        write_json(best_path, best_all, args.dry_run)

    # Finals phase.
    final_csv = drop_method(read_csv_rows(Path("final_results.csv")), METHOD) + final_rows
    write_csv_rows(Path("final_results.csv"), final_csv, args.dry_run)
    final_json_path = Path("final_results.json")
    if final_json_path.exists():
        merged = drop_method(json.loads(final_json_path.read_text()), METHOD) + final_rows
        write_json(final_json_path, merged, args.dry_run)

    # Summary (sorted by method, matching ``run_experiments.py``).
    summary_json_path = Path("final_summary.json")
    summary = drop_method(json.loads(summary_json_path.read_text()), METHOD) + [entry]
    summary.sort(key=lambda s: s["method"])
    write_json(summary_json_path, summary, args.dry_run)
    write_csv_rows(Path("final_summary.csv"), [flatten_summary(s) for s in summary], args.dry_run)


if __name__ == "__main__":
    main()
