"""Fold one downloaded sweep's tables into the accumulated canonical tables.

``remote_start_unsup_sweep.sh`` clears ``hp_*``/``final_*`` on the Studio before
each sweep, so the tables retrieved after a single-method run describe *only*
that method. The canonical tables in the repository root are the accumulation of
every sweep run so far (PCA, NTAC, LV, LV$+e$, GNN, ...), and downloading a new
sweep on top of them would silently discard the rest.

``merge_lv_results.py`` and ``merge_ntac_results.py`` each solved one instance of
this by name: download over the canonical files, then lift the methods that were
*not* in the new sweep back out of a pre-download backup. That is fragile in two
ways — it requires the download to happen first, and the preserved set has to be
enumerated by hand. This script inverts the direction. The freshly downloaded
artefacts are kept in their own directory, never overwriting anything, and are
merged *in*: the methods the incoming sweep covers replace their counterparts in
the canonical tables and every other method is left untouched.

Replacement is per method rather than per run name. A sweep re-selects its own
hyperparameters, so its ``hp_best``, its finals and its ``final_summary`` form
one internally consistent statement about that method; splicing new rows into an
older grid for the same method would leave a table whose selected configuration
is not the one its finals were run at. Rows for methods absent from the incoming
sweep are carried over verbatim.

Usage::

    uv run python merge_sweep_results.py --source gnn_sweep_results_20260809
    uv run python merge_sweep_results.py --source <dir> --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from run_experiments import DIAGNOSTIC_KEYS

ROW_TARGETS = (
    "hp_results.csv",
    "hp_results.json",
    "final_results.csv",
    "final_results.json",
)
BEST_TARGET = "hp_best.json"
SUMMARY_JSON = "final_summary.json"
SUMMARY_CSV = "final_summary.csv"
DEFAULT_BACKUP_SUFFIX = ".premerge.bak"


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".csv":
        return read_csv_rows(path)
    return list(json.loads(path.read_text()))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], dry_run: bool) -> None:
    """Write ``rows`` with the sorted union of keys, as ``run_experiments.py`` does."""
    fields = sorted({k for r in rows for k in r})
    print(f"  write {path} ({len(rows)} rows, {len(fields)} columns)")
    if dry_run:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any, dry_run: bool) -> None:
    print(f"  write {path}")
    if not dry_run:
        path.write_text(json.dumps(obj, indent=2))


def write_rows(path: Path, rows: list[dict[str, Any]], dry_run: bool) -> None:
    if path.suffix == ".csv":
        write_csv_rows(path, rows, dry_run)
    else:
        write_json(path, rows, dry_run)


def methods_in(rows: list[dict[str, Any]]) -> set[str]:
    return {str(r["method"]) for r in rows if r.get("method")}


def enrich(rows: list[dict[str, Any]], source: Path) -> list[dict[str, Any]]:
    """Backfill diagnostic columns from the per-run metrics beside the tables.

    A sweep launched before a diagnostic joined ``DIAGNOSTIC_KEYS`` wrote that
    quantity into its ``*_metrics.json`` but not into its tables. The metrics
    files travel with the tables, so the column can be recovered rather than
    lost to the version of the code that happened to be on the Studio.
    """
    filled = 0
    for row in rows:
        metrics_path = source / f"{row.get('name')}_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        for key in DIAGNOSTIC_KEYS:
            if key in metrics and row.get(key) in (None, ""):
                row[key] = metrics[key]
                filled += 1
    if filled:
        print(f"  backfilled {filled} diagnostic value(s) from {source}")
    return rows


def back_up(path: Path, suffix: str, dry_run: bool) -> None:
    if not path.exists():
        return
    backup = path.with_name(path.name + suffix)
    print(f"  backup {path} -> {backup}")
    if not dry_run:
        shutil.copy2(path, backup)


def merge_rows(
    canonical: Path, incoming: Path, methods: set[str], suffix: str, dry_run: bool
) -> None:
    rows_in = enrich(read_rows(incoming), incoming.parent)
    if not rows_in:
        print(f"{canonical}: nothing to merge from {incoming}")
        return
    kept = [r for r in read_rows(canonical) if r.get("method") not in methods]
    print(
        f"{canonical}: keeping {len(kept)} row(s), "
        f"replacing {', '.join(sorted(methods))} with {len(rows_in)} row(s)"
    )
    back_up(canonical, suffix, dry_run)
    write_rows(canonical, kept + rows_in, dry_run)


def merge_best(canonical: Path, incoming: Path, suffix: str, dry_run: bool) -> None:
    if not incoming.exists():
        print(f"{canonical}: no incoming {incoming.name}")
        return
    new = json.loads(incoming.read_text())
    best = json.loads(canonical.read_text()) if canonical.exists() else {}
    print(f"{canonical}: replacing best entries for {', '.join(sorted(new))}")
    back_up(canonical, suffix, dry_run)
    best.update(new)
    write_json(canonical, dict(sorted(best.items())), dry_run)


def flatten_summary(entry: dict[str, Any]) -> dict[str, Any]:
    row = {k: v for k, v in entry.items() if k != "best_hyperparams"}
    row.update({f"hp_{k}": v for k, v in entry.get("best_hyperparams", {}).items()})
    return row


def merge_summary(source: Path, methods: set[str], suffix: str, dry_run: bool) -> None:
    incoming = source / SUMMARY_JSON
    if not incoming.exists():
        print(f"{SUMMARY_JSON}: no incoming summary")
        return
    new = json.loads(incoming.read_text())
    path = Path(SUMMARY_JSON)
    kept = [
        s
        for s in (json.loads(path.read_text()) if path.exists() else [])
        if s["method"] not in methods
    ]
    summary = sorted(kept + new, key=lambda s: (s["method"], str(s.get("group", s["method"]))))
    print(f"{SUMMARY_JSON}: keeping {len(kept)} entrie(s), adding {len(new)}")
    back_up(path, suffix, dry_run)
    back_up(Path(SUMMARY_CSV), suffix, dry_run)
    write_json(path, summary, dry_run)
    write_csv_rows(Path(SUMMARY_CSV), [flatten_summary(s) for s in summary], dry_run)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="Directory holding one sweep's tables")
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to replace; defaults to whichever methods the incoming tables contain",
    )
    p.add_argument("--backup-suffix", default=DEFAULT_BACKUP_SUFFIX)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"No such directory: {source}")

    incoming_methods = set(args.methods or []) or (
        methods_in(read_rows(source / "hp_results.json"))
        | methods_in(read_rows(source / "final_results.json"))
    )
    if not incoming_methods:
        raise SystemExit(f"No method rows found under {source}")
    print(f"Merging {', '.join(sorted(incoming_methods))} from {source}")

    for name in ROW_TARGETS:
        merge_rows(Path(name), source / name, incoming_methods, args.backup_suffix, args.dry_run)
    merge_best(Path(BEST_TARGET), source / BEST_TARGET, args.backup_suffix, args.dry_run)
    merge_summary(source, incoming_methods, args.backup_suffix, args.dry_run)


if __name__ == "__main__":
    main()
