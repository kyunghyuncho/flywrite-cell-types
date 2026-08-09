"""Re-attach earlier PCA/NTAC records to freshly downloaded LV artefacts.

``remote_start_unsup_sweep.sh`` clears ``hp_*``/``final_*`` on the Studio before
each sweep, so the artefacts retrieved after an ``--methods lv`` run contain
*only* the latent-variable rows. The earlier PCA baseline and the reconstructed
NTAC records (see ``merge_ntac_results.py``) live solely in the local files that
the download overwrites, and we want them retained for cross-method comparison.

This script therefore performs the mirror image of ``merge_ntac_results.py``: the
methods listed in ``PRESERVE_METHODS`` are lifted out of the pre-download backup
(``*<BACKUP_SUFFIX>``, written before the download) and appended to the new
artefacts. It is idempotent — any pre-existing rows for the preserved methods are
dropped before the backup copies are appended — and it never touches the
latent-variable rows produced by the new sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PRESERVE_METHODS = ("pca", "ntac")
BACKUP_SUFFIX = ".prelv.bak"

ROW_TARGETS = (
    "hp_results.csv",
    "hp_results.json",
    "final_results.csv",
    "final_results.json",
)
BEST_TARGET = "hp_best.json"
SUMMARY_JSON = "final_summary.json"
SUMMARY_CSV = "final_summary.csv"


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


def keep(rows: list[dict[str, Any]], methods: tuple[str, ...]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("method") in methods]


def drop(rows: list[dict[str, Any]], methods: tuple[str, ...]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("method") not in methods]


def backup_of(path: Path) -> Path:
    return path.with_name(path.name + BACKUP_SUFFIX)


def flatten_summary(entry: dict[str, Any]) -> dict[str, Any]:
    row = {k: v for k, v in entry.items() if k != "best_hyperparams"}
    row.update({f"hp_{k}": v for k, v in entry.get("best_hyperparams", {}).items()})
    return row


def merge_rows(path: Path, methods: tuple[str, ...], dry_run: bool) -> None:
    backup = backup_of(path)
    if not backup.exists():
        print(f"no backup for {path}; nothing to restore")
        return
    is_csv = path.suffix == ".csv"
    reader = read_csv_rows if is_csv else (lambda p: json.loads(p.read_text()))
    restored = keep(reader(backup), methods)
    merged = drop(reader(path), methods) + restored
    print(f"{path}: restoring {len(restored)} row(s) for {', '.join(methods)}")
    if is_csv:
        write_csv_rows(path, merged, dry_run)
    else:
        write_json(path, merged, dry_run)


def merge_best(path: Path, methods: tuple[str, ...], dry_run: bool) -> None:
    backup = backup_of(path)
    if not backup.exists() or not path.exists():
        print(f"skip {path} (missing file or backup)")
        return
    best = json.loads(path.read_text())
    old = json.loads(backup.read_text())
    for method in methods:
        if method in old:
            best[method] = old[method]
            print(f"{path}: restoring best entry for {method}")
    write_json(path, dict(sorted(best.items())), dry_run)


def merge_summary(methods: tuple[str, ...], dry_run: bool) -> None:
    path, backup = Path(SUMMARY_JSON), backup_of(Path(SUMMARY_JSON))
    if not backup.exists() or not path.exists():
        print(f"skip {path} (missing file or backup)")
        return
    restored = keep(json.loads(backup.read_text()), methods)
    summary = drop(json.loads(path.read_text()), methods) + restored
    summary.sort(key=lambda s: s["method"])
    print(f"{path}: restoring {len(restored)} summary entrie(s)")
    write_json(path, summary, dry_run)
    write_csv_rows(Path(SUMMARY_CSV), [flatten_summary(s) for s in summary], dry_run)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(PRESERVE_METHODS),
        help="Methods to lift out of the backup and re-append",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    methods = tuple(args.methods)

    for name in ROW_TARGETS:
        merge_rows(Path(name), methods, args.dry_run)
    merge_best(Path(BEST_TARGET), methods, args.dry_run)
    merge_summary(methods, args.dry_run)


if __name__ == "__main__":
    main()
