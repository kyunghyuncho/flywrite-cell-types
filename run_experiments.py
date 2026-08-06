"""Run baselines + GNN-vSBM hyperparameter sweep and summarize Hungarian scores.

Designed to run on a Lightning Studio GPU. Writes ``sweep_results.json`` and
``sweep_results.csv`` in the working directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from evaluate_clustering import evaluate_pair, load_assignment_dict, load_ground_truth


@dataclass(frozen=True)
class RunSpec:
    name: str
    command: list[str]
    pred_path: str


def run_cmd(cmd: list[str]) -> None:
    print("\n=== RUN ===")
    print(" ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    dt = time.time() - t0
    print(f"exit={proc.returncode} elapsed={dt:.1f}s")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    py = sys.executable
    device = args.device
    seed = args.seed
    specs: list[RunSpec] = []

    # PCA baseline
    specs.append(
        RunSpec(
            name="pca_kmeans",
            command=[
                py,
                "train_pca_baseline.py",
                "--k",
                str(args.k),
                "--d",
                "32",
                "--max-iter",
                str(args.pca_max_iter),
                "--seed",
                str(seed),
                "--device",
                device,
                "--out-prefix",
                "pca_sweep",
            ],
            pred_path="pca_sweep_assignment_dict.npy",
        )
    )

    # Low-rank vSBM baseline
    specs.append(
        RunSpec(
            name="lv_vsbm",
            command=[
                py,
                "train_lv_vsbm.py",
                "--k",
                str(args.k),
                "--d",
                "32",
                "--epochs",
                str(args.epochs),
                "--minibatch",
                str(args.minibatch),
                "--lr",
                "0.1",
                "--seed",
                str(seed),
                "--device",
                device,
                "--out-prefix",
                "lv_sweep",
            ],
            pred_path="lv_sweep_assignment_dict.npy",
        )
    )

    # GNN hyperparameter grid
    for layers in args.layers:
        for lr in args.lrs:
            for d in args.dims:
                for entropy in args.entropy_weights:
                    name = f"gnn_L{layers}_d{d}_lr{lr}_ent{entropy}"
                    cmd = [
                        py,
                        "gnn_vsbm.py",
                        "--k",
                        str(args.k),
                        "--d",
                        str(d),
                        "--layers",
                        str(layers),
                        "--epochs",
                        str(args.epochs),
                        "--minibatch",
                        str(args.minibatch),
                        "--lr",
                        str(lr),
                        "--entropy-weight",
                        str(entropy),
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                        "--out-prefix",
                        name,
                    ]
                    specs.append(
                        RunSpec(
                            name=name,
                            command=cmd,
                            pred_path=f"{name}_assignment_dict_729.npy",
                        )
                    )
    return specs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=729)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--pca-max-iter", type=int, default=10_000)
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--lrs", type=float, nargs="+", default=[0.005, 0.01, 0.05])
    p.add_argument("--dims", type=int, nargs="+", default=[32, 64])
    p.add_argument("--entropy-weights", type=float, nargs="+", default=[1.0])
    p.add_argument("--gt", default="root_id_type_dict.pkl")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--only", nargs="*", default=None, help="Optional subset of run names")
    args = p.parse_args()

    specs = build_specs(args)
    if args.only:
        only = set(args.only)
        specs = [s for s in specs if s.name in only]
        missing = only - {s.name for s in specs}
        if missing:
            raise SystemExit(f"Unknown --only names: {sorted(missing)}")

    gt = load_ground_truth(Path(args.gt))
    results: list[dict] = []

    for spec in specs:
        pred_path = Path(spec.pred_path)
        if args.skip_existing and pred_path.exists():
            print(f"Skipping existing {spec.name}")
        else:
            run_cmd(spec.command)

        if not pred_path.exists():
            # gnn saves *_assignment_dict_729.npy; lv/pca save *_assignment_dict.npy
            alt = Path(str(pred_path).replace("_729.npy", ".npy"))
            if alt.exists():
                pred_path = alt
            else:
                results.append(
                    {
                        "name": spec.name,
                        "error": f"missing prediction file {spec.pred_path}",
                    }
                )
                continue

        pred = load_assignment_dict(pred_path)
        metrics = evaluate_pair(pred, gt)
        row = {"name": spec.name, "pred_path": str(pred_path), **metrics}
        results.append(row)
        print(
            f"RESULT {spec.name}: hungarian={metrics['hungarian']:.1f} "
            f"ari={metrics['ari']:.4f} nmi={metrics['nmi']:.4f} "
            f"pred_clusters={metrics['n_pred_clusters']:.0f}"
        )

    Path("sweep_results.json").write_text(json.dumps(results, indent=2))
    with Path("sweep_results.csv").open("w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    ranked = sorted(
        [r for r in results if "hungarian" in r],
        key=lambda r: r["hungarian"],
        reverse=True,
    )
    print("\n=== RANKED BY HUNGARIAN ===")
    for r in ranked:
        print(
            f"{r['hungarian']:7.1f}  {r['name']}  "
            f"(ari={r['ari']:.4f}, nmi={r['nmi']:.4f}, Khat={r['n_pred_clusters']:.0f})"
        )
    if ranked:
        Path("sweep_best.json").write_text(json.dumps(ranked[0], indent=2))
        print(f"\nBest: {ranked[0]['name']}")


if __name__ == "__main__":
    main()
