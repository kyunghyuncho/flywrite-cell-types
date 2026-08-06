"""Unsupervised HP search + multi-seed final evaluation.

Protocol
--------
1. Build fixed held-out splits (no ground-truth types).
2. For each method, sweep hyperparameters and select by held-out validation
   metric (Bernoulli LL for LV/GNN; negated row reconstruction MSE for PCA).
3. Retrain the selected setting with multiple seeds.
4. Only then score against visual neuron types (Hungarian / ARI / NMI) to
   report means and standard deviations.
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

import numpy as np
from scipy.sparse import load_npz

from evaluate_clustering import evaluate_pair, load_assignment_dict, load_ground_truth
from heldout import make_heldout_pairs, make_heldout_rows, save_heldout


@dataclass(frozen=True)
class RunSpec:
    name: str
    method: str
    command: list[str]
    prefix: str
    hyperparams: dict


def run_cmd(cmd: list[str]) -> None:
    print("\n=== RUN ===")
    print(" ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    print(f"exit={proc.returncode} elapsed={time.time() - t0:.1f}s")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def ensure_heldout(args: argparse.Namespace) -> None:
    adj = load_npz(args.adjacency)
    adj.data = (adj.data > 0).astype(np.float32)
    adj.eliminate_zeros()

    pairs_path = Path(args.heldout_pairs)
    rows_path = Path(args.heldout_rows)
    if not pairs_path.exists():
        print(f"Creating held-out pairs at {pairs_path} ...")
        pairs = make_heldout_pairs(
            adj, n_pos=args.n_heldout_pos, n_neg=args.n_heldout_neg, seed=args.split_seed
        )
        save_heldout(pairs_path, **pairs)
    if not rows_path.exists():
        print(f"Creating held-out rows at {rows_path} ...")
        rows = make_heldout_rows(adj.shape[0], fraction=args.row_holdout, seed=args.split_seed)
        save_heldout(rows_path, rows=rows)


def hp_specs(args: argparse.Namespace) -> list[RunSpec]:
    py = sys.executable
    specs: list[RunSpec] = []

    for d in args.pca_dims:
        for lr in args.pca_lrs:
            name = f"hp_pca_d{d}_lr{lr}"
            prefix = name
            specs.append(
                RunSpec(
                    name=name,
                    method="pca",
                    prefix=prefix,
                    hyperparams={"d": d, "lr": lr},
                    command=[
                        py,
                        "train_pca_baseline.py",
                        "--k",
                        str(args.k),
                        "--d",
                        str(d),
                        "--lr",
                        str(lr),
                        "--max-iter",
                        str(args.pca_max_iter),
                        "--seed",
                        str(args.split_seed),
                        "--device",
                        args.device,
                        "--heldout-rows",
                        args.heldout_rows,
                        "--out-prefix",
                        prefix,
                    ],
                )
            )

    for d in args.lv_dims:
        for lr in args.lv_lrs:
            name = f"hp_lv_d{d}_lr{lr}"
            prefix = name
            specs.append(
                RunSpec(
                    name=name,
                    method="lv",
                    prefix=prefix,
                    hyperparams={"d": d, "lr": lr},
                    command=[
                        py,
                        "train_lv_vsbm.py",
                        "--k",
                        str(args.k),
                        "--d",
                        str(d),
                        "--lr",
                        str(lr),
                        "--epochs",
                        str(args.epochs),
                        "--minibatch",
                        str(args.minibatch),
                        "--seed",
                        str(args.split_seed),
                        "--device",
                        args.device,
                        "--heldout-pairs",
                        args.heldout_pairs,
                        "--out-prefix",
                        prefix,
                    ],
                )
            )

    for layers in args.gnn_layers:
        for d in args.gnn_dims:
            for lr in args.gnn_lrs:
                name = f"hp_gnn_L{layers}_d{d}_lr{lr}"
                prefix = name
                specs.append(
                    RunSpec(
                        name=name,
                        method="gnn",
                        prefix=prefix,
                        hyperparams={"layers": layers, "d": d, "lr": lr},
                        command=[
                            py,
                            "gnn_vsbm.py",
                            "--k",
                            str(args.k),
                            "--d",
                            str(d),
                            "--layers",
                            str(layers),
                            "--lr",
                            str(lr),
                            "--epochs",
                            str(args.epochs),
                            "--minibatch",
                            str(args.minibatch),
                            "--seed",
                            str(args.split_seed),
                            "--device",
                            args.device,
                            "--heldout-pairs",
                            args.heldout_pairs,
                            "--out-prefix",
                            prefix,
                        ],
                    )
                )
    return specs


def final_specs(args: argparse.Namespace, best_by_method: dict[str, dict]) -> list[RunSpec]:
    py = sys.executable
    specs: list[RunSpec] = []
    for method, best in best_by_method.items():
        hp = best["hyperparams"]
        for seed in args.final_seeds:
            if method == "pca":
                name = f"final_pca_d{hp['d']}_lr{hp['lr']}_seed{seed}"
                cmd = [
                    py,
                    "train_pca_baseline.py",
                    "--k",
                    str(args.k),
                    "--d",
                    str(hp["d"]),
                    "--lr",
                    str(hp["lr"]),
                    "--max-iter",
                    str(args.pca_max_iter),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--heldout-rows",
                    args.heldout_rows,
                    "--out-prefix",
                    name,
                ]
            elif method == "lv":
                name = f"final_lv_d{hp['d']}_lr{hp['lr']}_seed{seed}"
                cmd = [
                    py,
                    "train_lv_vsbm.py",
                    "--k",
                    str(args.k),
                    "--d",
                    str(hp["d"]),
                    "--lr",
                    str(hp["lr"]),
                    "--epochs",
                    str(args.epochs),
                    "--minibatch",
                    str(args.minibatch),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--heldout-pairs",
                    args.heldout_pairs,
                    "--out-prefix",
                    name,
                ]
            else:
                name = f"final_gnn_L{hp['layers']}_d{hp['d']}_lr{hp['lr']}_seed{seed}"
                cmd = [
                    py,
                    "gnn_vsbm.py",
                    "--k",
                    str(args.k),
                    "--d",
                    str(hp["d"]),
                    "--layers",
                    str(hp["layers"]),
                    "--lr",
                    str(hp["lr"]),
                    "--epochs",
                    str(args.epochs),
                    "--minibatch",
                    str(args.minibatch),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--heldout-pairs",
                    args.heldout_pairs,
                    "--out-prefix",
                    name,
                ]
            specs.append(
                RunSpec(
                    name=name,
                    method=method,
                    prefix=name,
                    hyperparams={**hp, "seed": seed},
                    command=cmd,
                )
            )
    return specs


def read_metrics(prefix: str) -> dict:
    path = Path(f"{prefix}_metrics.json")
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def pred_path_for(prefix: str) -> Path:
    for cand in (f"{prefix}_assignment_dict.npy", f"{prefix}_assignment_dict_729.npy"):
        p = Path(cand)
        if p.exists():
            return p
    raise FileNotFoundError(f"No assignment dict for prefix {prefix}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adjacency", default="sparse_connectivity_matrix.npz")
    p.add_argument("--gt", default="root_id_type_dict.pkl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--k", type=int, default=729)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--pca-max-iter", type=int, default=10_000)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--heldout-pairs", default="heldout_pairs.npz")
    p.add_argument("--heldout-rows", default="heldout_rows.npz")
    p.add_argument("--n-heldout-pos", type=int, default=50_000)
    p.add_argument("--n-heldout-neg", type=int, default=50_000)
    p.add_argument("--row-holdout", type=float, default=0.1)
    p.add_argument("--pca-dims", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--pca-lrs", type=float, nargs="+", default=[0.001, 0.01])
    p.add_argument("--lv-dims", type=int, nargs="+", default=[32, 64])
    p.add_argument("--lv-lrs", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    p.add_argument("--gnn-layers", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--gnn-dims", type=int, nargs="+", default=[32, 64])
    p.add_argument("--gnn-lrs", type=float, nargs="+", default=[0.005, 0.01, 0.05])
    p.add_argument("--final-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--phase", choices=["all", "hp", "final"], default="all")
    args = p.parse_args()

    ensure_heldout(args)
    gt = load_ground_truth(Path(args.gt))

    hp_rows: list[dict] = []
    best_by_method: dict[str, dict] = {}

    if args.phase in {"all", "hp"}:
        for spec in hp_specs(args):
            metrics_path = Path(f"{spec.prefix}_metrics.json")
            if args.skip_existing and metrics_path.exists():
                print(f"Skipping existing HP run {spec.name}")
            else:
                run_cmd(spec.command)
            metrics = read_metrics(spec.prefix)
            row = {
                "phase": "hp",
                "name": spec.name,
                "method": spec.method,
                **spec.hyperparams,
                "val_metric": metrics["val_metric"],
                "val_metric_name": metrics["val_metric_name"],
                "n_pred_clusters": metrics.get("n_pred_clusters"),
            }
            # Optional GT score for analysis only — NOT used for selection.
            pred = load_assignment_dict(pred_path_for(spec.prefix))
            gt_metrics = evaluate_pair(pred, gt)
            row.update({f"gt_{k}": v for k, v in gt_metrics.items()})
            hp_rows.append(row)
            print(
                f"HP {spec.name}: val={row['val_metric']:.6f} "
                f"gt_hungarian={row['gt_hungarian']:.1f}"
            )

        for method in ("pca", "lv", "gnn"):
            cand = [r for r in hp_rows if r["method"] == method]
            if not cand:
                continue
            best = max(cand, key=lambda r: r["val_metric"])
            hp = {"d": int(best["d"]), "lr": float(best["lr"])}
            if method == "gnn":
                hp["layers"] = int(best["layers"])
            best_by_method[method] = {
                "name": best["name"],
                "val_metric": best["val_metric"],
                "hyperparams": hp,
            }
            print(f"BEST {method} by val_metric: {best['name']} ({best['val_metric']:.6f})")

        Path("hp_results.json").write_text(json.dumps(hp_rows, indent=2))
        Path("hp_best.json").write_text(json.dumps(best_by_method, indent=2))
        with Path("hp_results.csv").open("w", newline="") as f:
            fields = sorted({k for r in hp_rows for k in r})
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(hp_rows)
    else:
        best_by_method = json.loads(Path("hp_best.json").read_text())

    final_rows: list[dict] = []
    if args.phase in {"all", "final"}:
        for spec in final_specs(args, best_by_method):
            metrics_path = Path(f"{spec.prefix}_metrics.json")
            if args.skip_existing and metrics_path.exists():
                print(f"Skipping existing final run {spec.name}")
            else:
                run_cmd(spec.command)
            metrics = read_metrics(spec.prefix)
            pred = load_assignment_dict(pred_path_for(spec.prefix))
            gt_metrics = evaluate_pair(pred, gt)
            row = {
                "phase": "final",
                "name": spec.name,
                "method": spec.method,
                **spec.hyperparams,
                "val_metric": metrics["val_metric"],
                **{f"gt_{k}": v for k, v in gt_metrics.items()},
            }
            final_rows.append(row)
            print(
                f"FINAL {spec.name}: val={row['val_metric']:.6f} "
                f"gt_hungarian={row['gt_hungarian']:.1f}"
            )

        Path("final_results.json").write_text(json.dumps(final_rows, indent=2))
        with Path("final_results.csv").open("w", newline="") as f:
            fields = sorted({k for r in final_rows for k in r})
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(final_rows)

        # Uncertainty summary
        summary = []
        for method in sorted({r["method"] for r in final_rows}):
            sub = [r for r in final_rows if r["method"] == method]
            hun = np.array([r["gt_hungarian"] for r in sub], dtype=float)
            ari = np.array([r["gt_ari"] for r in sub], dtype=float)
            nmi = np.array([r["gt_nmi"] for r in sub], dtype=float)
            summary.append(
                {
                    "method": method,
                    "n_seeds": len(sub),
                    "hungarian_mean": float(hun.mean()),
                    "hungarian_std": float(hun.std(ddof=1)) if len(sub) > 1 else 0.0,
                    "ari_mean": float(ari.mean()),
                    "ari_std": float(ari.std(ddof=1)) if len(sub) > 1 else 0.0,
                    "nmi_mean": float(nmi.mean()),
                    "nmi_std": float(nmi.std(ddof=1)) if len(sub) > 1 else 0.0,
                    "best_hyperparams": best_by_method[method]["hyperparams"],
                    "selection_val_metric": best_by_method[method]["val_metric"],
                }
            )
            print(
                f"SUMMARY {method}: Hungarian "
                f"{summary[-1]['hungarian_mean']:.1f} ± {summary[-1]['hungarian_std']:.1f}"
            )
        Path("final_summary.json").write_text(json.dumps(summary, indent=2))
        with Path("final_summary.csv").open("w", newline="") as f:
            # flatten hyperparams for CSV
            flat = []
            for s in summary:
                row = {k: v for k, v in s.items() if k != "best_hyperparams"}
                row.update({f"hp_{k}": v for k, v in s["best_hyperparams"].items()})
                flat.append(row)
            writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)


if __name__ == "__main__":
    main()
