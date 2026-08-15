"""Unsupervised HP search + multi-seed final evaluation.

Protocol
--------
1. Build fixed held-out splits (no ground-truth types).
2. For each method, sweep hyperparameters and select by held-out validation
   metric (Bernoulli LL for LV; negated row reconstruction MSE for PCA).
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
from itertools import product
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from evaluate_clustering import evaluate_pair, load_assignment_dict, load_ground_truth
from heldout import LIKELIHOODS, make_heldout_pairs, make_heldout_rows, save_heldout
from training_utils import (
    DEFAULT_U_SCALE_INIT,
    LABEL_SMOOTHING_TARGETS,
    U_NORMS,
    U_SCALES,
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    method: str
    command: list[str]
    prefix: str
    hyperparams: dict
    # Rows sharing a group are aggregated into one entry of ``final_summary``.
    # Defaults to the method name.
    group: str = ""

    def summary_group(self) -> str:
        return self.group or self.method


def smoothing_tag(eps: float) -> str:
    """Run-name suffix for label smoothing, empty at ``eps=0``.

    Unsmoothed runs keep the prefixes every existing artefact was written under,
    so ``--skip-existing`` and the accumulated result tables stay valid.
    """
    return f"_ls{eps}" if eps else ""


def smoothing_flags(eps: float, target: str) -> list[str]:
    return ["--label-smoothing", str(eps), "--label-smoothing-target", target]


# One sweep point for the block-logit constraint: the scale and its
# initialisation, which only exist under ``--u-norm unit``.
UNorm = tuple[str, str, float]


def u_norm_settings(norms: list[str], scales: list[str], inits: list[float]) -> list[UNorm]:
    """Deduplicated $(\\texttt{u\\_norm}, \\texttt{u\\_scale}, s_0)$ grid points.

    The scale is inert under ``--u-norm none``, so the product is collapsed
    there rather than running the identical unconstrained configuration once per
    scale parameterisation.
    """
    settings: list[UNorm] = []
    for norm in norms:
        candidates = (
            [(norm, scales[0], float(inits[0]))]
            if norm == "none"
            else [(norm, s, float(i)) for s in scales for i in inits]
        )
        settings.extend(c for c in candidates if c not in settings)
    return settings


def u_norm_tag(setting: UNorm) -> str:
    """Run-name suffix, empty for the unconstrained decoder.

    As with ``smoothing_tag``, the default arm keeps the prefixes every existing
    artefact was written under, so ``--skip-existing`` stays valid.
    """
    u_norm, u_scale, init = setting
    return "" if u_norm == "none" else f"_unit_{u_scale}{init}"


def u_norm_flags(setting: UNorm) -> list[str]:
    u_norm, u_scale, init = setting
    return ["--u-norm", u_norm, "--u-scale", u_scale, "--u-scale-init", str(init)]


def u_norm_hyperparams(setting: UNorm) -> dict:
    u_norm, u_scale, init = setting
    return {"u_norm": u_norm, "u_scale": u_scale, "u_scale_init": init}


def u_norm_of(hp: dict) -> UNorm:
    """Recover a sweep point from a selected hyperparameter record."""
    return (
        str(hp.get("u_norm", "none")),
        str(hp.get("u_scale", "fixed")),
        float(hp.get("u_scale_init", DEFAULT_U_SCALE_INIT)),
    )


def entropy_tag(beta: float) -> str:
    """Run-name suffix; empty at the standard ELBO weight so default names stay stable."""
    return "" if abs(float(beta) - 1.0) < 1e-12 else f"_eb{beta:g}"


def entropy_flags(beta: float) -> list[str]:
    return ["--entropy-beta", str(beta)]


def partner_kl_tag(weight: float) -> str:
    """Run-name suffix; empty at the default (disabled) weight."""
    return "" if abs(float(weight)) < 1e-12 else f"_pkl{weight:g}"


def partner_kl_flags(weight: float) -> list[str]:
    return ["--partner-kl-weight", str(weight)]


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
    methods = set(args.methods)

    if "pca" in methods:
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

    if "lv" in methods:
        # Flattened with ``product``: the grid has grown past the depth at which
        # nested ``for`` blocks leave room for the command list.
        lv_grid = product(
            args.lv_dims,
            args.lv_lrs,
            args.lv_bfs_fracs,
            args.lv_likelihoods,
            args.lv_label_smoothings,
            args.lv_entropy_betas,
            args.lv_partner_kl_weights,
            u_norm_settings(args.lv_u_norms, args.lv_u_scales, args.lv_u_scale_inits),
        )
        for d, lr, bfs_frac, likelihood, eps, entropy_beta, partner_kl, u_setting in lv_grid:
            name = (
                f"hp_lv_d{d}_lr{lr}_bfs{bfs_frac}_{likelihood}"
                f"{smoothing_tag(eps)}{u_norm_tag(u_setting)}"
                f"{entropy_tag(entropy_beta)}{partner_kl_tag(partner_kl)}"
            )
            specs.append(
                RunSpec(
                    name=name,
                    method="lv",
                    prefix=name,
                    hyperparams={
                        "d": d,
                        "lr": lr,
                        "bfs_frac": bfs_frac,
                        "likelihood": likelihood,
                        "label_smoothing": eps,
                        "entropy_beta": float(entropy_beta),
                        "partner_kl_weight": float(partner_kl),
                        **u_norm_hyperparams(u_setting),
                    },
                    command=[
                        py,
                        "train_lv_vsbm.py",
                        "--k",
                        str(args.k),
                        "--d",
                        str(d),
                        "--lr",
                        str(lr),
                        "--bfs-frac",
                        str(bfs_frac),
                        "--bfs-seeds",
                        str(args.bfs_seeds),
                        "--likelihood",
                        likelihood,
                        "--grad-clip",
                        str(args.grad_clip),
                        "--select-metric",
                        args.select_metric,
                        *smoothing_flags(eps, args.label_smoothing_target),
                        *u_norm_flags(u_setting),
                        *entropy_flags(entropy_beta),
                        *partner_kl_flags(partner_kl),
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
                        name,
                    ],
                )
            )

    if "lv" in methods and args.lv_control_updates:
        # A coverage-defined epoch is longer at higher bfs_frac, so the epoch-matched
        # grid above also hands the BFS arms more gradient steps. This control gives
        # uniform sampling the same update budget, separating the two effects.
        # One dim, one likelihood, one smoothing level and one block-logit
        # constraint suffice: the control isolates the sampler, not the rank, the
        # observation model, the target form or the decoder parameterisation, and
        # each control costs as much as a BFS run.
        u_setting = u_norm_settings(args.lv_u_norms, args.lv_u_scales, args.lv_u_scale_inits)[0]
        entropy_beta = float(args.lv_entropy_betas[0])
        partner_kl = float(args.lv_partner_kl_weights[0])
        for d in args.lv_dims[:1]:
            for lr in args.lv_lrs:
                for likelihood in args.lv_likelihoods[:1]:
                    eps = args.lv_label_smoothings[0]
                    name = (
                        f"hp_lv_control_d{d}_lr{lr}_{likelihood}"
                        f"{smoothing_tag(eps)}{u_norm_tag(u_setting)}"
                        f"{entropy_tag(entropy_beta)}{partner_kl_tag(partner_kl)}"
                    )
                    specs.append(
                        RunSpec(
                            name=name,
                            method="lv",
                            prefix=name,
                            hyperparams={
                                "d": d,
                                "lr": lr,
                                "bfs_frac": 0.0,
                                "likelihood": likelihood,
                                "label_smoothing": eps,
                                "entropy_beta": entropy_beta,
                                "partner_kl_weight": partner_kl,
                                "target_updates": args.lv_control_updates,
                                **u_norm_hyperparams(u_setting),
                            },
                            command=[
                                py,
                                "train_lv_vsbm.py",
                                "--k",
                                str(args.k),
                                "--d",
                                str(d),
                                "--lr",
                                str(lr),
                                "--bfs-frac",
                                "0.0",
                                "--bfs-seeds",
                                str(args.bfs_seeds),
                                "--likelihood",
                                likelihood,
                                "--grad-clip",
                                str(args.grad_clip),
                                "--select-metric",
                                args.select_metric,
                                *smoothing_flags(eps, args.label_smoothing_target),
                                *u_norm_flags(u_setting),
                                *entropy_flags(entropy_beta),
                                *partner_kl_flags(partner_kl),
                                "--target-updates",
                                str(args.lv_control_updates),
                                "--epochs",
                                str(10**9),
                                "--minibatch",
                                str(args.minibatch),
                                "--seed",
                                str(args.split_seed),
                                "--device",
                                args.device,
                                "--heldout-pairs",
                                args.heldout_pairs,
                                "--out-prefix",
                                name,
                            ],
                        )
                    )

    if "ntac" in methods:
        for max_k in args.ntac_max_ks:
            for max_iter in args.ntac_max_iters:
                for frac in args.ntac_frac_seeds:
                    name = f"hp_ntac_k{max_k}_R{max_iter}_T{frac}"
                    prefix = name
                    specs.append(
                        RunSpec(
                            name=name,
                            method="ntac",
                            prefix=prefix,
                            hyperparams={
                                "d": int(max_k),
                                "max_k": int(max_k),
                                "max_iterations": int(max_iter),
                                "frac_seeds": float(frac),
                                "lr": 0.0,
                            },
                            command=[
                                py,
                                "train_ntac.py",
                                "--max-k",
                                str(max_k),
                                "--max-iterations",
                                str(max_iter),
                                "--frac-seeds",
                                str(frac),
                                "--seed",
                                str(args.split_seed),
                                "--device",
                                args.device,
                                "--out-prefix",
                                prefix,
                            ],
                        )
                    )
    return specs


def final_variants(_args: argparse.Namespace, method: str, hp: dict) -> list[tuple[str, dict]]:
    """``(summary group, hyperparameters)`` for each final arm of one method."""
    return [(method, hp)]


def final_specs(args: argparse.Namespace, best_by_method: dict[str, dict]) -> list[RunSpec]:
    py = sys.executable
    specs: list[RunSpec] = []
    variants = [
        (method, group, hp)
        for method, best in best_by_method.items()
        for group, hp in final_variants(args, method, best["hyperparams"])
    ]
    for method, group, hp in variants:
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
                    str(args.final_pca_max_iter),
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
                control = hp.get("target_updates")
                tag = "control" if control else f"bfs{hp['bfs_frac']}"
                eps = float(hp.get("label_smoothing", 0.0))
                entropy_beta = float(hp.get("entropy_beta", 1.0))
                partner_kl = float(hp.get("partner_kl_weight", 0.0))
                u_setting = u_norm_of(hp)
                name = (
                    f"final_lv_d{hp['d']}_lr{hp['lr']}_{tag}_{hp['likelihood']}"
                    f"{smoothing_tag(eps)}{u_norm_tag(u_setting)}"
                    f"{entropy_tag(entropy_beta)}{partner_kl_tag(partner_kl)}_seed{seed}"
                )
                cmd = [
                    py,
                    "train_lv_vsbm.py",
                    "--k",
                    str(args.k),
                    "--d",
                    str(hp["d"]),
                    "--lr",
                    str(hp["lr"]),
                    "--bfs-frac",
                    str(hp["bfs_frac"]),
                    "--bfs-seeds",
                    str(args.bfs_seeds),
                    "--likelihood",
                    str(hp["likelihood"]),
                    "--grad-clip",
                    str(args.grad_clip),
                    "--select-metric",
                    args.select_metric,
                    *smoothing_flags(eps, args.label_smoothing_target),
                    *u_norm_flags(u_setting),
                    *entropy_flags(entropy_beta),
                    *partner_kl_flags(partner_kl),
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
                if control:
                    # Scale the control budget with the finals epoch budget.
                    scaled = int(round(control * args.final_epochs / max(args.epochs, 1)))
                    cmd += ["--target-updates", str(scaled), "--epochs", str(10**9)]
                else:
                    cmd += ["--epochs", str(args.final_epochs)]
            elif method == "ntac":
                name = (
                    f"final_ntac_k{hp['max_k']}_R{hp['max_iterations']}_"
                    f"T{hp['frac_seeds']}_seed{seed}"
                )
                cmd = [
                    py,
                    "train_ntac.py",
                    "--max-k",
                    str(hp["max_k"]),
                    "--max-iterations",
                    str(hp["max_iterations"]),
                    "--frac-seeds",
                    str(hp["frac_seeds"]),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--out-prefix",
                    name,
                ]
            else:
                raise ValueError(f"Unknown method for finals: {method}")
            specs.append(
                RunSpec(
                    name=name,
                    method=method,
                    prefix=name,
                    hyperparams={**hp, "seed": seed},
                    command=cmd,
                    group=group,
                )
            )
    return specs


def read_metrics(prefix: str) -> dict:
    path = Path(f"{prefix}_metrics.json")
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


# Reported alongside the selection metric so a run that diverged mid-training is
# visible in the tables rather than only in its log.
DIAGNOSTIC_KEYS = (
    "val_auc",
    "last_val_metric",
    "last_val_auc",
    "checkpoint",
    "best_epoch",
    "last_epoch",
    "grad_clip",
    "label_smoothing",
    "label_smoothing_target",
    "entropy_beta",
    "partner_kl_weight",
    "label_smoothing_applied",
    "u_scale_value",
    "u_scale_value_max",
    "last_u_scale_value",
    "decoder_bias",
    "last_decoder_bias",
    "max_abs_train_logit",
    "skipped_updates",
)


def diagnostics(prefix: str, metrics: dict, gt: dict) -> dict:
    """Best-vs-last diagnostics for one run, including ground truth at both states.

    The trainers already score both states; this recomputes ``last_gt_*`` from the
    saved last-epoch assignment dict only when they were run with GT scoring off.
    """
    row = {k: metrics[k] for k in DIAGNOSTIC_KEYS if k in metrics}
    last_gt = {k: v for k, v in metrics.items() if k.startswith("last_gt_")}
    if not last_gt:
        last_path = Path(f"{prefix}_last_assignment_dict.npy")
        if last_path.exists():
            scored = evaluate_pair(load_assignment_dict(last_path), gt)
            last_gt = {f"last_gt_{k}": v for k, v in scored.items()}
    row.update(last_gt)
    return row


def best_vs_last(row: dict) -> str:
    """One-line summary of what the best-val checkpoint bought over the last epoch."""
    if "last_gt_hungarian" not in row:
        return ""
    return (
        f" | last: val={row.get('last_val_metric', float('nan')):.6f} "
        f"auc={row.get('last_val_auc', float('nan')):.4f} "
        f"gt_hungarian={row['last_gt_hungarian']:.1f} "
        f"(best epoch {row.get('best_epoch')})"
    )


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
    p.add_argument("--epochs", type=int, default=5, help="HP-search epoch budget for LV")
    p.add_argument(
        "--final-epochs",
        type=int,
        default=20,
        help="Full epoch budget for multi-seed final LV runs",
    )
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--pca-max-iter", type=int, default=2_000, help="HP-search PCA SGD steps")
    p.add_argument(
        "--final-pca-max-iter",
        type=int,
        default=10_000,
        help="Full PCA SGD steps for multi-seed finals",
    )
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--heldout-pairs", default="heldout_pairs.npz")
    p.add_argument("--heldout-rows", default="heldout_rows.npz")
    p.add_argument("--n-heldout-pos", type=int, default=50_000)
    p.add_argument("--n-heldout-neg", type=int, default=50_000)
    p.add_argument("--row-holdout", type=float, default=0.1)
    p.add_argument("--pca-dims", type=int, nargs="+", default=[32, 64])
    p.add_argument("--pca-lrs", type=float, nargs="+", default=[0.01])
    p.add_argument("--lv-dims", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--lv-lrs", type=float, nargs="+", default=[0.1])
    p.add_argument("--lv-bfs-fracs", type=float, nargs="+", default=[1.0])
    p.add_argument(
        "--lv-likelihoods", nargs="+", choices=LIKELIHOODS, default=["bernoulli", "poisson", "nb"]
    )
    p.add_argument("--bfs-seeds", type=int, default=4, help="BFS seeds per expansion round for LV")
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help=(
            "Global gradient-norm clip for LV / LV+e / GNN; 0 or less disables it. "
            "Clipping alone does not prevent the lr=0.1 decoder divergence, so pair "
            "it with a smaller --lv-lrs"
        ),
    )
    p.add_argument(
        "--select-metric",
        choices=("ll", "auc"),
        default="ll",
        help=(
            "Held-out metric used to rank LV configurations and pick each run's "
            "checkpoint. The held-out pairs are class-balanced while the graph is "
            "sparse, so a correctly calibrated model scores poorly under 'll'; use "
            "'auc' when log-likelihood peaks at initialization"
        ),
    )
    p.add_argument(
        "--lv-label-smoothings",
        type=float,
        nargs="+",
        default=[0.0],
        help=(
            "LV Bernoulli label-smoothing grid; 0.0 (default) keeps hard 0/1 targets "
            "and the historical run names"
        ),
    )
    p.add_argument(
        "--lv-u-norms",
        nargs="+",
        choices=U_NORMS,
        default=["none"],
        help=(
            "LV block-embedding constraint grid; 'none' (default) keeps the "
            "unbounded bilinear decoder and the historical run names"
        ),
    )
    p.add_argument(
        "--lv-u-scales",
        nargs="+",
        choices=U_SCALES,
        default=["fixed"],
        help="LV scale parameterisation grid, swept only under u_norm=unit",
    )
    p.add_argument(
        "--lv-u-scale-inits",
        type=float,
        nargs="+",
        default=[DEFAULT_U_SCALE_INIT],
        help="LV scale value / initialisation grid, swept only under u_norm=unit",
    )
    p.add_argument(
        "--lv-entropy-betas",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "LV entropy-weight grid on sum_i H(q_i); 1.0 is the uniform-prior ELBO. "
            "Non-unit values append _eb{beta} to run names"
        ),
    )
    p.add_argument(
        "--lv-partner-kl-weights",
        type=float,
        nargs="+",
        default=[0.0],
        help=(
            "LV partner-histogram KL weight grid; 0 disables. Nonzero values append "
            "_pkl{weight} to run names. hp_best must propagate partner_kl_weight"
        ),
    )
    p.add_argument(
        "--label-smoothing-target",
        choices=LABEL_SMOOTHING_TARGETS,
        default="base_rate",
        help=(
            "Prior the smoothed targets are pulled towards, shared by every method: "
            "'base_rate' preserves the edge marginal on a graph of density ~1e-4; "
            "'uniform' (0.5) does not"
        ),
    )
    p.add_argument(
        "--lv-control-updates",
        type=int,
        default=0,
        help=(
            "If set, add LV runs at bfs_frac=0 with this total update budget, "
            "matching the compute of the BFS arms (0 disables the control)"
        ),
    )
    p.add_argument("--ntac-max-ks", type=int, nargs="+", default=[729])
    p.add_argument("--ntac-max-iters", type=int, nargs="+", default=[12])
    p.add_argument("--ntac-frac-seeds", type=float, nargs="+", default=[0.1])
    p.add_argument("--final-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--methods",
        nargs="+",
        default=["pca", "lv", "ntac"],
        choices=["pca", "lv", "ntac"],
        help="Which methods to include in HP search / finals",
    )
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
                "val_native_ll": metrics.get("val_native_ll"),
                "val_native_ll_name": metrics.get("val_native_ll_name"),
                "n_pred_clusters": metrics.get("n_pred_clusters"),
            }
            # Optional GT score for analysis only — NOT used for selection.
            pred = load_assignment_dict(pred_path_for(spec.prefix))
            gt_metrics = evaluate_pair(pred, gt)
            row.update({f"gt_{k}": v for k, v in gt_metrics.items()})
            row.update(diagnostics(spec.prefix, metrics, gt))
            hp_rows.append(row)
            print(
                f"HP {spec.name}: val={row['val_metric']:.6f} "
                f"gt_hungarian={row['gt_hungarian']:.1f}" + best_vs_last(row)
            )

        for method in args.methods:
            cand = [r for r in hp_rows if r["method"] == method]
            if not cand:
                continue
            best = max(cand, key=lambda r: r["val_metric"])
            hp = {"d": int(best["d"]), "lr": float(best["lr"])}
            if method == "lv":
                hp["bfs_frac"] = float(best["bfs_frac"])
                hp["likelihood"] = str(best["likelihood"])
                hp["label_smoothing"] = float(best.get("label_smoothing", 0.0))
                hp["entropy_beta"] = float(best.get("entropy_beta", 1.0))
                hp["partner_kl_weight"] = float(best.get("partner_kl_weight", 0.0))
                hp.update(u_norm_hyperparams(u_norm_of(best)))
                if best.get("target_updates"):
                    # Carry the matched-compute budget so a selected control stays a
                    # control in the finals instead of reverting to epoch matching.
                    hp["target_updates"] = int(best["target_updates"])
            if method == "ntac":
                hp = {
                    "d": int(best["d"]),
                    "lr": float(best.get("lr", 0.0)),
                    "max_k": int(best["max_k"]),
                    "max_iterations": int(best["max_iterations"]),
                    "frac_seeds": float(best["frac_seeds"]),
                }
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
                "group": spec.summary_group(),
                **spec.hyperparams,
                "val_metric": metrics["val_metric"],
                "val_native_ll": metrics.get("val_native_ll"),
                "val_native_ll_name": metrics.get("val_native_ll_name"),
                **{f"gt_{k}": v for k, v in gt_metrics.items()},
                **diagnostics(spec.prefix, metrics, gt),
            }
            final_rows.append(row)
            print(
                f"FINAL {spec.name}: val={row['val_metric']:.6f} "
                f"gt_hungarian={row['gt_hungarian']:.1f}" + best_vs_last(row)
            )

        Path("final_results.json").write_text(json.dumps(final_rows, indent=2))
        with Path("final_results.csv").open("w", newline="") as f:
            fields = sorted({k for r in final_rows for k in r})
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(final_rows)

        # Uncertainty summary, one entry per group (one group per method).
        summary = []
        variant_hp = {
            group: hp
            for method, best in best_by_method.items()
            for group, hp in final_variants(args, method, best["hyperparams"])
        }
        for group in sorted({r.get("group", r["method"]) for r in final_rows}):
            sub = [r for r in final_rows if r.get("group", r["method"]) == group]
            method = sub[0]["method"]
            selected = variant_hp.get(group) == best_by_method[method]["hyperparams"]
            hun = np.array([r["gt_hungarian"] for r in sub], dtype=float)
            ari = np.array([r["gt_ari"] for r in sub], dtype=float)
            nmi = np.array([r["gt_nmi"] for r in sub], dtype=float)
            entry = {
                "group": group,
                "method": method,
                "n_seeds": len(sub),
                "hungarian_mean": float(hun.mean()),
                "hungarian_std": float(hun.std(ddof=1)) if len(sub) > 1 else 0.0,
                "ari_mean": float(ari.mean()),
                "ari_std": float(ari.std(ddof=1)) if len(sub) > 1 else 0.0,
                "nmi_mean": float(nmi.mean()),
                "nmi_std": float(nmi.std(ddof=1)) if len(sub) > 1 else 0.0,
                "best_hyperparams": variant_hp.get(group, best_by_method[method]["hyperparams"]),
                "selected_by_val": selected,
                # Only meaningful for the arm the HP search selected; a comparison
                # depth was never in the running for this number.
                "selection_val_metric": best_by_method[method]["val_metric"] if selected else None,
            }
            # Same seeds, same budget: the controlled answer to whether restoring the
            # best-validation checkpoint costs or buys ground-truth agreement.
            last_hun = [r["last_gt_hungarian"] for r in sub if "last_gt_hungarian" in r]
            if last_hun:
                arr = np.array(last_hun, dtype=float)
                entry["last_hungarian_mean"] = float(arr.mean())
                entry["last_hungarian_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                entry["hungarian_best_minus_last"] = entry["hungarian_mean"] - float(arr.mean())
            summary.append(entry)
            print(
                f"SUMMARY {group}: Hungarian "
                f"{summary[-1]['hungarian_mean']:.1f} ± {summary[-1]['hungarian_std']:.1f}"
                f"{'' if selected else ' (comparison arm, not selected)'}"
            )
        Path("final_summary.json").write_text(json.dumps(summary, indent=2))
        with Path("final_summary.csv").open("w", newline="") as f:
            # flatten hyperparams for CSV (union of keys across methods)
            flat = []
            for s in summary:
                row = {k: v for k, v in s.items() if k != "best_hyperparams"}
                row.update({f"hp_{k}": v for k, v in s["best_hyperparams"].items()})
                flat.append(row)
            fields = sorted({k for row in flat for k in row})
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat)


if __name__ == "__main__":
    main()
