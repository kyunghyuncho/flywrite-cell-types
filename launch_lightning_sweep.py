"""Launch unsupervised HP search + multi-seed finals on Lightning AI (L4/T4).

Detaches the long sweep on the Studio (nohup + log file) and polls progress
locally, so the client is not blocked on a multi-hour ``run_with_exit_code``.

Example:

    source ~/.ortet/lightning.env
    export LIGHTNING_USERNAME=kc119 LIGHTNING_TEAMSPACE=vision-model
    uv run python launch_lightning_sweep.py --machine L4 --stop-after
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from lightning_sdk import Machine, Studio

from heldout import LIKELIHOODS

REPO_ROOT = Path(__file__).resolve().parent
REMOTE_LOG = "unsup_sweep.log"
REMOTE_DONE = "unsup_sweep.done"
REMOTE_PID = "unsup_sweep.pid"

CODE_FILES = [
    "gnn_vsbm.py",
    "gnn_e_vsbm.py",
    "train_lv_vsbm.py",
    "train_lv_e.py",
    "train_ntac.py",
    "train_pca_baseline.py",
    "sparse_graph_pca.py",
    "heldout.py",
    "subgraph_sampler.py",
    "run_experiments.py",
    "evaluate_clustering.py",
    "index_mapping.py",
    "hidden_markov_graph.py",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "launch_lightning_sweep.py",
    "remote_start_unsup_sweep.sh",
    "remote_status_unsup_sweep.sh",
    "remote_stop_studio.py",
]

DATA_FILES = [
    "sparse_connectivity_matrix.npz",
    "root_id_to_index_mapping.json",
    "root_id_type_dict.pkl",
]

SUMMARY_ARTIFACTS = [
    "heldout_pairs.npz",
    "heldout_rows.npz",
    "hp_results.json",
    "hp_results.csv",
    "hp_best.json",
    "final_results.json",
    "final_results.csv",
    "final_summary.json",
    "final_summary.csv",
    REMOTE_LOG,
]


def require_auth() -> None:
    if not os.environ.get("LIGHTNING_USER_ID") or not os.environ.get("LIGHTNING_API_KEY"):
        print("Missing LIGHTNING_USER_ID / LIGHTNING_API_KEY", file=sys.stderr)
        sys.exit(1)


def resolve_machine(name: str) -> Machine:
    key = name.upper().replace("-", "_")
    if not hasattr(Machine, key):
        raise SystemExit(f"Unknown machine {name!r}")
    return getattr(Machine, key)


def studio_run(studio: Studio, cmd: str) -> tuple[str, int]:
    out, code = studio.run_with_exit_code(cmd)
    return out or "", int(code)


def download_artifacts(studio: Studio) -> None:
    print("Downloading summary artifacts...")
    for name in SUMMARY_ARTIFACTS:
        try:
            studio.download_file(name, str(REPO_ROOT / name))
            print(f"  <- {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name}: {exc}")

    out, _ = studio_run(
        studio,
        "cd /teamspace/studios/this_studio && "
        "python -c \"import glob; print('\\\\n'.join(sorted("
        "glob.glob('final_*_assignment_dict*.npy')+"
        "glob.glob('hp_*_assignment_dict*.npy')+"
        "glob.glob('*_metrics.json'))))\"",
    )
    for line in out.splitlines():
        name = line.strip()
        if not name:
            continue
        try:
            studio.download_file(name, str(REPO_ROOT / name))
            print(f"  <- {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name}: {exc}")

    summary_path = REPO_ROOT / "final_summary.json"
    if summary_path.exists():
        print("Final uncertainty summary:")
        print(json.dumps(json.loads(summary_path.read_text()), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio-name", default="flywrite-gnn-vsbm")
    parser.add_argument("--machine", default="L4")
    parser.add_argument("--epochs", type=int, default=15, help="HP-search LV/GNN epochs")
    parser.add_argument("--final-epochs", type=int, default=40, help="Final LV/GNN epochs")
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--pca-max-iter", type=int, default=2_000, help="HP-search PCA steps")
    parser.add_argument("--final-pca-max-iter", type=int, default=10_000, help="Final PCA steps")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--pca-lrs", type=float, nargs="+", default=[0.01])
    parser.add_argument("--lv-dims", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--lv-lrs", type=float, nargs="+", default=[0.1])
    parser.add_argument("--lv-bfs-fracs", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--lv-likelihoods", nargs="+", choices=LIKELIHOODS, default=["bernoulli", "poisson", "nb"]
    )
    parser.add_argument("--lv-e-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--lv-e-d-es", type=int, nargs="+", default=[16])
    parser.add_argument("--lv-e-lrs", type=float, nargs="+", default=[0.05])
    parser.add_argument("--lv-e-wds", type=float, nargs="+", default=[1e-2])
    parser.add_argument("--lv-e-bfs-fracs", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--lv-e-likelihoods", nargs="+", choices=LIKELIHOODS, default=["bernoulli", "poisson"]
    )
    parser.add_argument("--bfs-seeds", type=int, default=4)
    parser.add_argument(
        "--lv-control-updates",
        type=int,
        default=0,
        help="Total update budget for a matched-compute bfs_frac=0 LV control (0 disables)",
    )
    parser.add_argument("--gnn-layers", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--gnn-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--gnn-lrs", type=float, nargs="+", default=[0.005, 0.01])
    parser.add_argument("--gnn-e-layers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gnn-e-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--gnn-e-d-es", type=int, nargs="+", default=[16])
    parser.add_argument("--gnn-e-lrs", type=float, nargs="+", default=[0.005, 0.01])
    parser.add_argument("--gnn-e-wds", type=float, nargs="+", default=[1e-2])
    parser.add_argument("--ntac-max-ks", type=int, nargs="+", default=[729])
    parser.add_argument("--ntac-max-iters", type=int, nargs="+", default=[12])
    parser.add_argument("--ntac-frac-seeds", type=float, nargs="+", default=[0.1])
    parser.add_argument("--final-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["lv", "lv_e"],
        choices=["pca", "lv", "lv_e", "gnn", "gnn_e", "ntac"],
        help="Methods to sweep",
    )
    parser.add_argument("--phase", choices=["all", "hp", "final"], default="all")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--stop-after",
        action="store_true",
        help="Stop Studio from this client after a polling run finishes.",
    )
    parser.add_argument(
        "--remote-stop-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop Studio from inside the remote job when it finishes (safe if laptop closes).",
    )
    parser.add_argument(
        "--keep-auto-sleep",
        action="store_true",
        help=(
            "Leave Studio idle auto-sleep enabled. The default disables it: a detached "
            "nohup sweep does not reliably register as activity, and auto-sleep has "
            "already killed one multi-hour run mid-way. The Studio is still stopped at "
            "the end by --remote-stop-after."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument(
        "--detach-only",
        action="store_true",
        help="Start the remote job and exit without polling/downloading.",
    )
    args = parser.parse_args()
    require_auth()

    user = os.environ.get("LIGHTNING_USERNAME", "kc119")
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "vision-model")
    machine = resolve_machine(args.machine)

    print(f"Starting Studio {args.studio_name!r} on {args.machine}...")
    studio = Studio(
        name=args.studio_name,
        teamspace=teamspace,
        user=user,
        create_ok=True,
    )
    if not args.keep_auto_sleep:
        print(f"auto_sleep={studio.auto_sleep} (idle_timeout={studio.auto_sleep_time}s); disabling")
        studio.auto_sleep = False
        if studio.auto_sleep:
            raise SystemExit("Could not disable auto-sleep; refusing to start a long sweep")
    studio.start(machine)
    print(f"Studio ready on {studio.machine} (auto_sleep={studio.auto_sleep})")

    try:
        if not args.skip_upload:
            print("Uploading code and data...")
            for name in CODE_FILES + DATA_FILES:
                local = (REPO_ROOT / name).resolve()
                if not local.exists():
                    raise SystemExit(f"Missing {local}")
                print(f"  -> {name}")
                studio.upload_file(str(local), name)

        out, code = studio_run(
            studio,
            "pip install -q torch scipy numpy scikit-learn tqdm pandas lightning-sdk "
            "ntac numba bottleneck",
        )
        if code != 0:
            raise RuntimeError(f"pip failed: {out}")

        def join_nums(vals: list) -> str:
            return " ".join(str(x) for x in vals)

        methods = " ".join(args.methods)
        sweep_args = (
            f"--device cuda --phase {args.phase} "
            f"--methods {methods} "
            f"--split-seed {args.split_seed} "
            f"--epochs {args.epochs} --final-epochs {args.final_epochs} "
            f"--minibatch {args.minibatch} "
            f"--pca-max-iter {args.pca_max_iter} "
            f"--final-pca-max-iter {args.final_pca_max_iter} "
            f"--pca-dims {join_nums(args.pca_dims)} --pca-lrs {join_nums(args.pca_lrs)} "
            f"--lv-dims {join_nums(args.lv_dims)} --lv-lrs {join_nums(args.lv_lrs)} "
            f"--lv-bfs-fracs {join_nums(args.lv_bfs_fracs)} "
            f"--lv-likelihoods {join_nums(args.lv_likelihoods)} "
            f"--lv-e-dims {join_nums(args.lv_e_dims)} "
            f"--lv-e-d-es {join_nums(args.lv_e_d_es)} "
            f"--lv-e-lrs {join_nums(args.lv_e_lrs)} "
            f"--lv-e-wds {join_nums(args.lv_e_wds)} "
            f"--lv-e-bfs-fracs {join_nums(args.lv_e_bfs_fracs)} "
            f"--lv-e-likelihoods {join_nums(args.lv_e_likelihoods)} "
            f"--bfs-seeds {args.bfs_seeds} "
            f"--lv-control-updates {args.lv_control_updates} "
            f"--gnn-layers {join_nums(args.gnn_layers)} --gnn-dims {join_nums(args.gnn_dims)} "
            f"--gnn-lrs {join_nums(args.gnn_lrs)} "
            f"--gnn-e-layers {join_nums(args.gnn_e_layers)} "
            f"--gnn-e-dims {join_nums(args.gnn_e_dims)} "
            f"--gnn-e-d-es {join_nums(args.gnn_e_d_es)} "
            f"--gnn-e-lrs {join_nums(args.gnn_e_lrs)} "
            f"--gnn-e-wds {join_nums(args.gnn_e_wds)} "
            f"--ntac-max-ks {join_nums(args.ntac_max_ks)} "
            f"--ntac-max-iters {join_nums(args.ntac_max_iters)} "
            f"--ntac-frac-seeds {join_nums(args.ntac_frac_seeds)} "
            f"--final-seeds {join_nums(args.final_seeds)}"
        )
        if args.skip_existing:
            sweep_args += " --skip-existing"

        # Export auth into the Studio shell env for remote auto-stop (not written to disk).
        auth_exports = (
            f"export LIGHTNING_USER_ID={os.environ['LIGHTNING_USER_ID']!r} "
            f"LIGHTNING_API_KEY={os.environ['LIGHTNING_API_KEY']!r} "
            f"LIGHTNING_USERNAME={user!r} "
            f"LIGHTNING_TEAMSPACE={teamspace!r} "
            f"STUDIO_NAME={args.studio_name!r}; "
        )

        # Kill any previous detached sweep, then start a fresh nohup job.
        print(f"Detaching unsupervised protocol:\n  {sweep_args}")
        print(f"remote_stop_after={args.remote_stop_after}")
        start_cmd = (
            auth_exports + "chmod +x /teamspace/studios/this_studio/remote_start_unsup_sweep.sh "
            "/teamspace/studios/this_studio/remote_status_unsup_sweep.sh "
            "/teamspace/studios/this_studio/remote_stop_studio.py && "
            f"SWEEP_ARGS={sweep_args!r} "
            f"REMOTE_STOP_AFTER={'1' if args.remote_stop_after else '0'} "
            "bash /teamspace/studios/this_studio/remote_start_unsup_sweep.sh"
        )
        out, code = studio_run(studio, start_cmd)
        print(out)
        if code != 0:
            raise RuntimeError(f"Failed to detach sweep: {code}")

        if args.detach_only:
            print("Detached only; not polling. Tail remote log later with the Studio shell.")
            return

        print(f"Polling every {args.poll_seconds}s ...")
        t0 = time.time()
        last_tail = ""
        while True:
            out, _ = studio_run(
                studio, "bash /teamspace/studios/this_studio/remote_status_unsup_sweep.sh"
            )
            if out.strip() and out.strip() != last_tail:
                print(out)
                last_tail = out.strip()
            else:
                print(f"... still running ({time.time() - t0:.0f}s)")

            alive = "alive=1" in out
            done = "done=1" in out
            if done or not alive:
                break
            time.sleep(args.poll_seconds)

        # Read exit code written by the wrapper.
        done_out, _ = studio_run(
            studio,
            f"cd /teamspace/studios/this_studio && cat {REMOTE_DONE} 2>/dev/null || echo missing",
        )
        exit_code = done_out.strip().splitlines()[-1] if done_out.strip() else "missing"
        print(f"Remote sweep finished with exit code {exit_code} after {time.time() - t0:.1f}s")
        if exit_code not in {"0"}:
            raise RuntimeError(f"Sweep failed with exit code {exit_code}")

        download_artifacts(studio)
    finally:
        if args.stop_after and not args.detach_only:
            print("Stopping Studio...")
            studio.stop()


if __name__ == "__main__":
    main()
