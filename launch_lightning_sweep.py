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
import shlex
import sys
import tarfile
import time
from pathlib import Path

from lightning_sdk import Machine, Studio

from heldout import LIKELIHOODS
from run_experiments import resolve_graph_paths
from training_utils import (
    DEFAULT_U_SCALE_INIT,
    LABEL_SMOOTHING_TARGETS,
    U_NORMS,
    U_SCALES,
)

REPO_ROOT = Path(__file__).resolve().parent
REMOTE_LOG = "unsup_sweep.log"
REMOTE_DONE = "unsup_sweep.done"
REMOTE_PID = "unsup_sweep.pid"

CODE_FILES = [
    "train_lv_vsbm.py",
    "train_ntac.py",
    "train_pca_baseline.py",
    "sparse_graph_pca.py",
    "heldout.py",
    "subgraph_sampler.py",
    "training_utils.py",
    "partner_consistency.py",
    "run_experiments.py",
    "evaluate_clustering.py",
    "index_mapping.py",
    "hidden_markov_graph.py",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "launch_lightning_sweep.py",
    "export_visual_subgraph.py",
    "remote_start_unsup_sweep.sh",
    "remote_status_unsup_sweep.sh",
    "remote_stop_studio.py",
]

STATIC_DATA_FILES = ["root_id_type_dict.pkl"]

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


REMOTE_BUNDLE = "sweep_tables.tar.gz"


def download_tables(studio: Studio, dest: Path) -> None:
    """Fetch the tables, per-run metrics and log as one small archive.

    Deliberately excludes the ``*.npy`` assignment dicts and score matrices. A
    98 MB archive containing them failed to download while a 1.1 MB JSON/CSV/log
    archive succeeded first time, and everything needed to merge and interpret a
    sweep is in the small payload: the tables already carry the ground-truth
    scores that ``run_experiments.py`` computed from those arrays.

    Writing into ``dest`` rather than the repository root is what keeps the
    accumulated canonical tables intact until ``merge_sweep_results.py`` runs.
    """
    dest.mkdir(parents=True, exist_ok=True)
    # ``find`` rather than a glob list: the Studio shell is zsh, where a single
    # unmatched pattern aborts the whole command, and a partial sweep legitimately
    # has no finals and no tables yet.
    patterns = (
        "hp_*_metrics.json",
        "final_*_metrics.json",
        "hp_results.*",
        "hp_best.json",
        "final_results.*",
        "final_summary.*",
        REMOTE_LOG,
    )
    predicate = " -o ".join(f"-name '{p}'" for p in patterns)
    build = (
        "cd /teamspace/studios/this_studio && "
        f"rm -f {REMOTE_BUNDLE} && "
        f"find . -maxdepth 1 \\( {predicate} \\) -print0 | "
        f"tar czf {REMOTE_BUNDLE} --null -T - && "
        f"ls -l {REMOTE_BUNDLE}"
    )
    out, code = studio_run(studio, build)
    print(out)
    if code != 0:
        raise RuntimeError(f"Failed to build {REMOTE_BUNDLE} on the Studio")

    local_bundle = dest / REMOTE_BUNDLE
    studio.download_file(REMOTE_BUNDLE, str(local_bundle))
    with tarfile.open(local_bundle) as tar:
        tar.extractall(dest, filter="data")
    local_bundle.unlink()
    files = sorted(p.name for p in dest.iterdir())
    print(f"Extracted {len(files)} file(s) into {dest}")


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
    parser.add_argument("--epochs", type=int, default=15, help="HP-search LV epochs")
    parser.add_argument("--final-epochs", type=int, default=40, help="Final LV epochs")
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
    parser.add_argument("--bfs-seeds", type=int, default=4)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Global gradient-norm clip for LV; 0 or less disables it",
    )
    parser.add_argument(
        "--select-metric",
        choices=("ll", "auc"),
        default="ll",
        help="Held-out metric used to rank LV configurations and pick each checkpoint",
    )
    parser.add_argument(
        "--lv-label-smoothings",
        type=float,
        nargs="+",
        default=[0.0],
        help="LV Bernoulli label-smoothing grid; 0.0 keeps hard 0/1 targets",
    )
    parser.add_argument(
        "--lv-u-norms",
        nargs="+",
        choices=U_NORMS,
        default=["none"],
        help="Block-embedding constraint grid; 'none' keeps the unbounded decoder",
    )
    parser.add_argument("--lv-u-scales", nargs="+", choices=U_SCALES, default=["fixed"])
    parser.add_argument("--lv-u-scale-inits", type=float, nargs="+", default=[DEFAULT_U_SCALE_INIT])
    parser.add_argument(
        "--lv-entropy-betas",
        type=float,
        nargs="+",
        default=[1.0],
        help="LV entropy-weight grid on sum_i H(q_i); 1.0 is the uniform-prior ELBO",
    )
    parser.add_argument(
        "--lv-partner-kl-weights",
        type=float,
        nargs="+",
        default=[0.0],
        help="LV partner-histogram KL weight grid; 0 disables the term",
    )
    parser.add_argument(
        "--label-smoothing-target",
        choices=LABEL_SMOOTHING_TARGETS,
        default="base_rate",
        help="Prior the smoothed targets are pulled towards; 'base_rate' preserves the marginal",
    )
    parser.add_argument(
        "--lv-control-updates",
        type=int,
        default=0,
        help="Total update budget for a matched-compute bfs_frac=0 LV control (0 disables)",
    )
    parser.add_argument("--ntac-max-ks", type=int, nargs="+", default=[729])
    parser.add_argument("--ntac-max-iters", type=int, nargs="+", default=[12])
    parser.add_argument("--ntac-frac-seeds", type=float, nargs="+", default=[0.1])
    parser.add_argument("--final-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pca", "lv", "ntac"],
        choices=["pca", "lv", "ntac"],
        help="Methods to sweep",
    )
    parser.add_argument("--phase", choices=["all", "hp", "final"], default="all")
    parser.add_argument("--graph-scope", choices=("full", "visual"), default="full")
    parser.add_argument("--adjacency", help="Override the graph-scope adjacency path")
    parser.add_argument("--mapping", help="Override the graph-scope root-ID mapping path")
    parser.add_argument("--heldout-pairs", help="Override the graph-scope pair split path")
    parser.add_argument("--heldout-rows", help="Override the graph-scope row split path")
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
    parser.add_argument(
        "--download-only",
        metavar="DIR",
        help=(
            "Skip upload and launch: fetch the current tables, per-run metrics "
            "and log from the Studio into DIR, then exit. Merge them with "
            "merge_sweep_results.py; nothing in the repository root is touched."
        ),
    )
    args = parser.parse_args()
    if not args.download_only:
        resolve_graph_paths(args, require_inputs=not args.skip_upload)
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

    if args.download_only:
        try:
            download_tables(studio, Path(args.download_only))
        finally:
            if args.stop_after:
                print("Stopping Studio...")
                studio.stop()
        return

    try:
        remote_graph_paths = {
            name: Path(getattr(args, name)).name
            for name in ("adjacency", "mapping", "heldout_pairs", "heldout_rows")
        }
        if not args.skip_upload:
            print("Uploading code and data...")
            uploads = [((REPO_ROOT / name).resolve(), name) for name in CODE_FILES]
            uploads.extend(((REPO_ROOT / name).resolve(), name) for name in STATIC_DATA_FILES)
            uploads.extend(
                [
                    (Path(args.adjacency).resolve(), remote_graph_paths["adjacency"]),
                    (Path(args.mapping).resolve(), remote_graph_paths["mapping"]),
                ]
            )
            for name in ("heldout_pairs", "heldout_rows"):
                local = Path(getattr(args, name)).resolve()
                if local.exists():
                    uploads.append((local, remote_graph_paths[name]))
            for local, remote in uploads:
                if not local.exists():
                    raise SystemExit(f"Missing {local}")
                print(f"  -> {remote}")
                studio.upload_file(str(local), remote)

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
            f"--graph-scope {shlex.quote(args.graph_scope)} "
            f"--adjacency {shlex.quote(remote_graph_paths['adjacency'])} "
            f"--mapping {shlex.quote(remote_graph_paths['mapping'])} "
            f"--heldout-pairs {shlex.quote(remote_graph_paths['heldout_pairs'])} "
            f"--heldout-rows {shlex.quote(remote_graph_paths['heldout_rows'])} "
            f"--split-seed {args.split_seed} "
            f"--epochs {args.epochs} --final-epochs {args.final_epochs} "
            f"--minibatch {args.minibatch} "
            f"--pca-max-iter {args.pca_max_iter} "
            f"--final-pca-max-iter {args.final_pca_max_iter} "
            f"--pca-dims {join_nums(args.pca_dims)} --pca-lrs {join_nums(args.pca_lrs)} "
            f"--lv-dims {join_nums(args.lv_dims)} --lv-lrs {join_nums(args.lv_lrs)} "
            f"--lv-bfs-fracs {join_nums(args.lv_bfs_fracs)} "
            f"--lv-likelihoods {join_nums(args.lv_likelihoods)} "
            f"--bfs-seeds {args.bfs_seeds} "
            f"--grad-clip {args.grad_clip} "
            f"--select-metric {args.select_metric} "
            f"--lv-label-smoothings {join_nums(args.lv_label_smoothings)} "
            f"--label-smoothing-target {args.label_smoothing_target} "
            f"--lv-u-norms {join_nums(args.lv_u_norms)} "
            f"--lv-u-scales {join_nums(args.lv_u_scales)} "
            f"--lv-u-scale-inits {join_nums(args.lv_u_scale_inits)} "
            f"--lv-entropy-betas {join_nums(args.lv_entropy_betas)} "
            f"--lv-partner-kl-weights {join_nums(args.lv_partner_kl_weights)} "
            f"--lv-control-updates {args.lv_control_updates} "
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
            # --skip-existing exists to reuse completed runs; wiping them first
            # would make it a no-op, so resuming implies keeping the artefacts.
            f"CLEAN_ARTIFACTS={'0' if args.skip_existing else '1'} "
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
