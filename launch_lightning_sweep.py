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

REPO_ROOT = Path(__file__).resolve().parent
REMOTE_LOG = "unsup_sweep.log"
REMOTE_DONE = "unsup_sweep.done"
REMOTE_PID = "unsup_sweep.pid"

CODE_FILES = [
    "gnn_vsbm.py",
    "train_lv_vsbm.py",
    "train_pca_baseline.py",
    "sparse_graph_pca.py",
    "heldout.py",
    "run_experiments.py",
    "evaluate_clustering.py",
    "index_mapping.py",
    "hidden_markov_graph.py",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "launch_lightning_sweep.py",
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


def find_work_cmd() -> str:
    return r"""
python - <<'PY'
from pathlib import Path
roots = [Path('.').resolve(), Path.home(), Path('/teamspace/studios/this_studio')]
for root in roots:
    for p in root.rglob('run_experiments.py'):
        try:
            txt = p.read_text(errors='ignore')
        except Exception:
            continue
        if 'Unsupervised HP search' in txt:
            print(p.parent)
            raise SystemExit
raise SystemExit('new run_experiments.py not found')
PY
""".strip()


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--pca-max-iter", type=int, default=10_000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--pca-lrs", type=float, nargs="+", default=[0.001, 0.01])
    parser.add_argument("--lv-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--lv-lrs", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--gnn-layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--gnn-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--gnn-lrs", type=float, nargs="+", default=[0.005, 0.01, 0.05])
    parser.add_argument("--final-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--phase", choices=["all", "hp", "final"], default="all")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-after", action="store_true")
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
    studio.start(machine)
    print(f"Studio ready on {studio.machine}")

    try:
        if not args.skip_upload:
            print("Uploading code and data...")
            for name in CODE_FILES + DATA_FILES:
                local = (REPO_ROOT / name).resolve()
                if not local.exists():
                    raise SystemExit(f"Missing {local}")
                print(f"  -> {name}")
                studio.upload_file(str(local), name)

        out, code = studio_run(studio, "pip install -q torch scipy numpy scikit-learn tqdm pandas")
        if code != 0:
            raise RuntimeError(f"pip failed: {out}")

        def join_nums(vals: list) -> str:
            return " ".join(str(x) for x in vals)

        sweep_args = (
            f"--device cuda --phase {args.phase} "
            f"--split-seed {args.split_seed} --epochs {args.epochs} "
            f"--minibatch {args.minibatch} --pca-max-iter {args.pca_max_iter} "
            f"--pca-dims {join_nums(args.pca_dims)} --pca-lrs {join_nums(args.pca_lrs)} "
            f"--lv-dims {join_nums(args.lv_dims)} --lv-lrs {join_nums(args.lv_lrs)} "
            f"--gnn-layers {join_nums(args.gnn_layers)} --gnn-dims {join_nums(args.gnn_dims)} "
            f"--gnn-lrs {join_nums(args.gnn_lrs)} --final-seeds {join_nums(args.final_seeds)}"
        )
        if args.skip_existing:
            sweep_args += " --skip-existing"

        # Kill any previous detached sweep, then start a fresh nohup job.
        print(f"Detaching unsupervised protocol:\n  {sweep_args}")
        start_cmd = f"""
set -euo pipefail
WORK=$({find_work_cmd()})
echo "Using WORK=$WORK"
cd "$WORK"
# Stop prior detached sweep if still running.
if [ -f {REMOTE_PID} ]; then
  old=$(cat {REMOTE_PID} || true)
  if [ -n "${{old}}" ] && kill -0 "${{old}}" 2>/dev/null; then
    echo "Killing prior sweep pid ${{old}}"
    kill "${{old}}" || true
    sleep 2
    kill -9 "${{old}}" 2>/dev/null || true
  fi
fi
pkill -f 'python run_experiments.py' 2>/dev/null || true
rm -f {REMOTE_DONE} {REMOTE_LOG}
nohup bash -lc 'python -u run_experiments.py {sweep_args}; echo $? > {REMOTE_DONE}' \
  > {REMOTE_LOG} 2>&1 &
echo $! > {REMOTE_PID}
echo "Started pid=$(cat {REMOTE_PID})"
sleep 2
tail -n 40 {REMOTE_LOG} || true
"""
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
            status_cmd = f"""
cd /teamspace/studios/this_studio || exit 1
pid=$(cat {REMOTE_PID} 2>/dev/null || true)
alive=0
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then alive=1; fi
done=0
[ -f {REMOTE_DONE} ] && done=1
echo "STATUS alive=$alive done=$done pid=$pid"
tail -n 25 {REMOTE_LOG} 2>/dev/null || true
"""
            out, _ = studio_run(studio, status_cmd)
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
