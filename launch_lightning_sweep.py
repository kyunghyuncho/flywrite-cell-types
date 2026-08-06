"""Launch unsupervised HP search + multi-seed finals on Lightning AI (L4/T4).

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


def require_auth() -> None:
    if not os.environ.get("LIGHTNING_USER_ID") or not os.environ.get("LIGHTNING_API_KEY"):
        print("Missing LIGHTNING_USER_ID / LIGHTNING_API_KEY", file=sys.stderr)
        sys.exit(1)


def resolve_machine(name: str) -> Machine:
    key = name.upper().replace("-", "_")
    if not hasattr(Machine, key):
        raise SystemExit(f"Unknown machine {name!r}")
    return getattr(Machine, key)


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

        out, code = studio.run_with_exit_code(
            "pip install -q torch scipy numpy scikit-learn tqdm pandas"
        )
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

        # Uploads land in the Studio filesystem, but the shell cwd is not always that
        # directory. Resolve the directory that contains the *new* orchestrator.
        cmd = f"""
set -euo pipefail
WORK=$(python - <<'PY'
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
)
echo "Using WORK=$WORK"
cd "$WORK"
python -c "import run_experiments; print(run_experiments.__doc__.splitlines()[0])"
python run_experiments.py {sweep_args}
"""
        print(f"Running unsupervised protocol:\n  {sweep_args}")
        t0 = time.time()
        out, code = studio.run_with_exit_code(cmd)
        print(out)
        if code != 0:
            raise RuntimeError(f"Sweep failed with exit code {code}")
        print(f"Finished in {time.time() - t0:.1f}s")

        artifacts = [
            "heldout_pairs.npz",
            "heldout_rows.npz",
            "hp_results.json",
            "hp_results.csv",
            "hp_best.json",
            "final_results.json",
            "final_results.csv",
            "final_summary.json",
            "final_summary.csv",
        ]
        print("Downloading summary artifacts...")
        for name in artifacts:
            try:
                studio.download_file(name, str(REPO_ROOT / name))
                print(f"  <- {name}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {name}: {exc}")

        # Prefer final multi-seed assignment dicts; fall back to HP ones.
        list_cmd = (
            'python -c "import glob,json; '
            "print('\\\\n'.join(sorted(glob.glob('final_*_assignment_dict*.npy') "
            "+ glob.glob('hp_*_assignment_dict*.npy') "
            "+ glob.glob('*_metrics.json'))))\""
        )
        out, _ = studio.run_with_exit_code(list_cmd)
        for line in (out or "").splitlines():
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
    finally:
        if args.stop_after:
            print("Stopping Studio...")
            studio.stop()


if __name__ == "__main__":
    main()
