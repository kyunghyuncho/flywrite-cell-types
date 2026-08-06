"""Launch full baseline + GNN hyperparameter sweep on Lightning AI (L4/T4).

Example:

    source ~/.ortet/lightning.env
    export LIGHTNING_USERNAME=kc119 LIGHTNING_TEAMSPACE=vision-model
    uv run python launch_lightning_sweep.py --machine L4 --stop-after
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.005, 0.01, 0.05])
    parser.add_argument("--dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--entropy-weights", type=float, nargs="+", default=[1.0])
    parser.add_argument("--skip-upload", action="store_true")
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

        layers = " ".join(str(x) for x in args.layers)
        lrs = " ".join(str(x) for x in args.lrs)
        dims = " ".join(str(x) for x in args.dims)
        ents = " ".join(str(x) for x in args.entropy_weights)
        cmd = (
            "python run_experiments.py "
            f"--device cuda --seed {args.seed} --epochs {args.epochs} "
            f"--minibatch {args.minibatch} --pca-max-iter {args.pca_max_iter} "
            f"--layers {layers} --lrs {lrs} --dims {dims} "
            f"--entropy-weights {ents}"
        )
        print(f"Running sweep:\n  {cmd}")
        t0 = time.time()
        out, code = studio.run_with_exit_code(cmd)
        print(out)
        if code != 0:
            raise RuntimeError(f"Sweep failed with exit code {code}")
        print(f"Sweep finished in {time.time() - t0:.1f}s")

        artifacts = [
            "sweep_results.json",
            "sweep_results.csv",
            "sweep_best.json",
            "lv_sweep_assignment_dict.npy",
            "pca_sweep_assignment_dict.npy",
        ]
        # Also download the best GNN assignment if listed in sweep_best.json
        print("Downloading summary artifacts...")
        for name in artifacts:
            try:
                studio.download_file(name, str(REPO_ROOT / name))
                print(f"  <- {name}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {name}: {exc}")

        # Download all assignment dicts produced by the sweep.
        list_cmd = (
            'python -c "import glob; '
            "print('\\\\n'.join(sorted(glob.glob('*_assignment_dict*.npy'))))\""
        )
        out, _ = studio.run_with_exit_code(list_cmd)
        for line in (out or "").splitlines():
            name = line.strip()
            if not name.endswith(".npy"):
                continue
            try:
                studio.download_file(name, str(REPO_ROOT / name))
                print(f"  <- {name}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {name}: {exc}")
    finally:
        if args.stop_after:
            print("Stopping Studio...")
            studio.stop()


if __name__ == "__main__":
    main()
