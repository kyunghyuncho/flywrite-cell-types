"""Launch GNN-vSBM training on a Lightning AI Studio GPU (T4 or L4).

Requires programmatic access keys (Settings → Keys on lightning.ai):

    export LIGHTNING_USER_ID=...
    export LIGHTNING_API_KEY=...

Optional overrides:

    export LIGHTNING_USERNAME=...
    export LIGHTNING_TEAMSPACE=...
    export LIGHTNING_ORG=...

Example:

    source ~/.ortet/lightning.env
    uv run python launch_lightning_train.py --machine L4 --epochs 20 --stop-after
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
    "evaluate_clustering.py",
    "index_mapping.py",
    "hidden_markov_graph.py",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
]

DATA_FILES = [
    "sparse_connectivity_matrix.npz",
    "root_id_to_index_mapping.json",
    "root_id_type_dict.pkl",
    "cluster_assignment_dict_729.npy",
    "pca_cluster_assignment_dict_729.npy",
]


def require_auth() -> None:
    if not os.environ.get("LIGHTNING_USER_ID") or not os.environ.get("LIGHTNING_API_KEY"):
        print(
            "Missing Lightning credentials.\n"
            "Export LIGHTNING_USER_ID and LIGHTNING_API_KEY, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)


def resolve_machine(name: str) -> Machine:
    key = name.upper().replace("-", "_")
    if not hasattr(Machine, key):
        raise SystemExit(f"Unknown machine {name!r}. Try T4 or L4.")
    return getattr(Machine, key)


def resolve_owner_and_teamspace() -> tuple[dict[str, str], str]:
    """Return Studio kwargs fragment for owner + teamspace name."""
    org = os.environ.get("LIGHTNING_ORG")
    user = os.environ.get("LIGHTNING_USERNAME")
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE")

    if not user and not org:
        try:
            from lightning_sdk.utils.resolve import _get_authed_user

            user = _get_authed_user().name
            print(f"Resolved Lightning username: {user}")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                "Could not resolve Lightning username. "
                "Set LIGHTNING_USERNAME or LIGHTNING_ORG.\n"
                f"Underlying error: {exc}"
            ) from exc

    if not teamspace:
        try:
            from lightning_sdk.utils.resolve import _get_teamspace_names_for_authed_user

            names = list(_get_teamspace_names_for_authed_user())
            if not names:
                raise RuntimeError("no teamspaces returned")
            teamspace = names[0]
            print(f"Resolved teamspaces {names}; using {teamspace!r}")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                "Could not resolve LIGHTNING_TEAMSPACE automatically. "
                "Set it explicitly (from the Studio URL).\n"
                f"Underlying error: {exc}"
            ) from exc

    owner: dict[str, str] = {"org": org} if org else {"user": user}  # type: ignore[dict-item]
    return owner, teamspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio-name", default="flywrite-gnn-vsbm")
    parser.add_argument("--machine", default="L4", help="T4 or L4 (default: L4)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--stop-after", action="store_true")
    args = parser.parse_args()
    require_auth()

    machine = resolve_machine(args.machine)
    owner, teamspace = resolve_owner_and_teamspace()

    print(f"Opening Studio {args.studio_name!r} in teamspace {teamspace!r} on {args.machine}...")
    studio = Studio(
        name=args.studio_name,
        teamspace=teamspace,
        create_ok=True,
        **owner,
    )
    studio.start(machine)
    print(f"Studio ready on machine={studio.machine}")

    try:
        if not args.skip_upload:
            missing = [f for f in DATA_FILES if not (REPO_ROOT / f).exists()]
            if missing:
                raise SystemExit(f"Missing local data files: {missing}")
            print("Uploading code and data...")
            for name in CODE_FILES + DATA_FILES:
                local = (REPO_ROOT / name).resolve()
                print(f"  -> {name} ({local.stat().st_size / 1e6:.1f} MB)")
                studio.upload_file(str(local), name)

        print("Installing Python deps on the Studio...")
        out, code = studio.run_with_exit_code(
            "pip install -q torch scipy numpy scikit-learn tqdm pandas"
        )
        if code != 0:
            raise RuntimeError(f"pip install failed ({code}):\n{out}")

        train_cmd = (
            "python gnn_vsbm.py "
            f"--epochs {args.epochs} "
            f"--minibatch {args.minibatch} "
            f"--layers {args.layers} "
            f"--lr {args.lr} "
            f"--seed {args.seed} "
            "--device cuda "
            "--out-prefix gnn"
        )
        if args.max_updates is not None:
            train_cmd += f" --max-updates {args.max_updates}"

        print(f"Running: {train_cmd}")
        t0 = time.time()
        out, code = studio.run_with_exit_code(train_cmd)
        print(out)
        if code != 0:
            raise RuntimeError(f"Training failed with exit code {code}")
        print(f"Training finished in {time.time() - t0:.1f}s")

        print("Evaluating...")
        out, code = studio.run_with_exit_code(
            "python evaluate_clustering.py "
            "--pred gnn_assignment_dict_729.npy "
            "cluster_assignment_dict_729.npy "
            "pca_cluster_assignment_dict_729.npy"
        )
        print(out)
        if code != 0:
            raise RuntimeError(f"Evaluation failed with exit code {code}")

        artifacts = [
            "gnn_assignment_dict_729.npy",
            "gnn_assignments.npy",
            "gnn_scores.npy",
            "gnn_U.npy",
            "gnn_checkpoint.pt",
        ]
        print("Downloading artifacts...")
        for name in artifacts:
            try:
                studio.download_file(name, str(REPO_ROOT / name))
                print(f"  <- {name}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! failed to download {name}: {exc}")
    finally:
        if args.stop_after:
            print("Stopping Studio...")
            studio.stop()
        else:
            print(
                f"Studio left running on {args.machine}. Re-run with --stop-after to shut it down."
            )


if __name__ == "__main__":
    main()
