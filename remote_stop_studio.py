"""Stop the Lightning Studio used for this sweep (called on the Studio itself)."""

from __future__ import annotations

import os
import sys


def main() -> None:
    # Prefer already-exported env; do not require a credentials file on disk.
    if not os.environ.get("LIGHTNING_USER_ID") or not os.environ.get("LIGHTNING_API_KEY"):
        print("Missing LIGHTNING_USER_ID / LIGHTNING_API_KEY in environment", file=sys.stderr)
        sys.exit(2)

    from lightning_sdk import Studio

    name = os.environ.get("STUDIO_NAME", "flywrite-gnn-vsbm")
    user = os.environ.get("LIGHTNING_USERNAME", "kc119")
    team = os.environ.get("LIGHTNING_TEAMSPACE", "vision-model")
    print(f"Requesting stop for Studio {name!r} ({user}/{team})...")
    Studio(name=name, teamspace=team, user=user, create_ok=True).stop()
    print("Studio stop requested.")


if __name__ == "__main__":
    main()
