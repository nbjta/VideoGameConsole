#2
"""Kør upstream game_neck.py i egen proces, så dets vindue ikke lukker menuen."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from gamConsole.tilt_of_time_launcher import REPO, bind_camera_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if not (REPO / "game_neck.py").is_file():
        print("Mangler vendor/tilt-of-time. Kør ./scripts/setup_tilt_of_time.sh", file=sys.stderr)
        sys.exit(1)
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import game_neck

    bind_camera_index(game_neck, args.camera)
    game_neck.main()


if __name__ == "__main__":
    main()
