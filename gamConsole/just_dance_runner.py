"""Kør olincollege just-dance fra gamConsole (korrekt cwd + sys.path)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor" / "just-dance"


def main() -> None:
    parser = argparse.ArgumentParser(description="gamConsole → Just Dance")
    parser.add_argument(
        "song",
        nargs="?",
        default="cheapthrills",
        help="cheapthrills | callmemaybe | uptownfunk | ghungroo | dontstartnow",
    )
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not (REPO / "just_dance_main.py").is_file():
        print("Mangler vendor/just-dance — kør ./scripts/setup_just_dance.sh", file=sys.stderr)
        sys.exit(1)

    os.chdir(REPO)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    video = REPO / "songs" / f"{args.song}.mp4"
    audio = REPO / "songs_audio" / f"{args.song}.mp3"
    model = REPO / "model" / "model.tflite"
    for path in (video, audio, model):
        if not path.is_file():
            print(f"Mangler fil: {path}", file=sys.stderr)
            sys.exit(1)

    from gamConsole.just_dance_controls import apply_just_dance_quit_patch
    from just_dance_main import JustDanceGame

    apply_just_dance_quit_patch()

    game = JustDanceGame(
        model_path="model/model.tflite",
        video_path=f"songs/{args.song}.mp4",
        camera_index=args.camera,
    )
    try:
        game.run(args.song)
        game.calculate_final_score()
        game.store_leaderboard("leaderboard.csv")
        print(f"Score gemt. Din score: {game.score}")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            game.controller.release_capture()
            game.controller.close_windows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
