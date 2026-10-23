#!/usr/bin/env python3
"""Skriv assets/game_thumbs/<game-id>.png til menu-gitter."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from gamConsole.game_thumbnails import THUMB_DIR, _generate  # noqa: E402
from gamConsole.games_registry import GAMES  # noqa: E402


def main() -> None:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    for game in GAMES:
        img = _generate(game.id, 256)
        path = THUMB_DIR / f"{game.id}.png"
        cv2.imwrite(str(path), img)
        print("Wrote", path)


if __name__ == "__main__":
    main()
