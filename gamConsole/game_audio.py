#9 #1
"""Lyde fra Kenney, licens CC0. Filerne ligger i assets/audio/kenney.

afplay bruges, samme måde som biltræffen allerede gjorde.
Under unittest bliver der ikke spillet lyd ud af højttaleren.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "assets" / "audio" / "kenney"


def _under_test() -> bool:
    return any("unittest" in arg for arg in sys.argv)


class GameAudio:
    """Korte kenney-lyde til begge spil. Navnene er cue-navnene på wav-filerne."""

    def __init__(self, folder: Path = AUDIO_DIR) -> None:
        self.folder = folder
        self.played: List[str] = []

    def play(self, cue: str) -> None:
        self.played.append(cue)
        path = self.folder / f"{cue}.wav"
        if _under_test() or not path.is_file():
            return
        try:
            subprocess.Popen(
                ["afplay", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return

    def click(self) -> None:
        #9 #1 Spil igen.
        self.play("click")

    def points(self, value: int) -> None:
        #9 #1 Plus giver confirmation_001. Minus giver error_001.
        self.play("positive" if value >= 0 else "negative")

    def bubble(self) -> None:
        #9 Sæbeboblen bruger impactGlass_light_000.
        self.play("bubble")

    def car(self) -> None:
        #1 Bilen bruger impactMetal_medium_000.
        self.play("car")

    def round_over(self) -> None:
        #9 #1 jingler_NES00, cirka 1,8 sekund. Det er ikke baggrundsmusik.
        self.play("round")


game_audio = GameAudio()
