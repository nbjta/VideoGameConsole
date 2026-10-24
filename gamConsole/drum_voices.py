#26
"""Trommelyd når et slag rammer.

Filerne er lavet i spillet, ikke optaget fra et trommesæt.
Hi-hat og crash er kort lys støj. Lilletrommen blander støj og en tone.
Tammen er en dybere tone. Under unittest bliver der ikke spillet lyd.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
HIT_DIR = ROOT / "assets" / "drum_hero"
#26 Samme rækkefølge som HIHAT, SNARE, TOM og CRASH.
HIT_FILES: Tuple[str, ...] = ("hihat.wav", "snare.wav", "tom.wav", "crash.wav")


def _under_test() -> bool:
    return any("unittest" in arg for arg in sys.argv)


class DrumVoices:
    """Én lyd pr. tromme. Flere slag kan lyde oven i hinanden."""

    def __init__(self, folder: Path = HIT_DIR) -> None:
        self.folder = folder
        self.played: List[int] = []

    def hit(self, lane: int) -> None:
        self.played.append(lane)
        if _under_test():
            return
        path = self._path(lane)
        if path is None:
            return
        try:
            #26 afplay kan overlappe. Sangen selv stoppes stadig via NSSound.
            subprocess.Popen(
                ["afplay", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return

    def _path(self, lane: int) -> Optional[Path]:
        if lane < 0 or lane >= len(HIT_FILES):
            return None
        path = self.folder / HIT_FILES[lane]
        if not path.is_file():
            return None
        return path
