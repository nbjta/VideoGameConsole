from pathlib import Path
from typing import Tuple

import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor" / "just-dance"


def is_installed() -> bool:
    return (REPO / "just_dance_main.py").is_file()


def setup_hint() -> str:
    return "Kør: ./scripts/setup_just_dance.sh\nKilde: vendor/just-dance"


def launch(song: str = "cheapthrills", camera: int = 0) -> Tuple[bool, str]:
    if not is_installed():
        return False, f"just-dance er ikke klonet.\n\n{setup_hint()}"
    main = REPO / "just_dance_main.py"
    model = REPO / "model" / "model.tflite"
    video = REPO / "songs" / f"{song}.mp4"
    audio = REPO / "songs_audio" / f"{song}.mp3"
    missing = [p for p in (model, video, audio) if not p.is_file()]
    if missing:
        return False, (
            "Mangler filer i vendor/just-dance (model/songs).\n"
            "Kør ./scripts/setup_just_dance.sh og tjek LFS/store downloads."
        )
    cmd = [
        sys.executable,
        "-c",
        (
            "from just_dance_main import JustDanceGame; "
            f"g=JustDanceGame('model/model.tflite','songs/{song}.mp4',{camera}); "
            f"g.run('{song}'); g.calculate_final_score(); g.store_leaderboard('leaderboard.csv')"
        ),
    ]
    try:
        subprocess.Popen(cmd, cwd=str(REPO))
    except OSError as exc:
        return False, str(exc)
    return True, f"Startede Just Dance ({song}). Luk med Q i spilvinduet."
