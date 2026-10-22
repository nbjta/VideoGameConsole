from pathlib import Path
from typing import Tuple

import subprocess
import sys

from gamConsole.child_processes import register_child

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor" / "just-dance"

SONGS = (
    "cheapthrills",
    "callmemaybe",
    "uptownfunk",
    "ghungroo",
    "dontstartnow",
)


def is_installed() -> bool:
    return (REPO / "just_dance_main.py").is_file()


def setup_hint() -> str:
    return "Kør: ./scripts/setup_just_dance.sh\nKilde: vendor/just-dance"


def launch(song: str = "cheapthrills", camera: int = 0) -> Tuple[bool, str]:
    if not is_installed():
        return False, f"just-dance er ikke klonet.\n\n{setup_hint()}"

    if song not in SONGS:
        song = "cheapthrills"

    model = REPO / "model" / "model.tflite"
    video = REPO / "songs" / f"{song}.mp4"
    audio = REPO / "songs_audio" / f"{song}.mp3"
    missing = [p for p in (model, video, audio) if not p.is_file()]
    if missing:
        return False, (
            "Mangler filer i vendor/just-dance.\n"
            f"Eksempel: {missing[0]}\n"
            "Kør ./scripts/setup_just_dance.sh"
        )

    cmd = [
        sys.executable,
        "-m",
        "gamConsole.just_dance_runner",
        song,
        f"--camera={camera}",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=None,
        )
        register_child(proc)
    except OSError as exc:
        return False, str(exc)
    return (
        True,
        f"Just Dance ({song}) starter — vent på TensorFlow. Luk med Q. "
        "Menu-webcam genoptages når spillet lukkes.",
    )
