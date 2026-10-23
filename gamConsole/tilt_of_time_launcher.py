#2
"""Start Tilt of Time (ebinrajiiit/Tilt_of_Time) fra gamConsole."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from gamConsole.child_processes import register_child

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor" / "tilt-of-time"


def is_installed() -> bool:
    return (REPO / "game_neck.py").is_file() and (REPO / "vendor" / "face_landmarker.task").is_file()


def setup_hint() -> str:
    return "Kør: ./scripts/setup_tilt_of_time.sh\nKilde: https://github.com/ebinrajiiit/Tilt_of_Time"


def bind_camera_index(module, camera: int) -> None:
    """Upstream `main()` åbner altid `cv2.VideoCapture(0)`.

    Spillet har intet kamera-argument. Denne binding sender menuens
    kamera-index videre, kun når kaldet er heltals-indeks 0.
    """
    original = module.cv2.VideoCapture

    def capture(index=0, *args, **kwargs):
        if camera != 0 and isinstance(index, int) and index == 0:
            index = camera
        return original(index, *args, **kwargs)

    module.cv2.VideoCapture = capture


def start(camera: int = 0) -> Tuple[Optional[subprocess.Popen], str]:
    if not is_installed():
        return None, f"Tilt of Time er ikke klonet.\n\n{setup_hint()}"
    cmd = [
        sys.executable,
        "-m",
        "gamConsole.tilt_of_time_runner",
        f"--camera={camera}",
    ]
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT))
    except OSError as exc:
        return None, f"Kunne ikke starte Tilt of Time: {exc}"
    register_child(proc)
    return proc, "Tilt of Time starter i eget vindue. Hjem her vender tilbage til menuen."


def launch(camera: int = 0) -> Tuple[bool, str]:
    proc, msg = start(camera)
    return proc is not None, msg
