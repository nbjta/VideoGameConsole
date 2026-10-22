"""Start spil via CvGames eller indbyggede Python-webcam-spil."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from gamConsole.cvgames_launcher import cvgames_binary, launch_cvgames
from gamConsole.games_registry import GameEntry
from gamConsole.jdwebcam_launcher import launch as launch_jdwebcam
from gamConsole.just_dance_launcher import launch as launch_just_dance

ROOT = Path(__file__).resolve().parents[1]
_child_procs: List[subprocess.Popen] = []


def stop_all_games() -> None:
    global _child_procs
    for proc in _child_procs:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
    _child_procs = []


def launch_game(mode: str, camera: int = 0) -> Tuple[bool, str]:
    if cvgames_binary() is not None:
        return launch_cvgames(mode, camera)
    return _launch_builtin(mode, camera)


def launch_game_entry(entry: GameEntry, camera: int = 0) -> Tuple[bool, str]:
    if entry.launcher == "just_dance":
        return launch_just_dance(entry.mode_arg or "cheapthrills", camera)
    if entry.launcher == "jdwebcam":
        return launch_jdwebcam()
    return launch_game(entry.mode_arg or "", camera)


def _launch_builtin(mode: str, camera: int) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "gamConsole.builtin_game",
        mode,
        f"--camera={camera}",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=None,
        )
        _child_procs.append(proc)
    except OSError as exc:
        return False, f"Kunne ikke starte spil: {exc}"
    titles = {
        "3d": "3D rotation",
        "bub": "Bubble shot",
        "vj": "WebCam VJ",
        "debug": "Debug view",
    }
    name = titles.get(mode, mode)
    return True, f"Startede {name}. Luk spilvindue nederst (LUK) eller Q."
