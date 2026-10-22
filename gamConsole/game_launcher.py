"""Start spil via CvGames eller indbyggede Python-webcam-spil."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Tuple

from gamConsole.child_processes import register_child
from gamConsole.cvgames_launcher import cvgames_binary, launch_cvgames
from gamConsole.games_registry import GameEntry

ROOT = Path(__file__).resolve().parents[1]


def launch_game(mode: str, camera: int = 0) -> Tuple[bool, str]:
    if cvgames_binary() is not None:
        return launch_cvgames(mode, camera)
    return _launch_builtin(mode, camera)


def launch_game_entry(entry: GameEntry, camera: int = 0) -> Tuple[bool, str]:
    if entry.launcher == "just_dance":
        from gamConsole.just_dance_launcher import launch as launch_just_dance

        return launch_just_dance(entry.mode_arg or "cheapthrills", camera)
    if entry.launcher == "jdwebcam":
        from gamConsole.jdwebcam_launcher import launch as launch_jdwebcam

        return launch_jdwebcam()
    # issue #1
    if entry.launcher == "builtin":
        return _launch_python_module(entry, camera)
    return launch_game(entry.mode_arg or "", camera)


# issue #1
def _launch_python_module(entry: GameEntry, camera: int) -> Tuple[bool, str]:
    modules = {
        "flying_users": "gamConsole.flying_users_ar",
    }
    module = modules.get(entry.mode_arg or "")
    if module is None:
        return _launch_builtin(entry.mode_arg or "", camera)
    cmd = [
        sys.executable,
        "-m",
        module,
        f"--camera={camera}",
    ]
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=None)
        register_child(proc)
    except OSError as exc:
        return False, f"Kunne ikke starte spil: {exc}"
    return True, f"Startede {entry.title}. Luk spilvindue nederst (LUK) eller Q."


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
        register_child(proc)
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
