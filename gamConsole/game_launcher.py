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
    #6
    if entry.launcher == "handora":
        from gamConsole.handora_launcher import launch as launch_handora

        return launch_handora(camera)
    #3
    if entry.launcher == "wiisports":
        from gamConsole.wiisports_launcher import launch as launch_wiisports

        return launch_wiisports(camera)
    if entry.launcher == "just_dance":
        from gamConsole.just_dance_launcher import launch as launch_just_dance

        return launch_just_dance(entry.mode_arg or "cheapthrills", camera)
    #2
    if entry.launcher == "tilt_of_time":
        from gamConsole.tilt_of_time_launcher import launch as launch_tilt

        return launch_tilt(camera)
    #10
    if entry.launcher == "builtin" and entry.mode_arg == "mars_rover":
        return _start_module("gamConsole.mars_rover", entry.title, camera)
    if entry.launcher == "builtin":
        return _launch_python_module(entry, camera)
    return launch_game(entry.mode_arg or "", camera)


#10
def _start_module(module: str, title: str, camera: int) -> Tuple[bool, str]:
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
    return True, f"Startede {title}. Luk spilvindue nederst (LUK) eller Q."


def _launch_python_module(entry: GameEntry, camera: int) -> Tuple[bool, str]:
    modules = {
        "flying_users": "gamConsole.flying_users_ar",
        #9
        "gurli_gris": "gamConsole.gurli_gris",
        #11
        "maalmand": "gamConsole.maalmand",
        #12
        "boksning": "gamConsole.boksning",
        #13
        "drum_hero": "gamConsole.drum_hero",
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
    #8
    titles = {
        "3d": "Husbygning",
        "bub": "Bubble shot",
        "debug": "Debug view",
    }
    name = titles.get(mode, mode)
    return True, f"Startede {name}. Luk spilvindue nederst (LUK) eller Q."
