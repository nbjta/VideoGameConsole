"""Subprocess-spil startet fra gamConsole."""

from __future__ import annotations

import subprocess
from typing import List

_child_procs: List[subprocess.Popen] = []


def register_child(proc: subprocess.Popen) -> None:
    _child_procs.append(proc)


def reap_finished_children() -> bool:
    global _child_procs
    _child_procs = [p for p in _child_procs if p.poll() is None]
    return bool(_child_procs)


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
