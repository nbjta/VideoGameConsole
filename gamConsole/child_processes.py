"""Subprocess-spil startet fra gamConsole."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import List, Set

_child_procs: List[subprocess.Popen] = []
_process_groups: Set[int] = set()


def register_child(proc: subprocess.Popen, process_group: bool = False) -> None:
    _child_procs.append(proc)
    #3 Vite-servere startes som egen procesgruppe, så hele træet kan stoppes.
    if process_group:
        _process_groups.add(proc.pid)


def reap_finished_children() -> bool:
    global _child_procs
    _child_procs = [p for p in _child_procs if p.poll() is None]
    return bool(_child_procs)


def stop_child(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if proc.pid in _process_groups:
        _signal_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _signal_group(proc, signal.SIGKILL)
            proc.wait(timeout=2)
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return


def stop_all_games() -> None:
    global _child_procs
    for proc in _child_procs:
        stop_child(proc)
    _child_procs = []
    _process_groups.clear()
