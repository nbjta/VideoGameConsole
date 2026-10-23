#2
"""Status i gamConsole-vinduet mens Tilt of Time kører i sit eget vindue.

Upstream kalder cv2.destroyAllWindows() ved afslutning, så spillet skal
køre i en separat proces. Den proces ejer også kameraet.
"""

from __future__ import annotations

import subprocess
from typing import List, Optional

import cv2
import numpy as np

from gamConsole.tilt_of_time_launcher import start


class TiltOfTimeSession:
    title = "Tilt of Time"

    def __init__(self, camera: int) -> None:
        self.camera = camera
        self._proc: Optional[subprocess.Popen] = None
        self._error: Optional[str] = None
        self.message = ""
        self._start()

    def _start(self) -> None:
        proc, msg = start(self.camera)
        self.message = msg
        self._proc = proc
        if proc is None:
            self._error = msg

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    def on_key(self, key: int) -> None:
        return

    def tick(self) -> Optional[np.ndarray]:
        if self._error:
            return _text_frame(self._error.splitlines())
        proc = self._proc
        if proc is None:
            return _text_frame(["Tilt of Time er ikke startet."])
        code = proc.poll()
        if code is not None:
            if code != 0:
                self._error = f"Tilt of Time stoppede (kode {code}). Se terminalen."
                return _text_frame(self._error.splitlines())
            return None
        return _text_frame(
            [
                "Tilt of Time er i gang i eget vindue.",
                "Vinduet kommer om et par sekunder.",
                "1 solo    2 duel    mellemrum starter",
                "Q i spilvinduet lukker spillet.",
                "Hjem her stopper spillet.",
            ]
        )


def _text_frame(lines: List[str]) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (26, 26, 46)
    y = 220
    for line in lines:
        if not line:
            y += 28
            continue
        cv2.putText(
            frame,
            line[:90],
            (80, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (234, 234, 234),
            2,
            cv2.LINE_AA,
        )
        y += 48
    return frame
