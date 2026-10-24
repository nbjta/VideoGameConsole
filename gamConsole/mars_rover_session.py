#10
"""Mars-rover som indlejret session i spilmenuen."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from gamConsole.hand_pip_overlay import HandPipOverlay
from gamConsole.house_build import apply_hand_input
from gamConsole.mars_rover import MarsRoverBuildScene


class MarsRoverSession:
    title = "Mars-rover"

    def __init__(self, camera: int) -> None:
        self.cap = cv2.VideoCapture(camera)
        self._scene = MarsRoverBuildScene()
        self._hand_pip = HandPipOverlay()

    def stop(self) -> None:
        self._hand_pip.close()
        if self.cap.isOpened():
            self.cap.release()

    def on_key(self, key: int) -> None:
        return

    def tick(self) -> Optional[np.ndarray]:
        if not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        #10 Spejlvendt webcam — samme klem-styring som Husbygning.
        camera = cv2.flip(frame, 1)
        track = self._hand_pip.read(camera)
        apply_hand_input(self._scene, track, camera)
        self._scene.draw(frame)
        self._hand_pip.draw(frame, camera=camera)
        return frame
