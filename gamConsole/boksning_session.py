#12
"""Boksning som indlejret session."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from gamConsole.boksning import BoxingGame
from gamConsole.round_score import RoundScoreSession


class BoksningSession(RoundScoreSession):
    title = "Boksning"

    def __init__(self, camera: int) -> None:
        self.game = BoxingGame(camera)
        self.cap = cv2.VideoCapture(camera)

    def stop(self) -> None:
        #12 Frigiv pose-modellen sammen med kameraet.
        self.game.close()
        if self.cap.isOpened():
            self.cap.release()

    def on_key(self, key: int) -> None:
        self.game.on_key(key)

    def tick(self) -> Optional[np.ndarray]:
        if not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        return self.game.on_frame(frame)
