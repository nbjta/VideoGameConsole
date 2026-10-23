"""Humdingers killinger AR som indlejret session."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from gamConsole.flying_users_ar import FlyingUsersARGame
from gamConsole.round_score import RoundScoreSession


class FlyingUsersSession(RoundScoreSession):
    title = "Humdingers killinger (AR)"

    def __init__(self, camera: int) -> None:
        self.game = FlyingUsersARGame(camera)
        self.cap = cv2.VideoCapture(camera)

    def stop(self) -> None:
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
