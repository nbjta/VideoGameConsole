"""Indbyggede webcam-spil som tick-sessioner (ét vindue)."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from gamConsole.hand_pip_overlay import HandPipOverlay
from gamConsole.house_build import HouseBuildScene, apply_hand_input


def _motion_mask(gray: np.ndarray, prev: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if prev is None:
        return np.zeros_like(gray), gray
    diff = cv2.absdiff(gray, prev)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask, gray


def _motion_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    m = cv2.moments(mask)
    if m["m00"] < 5000:
        return None
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


class BuiltinSession:
    title = "Webcam-spil"

    def __init__(self, mode: str, camera: int) -> None:
        self.mode = mode
        self.cap = cv2.VideoCapture(camera)
        self._prev: Optional[np.ndarray] = None
        self._bubbles: list = []
        self._rng = np.random.default_rng(42)
        self._house = HouseBuildScene()
        self._hand_pip: Optional[HandPipOverlay] = HandPipOverlay() if mode == "3d" else None
        titles = {
            #8
            "3d": "Husbygning",
            "bub": "Bubble shot",
            "debug": "Debug view",
        }
        self.title = titles.get(mode, mode)

    def stop(self) -> None:
        if self._hand_pip is not None:
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
        if self.mode == "debug":
            return self._tick_debug(frame)
        if self.mode == "bub":
            return self._tick_bub(frame)
        if self.mode == "3d":
            return self._tick_3d(frame)
        return frame

    def _gray(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        mask, self._prev = _motion_mask(gray, self._prev)
        return gray, mask

    def _tick_debug(self, frame: np.ndarray) -> np.ndarray:
        _, mask = self._gray(frame)
        return np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])

    def _tick_bub(self, frame: np.ndarray) -> np.ndarray:
        _, mask = self._gray(frame)
        c = _motion_centroid(mask)
        if c is not None:
            color = tuple(int(x) for x in self._rng.integers(80, 255, size=3))
            self._bubbles.append((float(c[0]), float(c[1]), 18.0, color))
        alive = []
        for x, y, r, col in self._bubbles:
            r += 0.6
            if r < 120:
                alive.append((x, y, r, col))
                cv2.circle(frame, (int(x), int(y)), int(r), col, 2)
        self._bubbles = alive
        return frame

    def _tick_3d(self, frame: np.ndarray) -> np.ndarray:
        #8 Spejlvendt webcam — hånd til højre flytter bjælken til højre.
        camera = cv2.flip(frame, 1)
        assert self._hand_pip is not None
        #8 Bjælken flytter sig kun mens hånden er klemt.
        track = self._hand_pip.read(camera)
        apply_hand_input(self._house, track, camera)
        self._house.draw(frame)
        self._hand_pip.draw(frame, camera=camera)
        return frame
