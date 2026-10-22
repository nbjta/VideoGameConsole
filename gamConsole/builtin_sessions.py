"""Indbyggede webcam-spil som tick-sessioner (ét vindue)."""

from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "vendor" / "webcam-games" / "Data"


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
        self._vj_images: Optional[List[np.ndarray]] = None
        self._vj_idx = 0
        self._vj_blend = 0.0
        self._ax, self._ay = 0.4, 0.4
        titles = {
            "3d": "3D rotation",
            "bub": "Bubble shot",
            "vj": "WebCam VJ",
            "debug": "Debug view",
        }
        self.title = titles.get(mode, mode)

    def stop(self) -> None:
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
        if self.mode == "vj":
            return self._tick_vj(frame)
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

    def _load_vj(self) -> List[np.ndarray]:
        if self._vj_images is not None:
            return self._vj_images
        paths: List[str] = []
        if DATA.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                paths.extend(glob.glob(str(DATA / ext)))
        images: List[np.ndarray] = []
        for p in sorted(paths)[:12]:
            img = cv2.imread(p)
            if img is not None:
                images.append(img)
        if not images:
            for hue in range(0, 360, 45):
                block = np.zeros((480, 640, 3), dtype=np.uint8)
                block[:] = cv2.cvtColor(np.uint8([[[hue // 2, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
                images.append(block)
        self._vj_images = images
        return images

    def _tick_vj(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        _, mask = self._gray(frame)
        motion = float(np.count_nonzero(mask)) / mask.size
        if motion > 0.02:
            self._vj_blend = min(1.0, self._vj_blend + 0.04)
        else:
            self._vj_blend = max(0.0, self._vj_blend - 0.01)
        images = self._load_vj()
        if self._vj_blend >= 0.98:
            self._vj_idx = (self._vj_idx + 1) % len(images)
            self._vj_blend = 0.0
        overlay = cv2.resize(images[self._vj_idx], (w, h))
        return cv2.addWeighted(frame, 1.0 - self._vj_blend, overlay, self._vj_blend, 0)

    def _tick_3d(self, frame: np.ndarray) -> np.ndarray:
        _, mask = self._gray(frame)
        c = _motion_centroid(mask)
        if c is not None:
            h, w = frame.shape[:2]
            self._ay = (c[0] / w - 0.5) * 3
            self._ax = (c[1] / h - 0.5) * 3
        self._draw_cube(frame, self._ax, self._ay)
        return frame

    def _draw_cube(self, frame: np.ndarray, angle_x: float, angle_y: float) -> None:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        s = min(w, h) // 6
        pts = np.array(
            [
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            ],
            dtype=float,
        )
        rot_x = np.array([[1, 0, 0], [0, math.cos(angle_x), -math.sin(angle_x)], [0, math.sin(angle_x), math.cos(angle_x)]])
        rot_y = np.array([[math.cos(angle_y), 0, math.sin(angle_y)], [0, 1, 0], [-math.sin(angle_y), 0, math.cos(angle_y)]])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        projected = []
        for p in pts:
            v = rot_y @ rot_x @ (p * s)
            projected.append((int(cx + v[0]), int(cy + v[1])))
        for a, b in edges:
            cv2.line(frame, projected[a], projected[b], (80, 180, 255), 2)
