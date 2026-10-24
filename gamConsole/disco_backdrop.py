"""Animeret disco-scene til Just Dance."""

from __future__ import annotations

import math

import cv2
import numpy as np

# BGR
_PINK = (180, 40, 255)
_CYAN = (255, 210, 20)
_PURPLE = (255, 40, 170)
_YELLOW = (30, 230, 255)
_GREEN = (60, 255, 90)
_BLUE = (255, 70, 40)
_LIGHTS = (_PINK, _CYAN, _YELLOW, _GREEN, _PURPLE, _BLUE)


class DiscoBackdrop:
    """Mørk sal, roterende spots, discokugle og et ternet dansegulv."""

    def __init__(self) -> None:
        self.t = 0.0

    def tick(self) -> None:
        self.t += 1.0

    def render(self, width: int, height: int) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        horizon = int(height * 0.58)
        self._gradient(canvas)
        self._sparkles(canvas, horizon)
        self._beams(canvas)
        self._floor(canvas, horizon)
        self._floor_glow(canvas, horizon)
        self._ball(canvas)
        return canvas

    def _gradient(self, canvas: np.ndarray) -> None:
        height = canvas.shape[0]
        u = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        canvas[:, :, 0] = (55 + 70 * (1.0 - u)).astype(np.uint8)
        canvas[:, :, 1] = (8 + 22 * u).astype(np.uint8)
        canvas[:, :, 2] = (42 + 55 * (1.0 - u)).astype(np.uint8)

    def _beams(self, canvas: np.ndarray) -> None:
        height, width = canvas.shape[:2]
        origin = (width // 2, int(height * 0.14))
        overlay = canvas.copy()
        length = int(math.hypot(width, height))
        for index, color in enumerate(_LIGHTS):
            angle = self.t * 0.045 + index * (math.pi / len(_LIGHTS))
            spread = 0.11
            left = (
                int(origin[0] + math.cos(angle - spread) * length),
                int(origin[1] + math.sin(angle - spread) * length),
            )
            right = (
                int(origin[0] + math.cos(angle + spread) * length),
                int(origin[1] + math.sin(angle + spread) * length),
            )
            pts = np.array([origin, left, right], dtype=np.int32)
            cv2.fillConvexPoly(overlay, pts, color)
        cv2.addWeighted(overlay, 0.38, canvas, 0.62, 0, canvas)

    def _floor(self, canvas: np.ndarray, horizon: int) -> None:
        height, width = canvas.shape[:2]
        rows = 8
        cols = 10
        phase = int(self.t // 10) % 2
        span = height - horizon
        for row in range(rows):
            y0 = horizon + int(span * (row / rows) ** 1.55)
            y1 = horizon + int(span * ((row + 1) / rows) ** 1.55)
            if y1 <= y0:
                y1 = y0 + 1
            frac0 = row / rows
            frac1 = (row + 1) / rows
            for col in range(cols):
                color = _LIGHTS[(row + col + phase) % len(_LIGHTS)]
                pts = np.array(
                    [
                        [self._floor_x(col / cols, frac0, width), y0],
                        [self._floor_x((col + 1) / cols, frac0, width), y0],
                        [self._floor_x((col + 1) / cols, frac1, width), y1],
                        [self._floor_x(col / cols, frac1, width), y1],
                    ],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(canvas, pts, color)

    def _floor_glow(self, canvas: np.ndarray, horizon: int) -> None:
        glow = canvas.copy()
        height, width = canvas.shape[:2]
        pulse = 0.5 + 0.5 * math.sin(self.t * 0.18)
        for index, color in enumerate((_PINK, _CYAN, _YELLOW)):
            cx = int(width * (0.22 + 0.28 * index))
            cy = horizon + int((height - horizon) * (0.35 + 0.08 * math.sin(self.t * 0.1 + index)))
            radius = max(8, int(width * (0.08 + 0.03 * pulse)))
            cv2.circle(glow, (cx, cy), radius, color, -1, cv2.LINE_AA)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[horizon:, :] = 255
        colored = canvas.copy()
        cv2.addWeighted(glow, 0.28, canvas, 0.72, 0, colored)
        canvas[mask > 0] = colored[mask > 0]

    def _ball(self, canvas: np.ndarray) -> None:
        height, width = canvas.shape[:2]
        center = (width // 2, max(8, int(height * 0.13)))
        radius = max(8, int(min(width, height) * 0.075))
        cv2.circle(canvas, center, radius, (210, 210, 220), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, radius, (255, 255, 255), 1, cv2.LINE_AA)
        for ring in range(3):
            for facet in range(8):
                angle = self.t * 0.07 + facet * (math.pi / 4) + ring
                dist = radius * (0.35 + 0.22 * ring)
                point = (
                    int(center[0] + math.cos(angle) * dist),
                    int(center[1] + math.sin(angle) * dist * 0.72),
                )
                color = _LIGHTS[(facet + ring + int(self.t // 6)) % len(_LIGHTS)]
                cv2.circle(canvas, point, max(1, radius // 7), color, -1, cv2.LINE_AA)

    def _sparkles(self, canvas: np.ndarray, horizon: int) -> None:
        height, width = canvas.shape[:2]
        for index in range(16):
            x = int((math.sin(self.t * 0.06 + index * 1.7) * 0.5 + 0.5) * (width - 1))
            y = int((math.cos(self.t * 0.05 + index * 2.1) * 0.5 + 0.5) * max(horizon - 4, 1))
            y = max(0, min(height - 1, y))
            color = _LIGHTS[index % len(_LIGHTS)]
            radius = 1 + (index + int(self.t)) % 3
            cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)

    @staticmethod
    def _floor_x(col_frac: float, depth: float, width: int) -> int:
        span = 0.16 + 0.84 * depth
        return int(width * (0.5 + (col_frac - 0.5) * span))
