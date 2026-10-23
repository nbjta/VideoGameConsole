#1
"""Sol-gren kort forbi i toppen — buk dig under (som skitse)."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

BARK = (42, 58, 88)
LEAF = (38, 95, 42)
LEAF_DARK = (28, 72, 32)
SUN = (80, 230, 255)
SUN_RING = (140, 240, 255)


class OverheadBranch:
    """Gren på tværs i toppen med sol bagved — kun synlig kort tid."""

    DURATION = 50

    def __init__(
        self,
        phase: float,
        speed: float,
        *,
        enter_from_left: bool = True,
        low_y_ratio: float = 0.24,
    ) -> None:
        del phase, speed, enter_from_left
        self.start_t = 0.0
        self.low_y_ratio = low_y_ratio
        self.hit = False

    def active(self, t: float) -> bool:
        return self.start_t <= t < self.start_t + self.DURATION

    def passed(self, t: float, width: int) -> bool:
        del width
        return t >= self.start_t + self.DURATION

    def threatens_center(self, t: float, width: int) -> bool:
        del width
        return self.active(t)

    def anchor_x(self, t: float, width: int, margin: int = 160) -> int:
        del t, margin
        return width // 2

    def clearance_y(self, height: int) -> int:
        return int(height * self.low_y_ratio)

    def draw(self, canvas: np.ndarray, t: float) -> None:
        if not self.active(t):
            return
        h, w = canvas.shape[:2]
        elapsed = t - self.start_t
        fade = 1.0
        if elapsed < 6:
            fade = elapsed / 6.0
        elif elapsed > self.DURATION - 8:
            fade = max(0.0, (self.start_t + self.DURATION - t) / 8.0)
        _draw_sunlit_branch(canvas, w, h, self.low_y_ratio, fade)


def _draw_sunlit_branch(canvas: np.ndarray, w: int, h: int, low_y_ratio: float, fade: float) -> None:
    if fade <= 0.02:
        return
    overlay = canvas.copy()
    ly = int(h * low_y_ratio)
    branch_y = int(h * 0.11)
    sun_c = (int(w * 0.54), int(h * 0.13))
    sun_r = int(h * 0.075)
    cv2.circle(overlay, sun_c, sun_r, SUN, -1, cv2.LINE_AA)
    cv2.circle(overlay, sun_c, sun_r + int(h * 0.018), SUN_RING, 2, cv2.LINE_AA)
    cv2.circle(overlay, sun_c, sun_r + int(h * 0.032), SUN_RING, 1, cv2.LINE_AA)

    x_left = int(w * 0.06)
    x_right = int(w * 0.88)
    cv2.line(overlay, (x_left, branch_y + 6), (x_right, branch_y + 4), BARK, 8, cv2.LINE_AA)

    twig_x = int(w * 0.50)
    twig_top = branch_y + 5
    cv2.line(overlay, (twig_x, twig_top), (twig_x, ly - 6), BARK, 5, cv2.LINE_AA)
    cv2.ellipse(overlay, (twig_x - 10, twig_top + 8), (10, 5), -20, 0, 360, LEAF, -1, cv2.LINE_AA)
    cv2.ellipse(overlay, (twig_x + 12, twig_top + 14), (9, 5), 25, 0, 360, LEAF_DARK, -1, cv2.LINE_AA)
    cv2.ellipse(overlay, (twig_x - 6, ly - 4), (11, 6), 10, 0, 360, LEAF, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, fade, canvas, 1.0 - fade, 0, dst=canvas)


def draw_branch_warning(canvas: np.ndarray, clearance_y: int) -> None:
    w = canvas.shape[1]
    y = clearance_y
    for x in range(0, w, 24):
        cv2.line(canvas, (x, y), (x + 12, y), (100, 200, 255), 1, cv2.LINE_AA)


def reset_branches(count: int = 2) -> List[OverheadBranch]:
    del count
    return []
