#9 #1
"""Fælles hovedvip for Gurli Gris og Humdingers killinger."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from gamConsole.pose_body_overlay import BodyPose

# MediaPipe: ører, ydre øjne, øjne. Samme indeks som pose_body_overlay bruger på hovedet.
_HEAD_PAIRS: Tuple[Tuple[int, int], ...] = ((7, 8), (3, 6), (2, 5))
_DEAD_DEGREES = 4.0
_FULL_DEGREES = 30.0
_FULL_SHIFT = 0.42


class HeadTilt:
    """Vip til højre giver et positivt x, så figuren flytter sig til højre."""

    def offset_x(self, pose: Optional[BodyPose], width: int) -> int:
        if pose is None or width <= 0:
            return 0
        pair = _screen_pair(pose)
        if pair is None:
            return 0
        left, right = pair
        dx = float(right[0] - left[0])
        dy = float(right[1] - left[1])
        if dx < 4.0:
            return 0
        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) < _DEAD_DEGREES:
            return 0
        angled = max(-_FULL_DEGREES, min(_FULL_DEGREES, angle))
        return int(round((angled / _FULL_DEGREES) * width * _FULL_SHIFT))


def _screen_pair(pose: BodyPose) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    for left_i, right_i in _HEAD_PAIRS:
        left = pose.get(left_i)
        right = pose.get(right_i)
        if left is None or right is None:
            continue
        if left[0] <= right[0]:
            return left, right
        return right, left
    return None
