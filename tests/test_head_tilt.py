#9 #1
"""Hovedvip flytter Gurli og Humdingers-lederen til samme side."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from gamConsole.head_tilt import HeadTilt  # noqa: E402
from gamConsole.pose_body_overlay import BodyPose  # noqa: E402


def _pose(left_ear: tuple, right_ear: tuple) -> BodyPose:
    points = [(-1, -1)] * 33
    points[7] = left_ear
    points[8] = right_ear
    return BodyPose(640, 360, tuple(points))


class HeadTiltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tilt = HeadTilt()

    def test_level_head_stays_at_center(self) -> None:
        self.assertEqual(self.tilt.offset_x(_pose((200, 80), (280, 80)), 640), 0)

    def test_tilt_right_moves_positive(self) -> None:
        straight = self.tilt.offset_x(_pose((200, 80), (280, 80)), 640)
        right = self.tilt.offset_x(_pose((200, 70), (280, 130)), 640)
        self.assertGreater(right, straight)
        self.assertGreater(right, 0)

    def test_tilt_left_moves_negative(self) -> None:
        left = self.tilt.offset_x(_pose((200, 130), (280, 70)), 640)
        self.assertLess(left, 0)

    def test_missing_pose_does_not_move(self) -> None:
        self.assertEqual(self.tilt.offset_x(None, 640), 0)

    def test_gurli_and_humdingers_share_the_same_offset(self) -> None:
        from gamConsole.ar_player_view import HeadTilt as HumdingersTilt
        from gamConsole.gurli_gris import HeadTilt as GurliTilt

        pose = _pose((200, 60), (280, 140))
        self.assertEqual(GurliTilt().offset_x(pose, 640), HumdingersTilt().offset_x(pose, 640))
        self.assertGreater(GurliTilt().offset_x(pose, 640), 0)


class GurliMovesWithHeadTest(unittest.TestCase):
    def test_figure_shifts_right_when_the_head_tilts_right(self) -> None:
        from gamConsole.gurli_gris import GurliGrisFigure

        figure = GurliGrisFigure()
        background = (255, 180, 80)
        level = np.full((360, 640, 3), background, dtype=np.uint8)
        tilted = level.copy()
        figure.draw(level, _pose((220, 90), (300, 90)))
        figure.draw(tilted, _pose((220, 70), (300, 150)))
        self.assertGreater(_mass_x(tilted, background), _mass_x(level, background))


def _mass_x(image: np.ndarray, background: tuple) -> float:
    mask = np.any(image != np.array(background, dtype=np.uint8), axis=2)
    xs = np.where(mask)[1]
    return float(xs.mean())


if __name__ == "__main__":
    unittest.main()
