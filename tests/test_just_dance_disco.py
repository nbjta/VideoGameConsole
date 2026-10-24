"""Disco-baggrund bag danseren i Just Dance."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from gamConsole.disco_backdrop import DiscoBackdrop
from gamConsole.disco_dancer import DiscoDancer, SilhouetteShift
from gamConsole.head_tilt import HeadTilt
from gamConsole.jointed_figure import L_INDEX, R_INDEX, turn
from gamConsole.game_thumbnails import clear_thumbnail_cache, get_game_thumbnail
from gamConsole.just_dance_stage import DanceAvatar, JustDanceStage
from gamConsole.pose_body_overlay import (
    BodyPose,
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)


class DiscoBackdropTest(unittest.TestCase):
    def test_render_is_colorful_and_moves(self) -> None:
        still = DiscoBackdrop()
        still.t = 0.0
        first = still.render(160, 120)
        later = DiscoBackdrop()
        later.t = 30.0
        second = later.render(160, 120)

        self.assertEqual(first.shape, (120, 160, 3))
        hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
        self.assertGreater(int(hsv[:, :, 1].max()), 180)
        self.assertFalse(np.array_equal(first, second))
        floor = first[90:, :, :]
        self.assertGreater(int(floor.std()), 20)

    def test_menu_thumb_uses_disco(self) -> None:
        clear_thumbnail_cache()
        thumb = get_game_thumbnail("just-dance", 96)
        self.assertEqual(thumb.shape, (96, 96, 3))
        hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
        self.assertGreater(int(hsv[:, :, 1].max()), 150)


def _pose(arm_y: int, leg_y: int) -> BodyPose:
    points = [(-1, -1)] * 33
    points[L_SHOULDER] = (220, 150)
    points[R_SHOULDER] = (300, 150)
    points[L_ELBOW] = (180, arm_y)
    points[R_ELBOW] = (340, arm_y)
    points[L_WRIST] = (160, arm_y)
    points[R_WRIST] = (360, arm_y)
    points[L_HIP] = (230, 230)
    points[R_HIP] = (290, 230)
    points[L_KNEE] = (230, leg_y)
    points[R_KNEE] = (290, leg_y)
    points[L_ANKLE] = (230, leg_y)
    points[R_ANKLE] = (290, leg_y)
    return BodyPose(640, 360, tuple(points))


def _torso(shift: int, ears: tuple) -> BodyPose:
    points = [(-1, -1)] * 33
    points[7], points[8] = ears
    points[L_SHOULDER] = (280 + shift, 150)
    points[R_SHOULDER] = (360 + shift, 150)
    points[L_HIP] = (300 + shift, 230)
    points[R_HIP] = (340 + shift, 230)
    return BodyPose(640, 360, tuple(points))


def _green_x(image: np.ndarray) -> float:
    green = (image[:, :, 1] > 140) & (image[:, :, 1] > image[:, :, 2] + 20) & (image[:, :, 1] > image[:, :, 0] + 20)
    xs = np.where(green)[1]
    return float(xs.mean()) if len(xs) else 0.0


def _mean_y(mask: np.ndarray) -> float:
    ys = np.where(mask)[0]
    return float(ys.mean()) if len(ys) else 0.0


class JustDanceStageTest(unittest.TestCase):
    def test_webcam_is_replaced_by_the_avatar(self) -> None:
        stage = JustDanceStage()
        try:
            webcam = np.full((180, 240, 3), (0, 0, 255), dtype=np.uint8)
            out = stage.composite(webcam)
        finally:
            stage.close()
        self.assertEqual(int(np.all(out == webcam, axis=2).sum()), 0)
        blank = np.zeros_like(webcam)
        stage.avatar.draw(blank, None)
        orange = (blank[:, :, 2] > 200) & (blank[:, :, 1] < 170) & (blank[:, :, 0] < 140)
        green = (blank[:, :, 1] > 140) & (blank[:, :, 1] > blank[:, :, 2] + 20) & (blank[:, :, 1] > blank[:, :, 0] + 20)
        self.assertGreater(int(orange.sum()), 30)
        self.assertGreater(int(green.sum()), 30)

    def test_arms_and_legs_follow_the_pose(self) -> None:
        avatar = DanceAvatar()
        try:
            down = np.zeros((360, 640, 3), dtype=np.uint8)
            up = down.copy()
            avatar.draw(down, _pose(arm_y=280, leg_y=330))
            avatar.draw(up, _pose(arm_y=40, leg_y=160))
        finally:
            avatar.close()
        changed = np.any(up != down, axis=2)
        up_only = changed & np.any(up != 0, axis=2)
        down_only = changed & np.any(down != 0, axis=2)
        self.assertGreater(int(up_only.sum()), 20)
        self.assertGreater(int(down_only.sum()), 20)
        self.assertLess(_mean_y(up_only), _mean_y(down_only))


class DiscoDancerCutTest(unittest.TestCase):
    def test_cut_covers_neck_shoulders_elbows_and_knees(self) -> None:
        dancer = DiscoDancer()
        self.assertEqual(
            set(dancer.parts),
            {
                "head",
                "torso",
                "upper_left",
                "lower_left",
                "hand_left",
                "upper_right",
                "lower_right",
                "hand_right",
                "thigh_left",
                "shin_left",
                "thigh_right",
                "shin_right",
            },
        )
        for name, part in dancer.parts.items():
            minimum = 80 if name.startswith("hand_") else 800
            self.assertGreater(part.area, minimum)

    def test_wrist_bends_without_moving_the_forearm(self) -> None:
        dancer = DiscoDancer()
        straight = _pose(arm_y=280, leg_y=330)
        bent = _pose(arm_y=280, leg_y=330)
        points = list(bent.points)
        points[L_INDEX] = (120, 240)
        points[R_INDEX] = (400, 240)
        bent = BodyPose(bent.width, bent.height, tuple(points))
        left = np.zeros((360, 640, 3), np.uint8)
        right = left.copy()
        dancer.draw(left, straight)
        dancer.draw(right, bent)
        self.assertGreater(int(np.any(left != right, axis=2).sum()), 20)

    def test_left_leg_rests_parallel_to_the_right(self) -> None:
        dancer = DiscoDancer()
        left = dancer.parts["shin_left"].rest
        right = dancer.parts["shin_right"].rest
        self.assertAlmostEqual(left, -right, delta=1.0)
        self.assertAlmostEqual(dancer.parts["thigh_left"].rest, -dancer.parts["thigh_right"].rest, delta=1.0)

    def test_head_tilt_does_not_slide_the_dancer(self) -> None:
        centered = _torso(0, ((200, 70), (280, 130)))
        level = _torso(0, ((200, 80), (280, 80)))
        stepped = _torso(80, ((200, 80), (280, 80)))
        shift = SilhouetteShift()
        self.assertIsInstance(shift, HeadTilt)
        self.assertNotEqual(HeadTilt().offset_x(centered, 640), 0)
        self.assertEqual(shift.offset_x(centered, 640), 0)
        self.assertEqual(shift.offset_x(level, 640), 0)
        self.assertEqual(shift.offset_x(stepped, 640), 80)
        dancer = DiscoDancer()
        still = np.zeros((360, 640, 3), np.uint8)
        tilted = still.copy()
        moved = still.copy()
        dancer.draw(still, level)
        dancer.draw(tilted, centered)
        dancer.draw(moved, stepped)
        self.assertAlmostEqual(_green_x(tilted), _green_x(still), delta=1.0)
        self.assertGreater(_green_x(moved), _green_x(still) + 40)

    def test_clockwise_turn_sends_an_upward_point_to_the_right(self) -> None:
        moved = turn((0.0, -10.0), (0.0, 0.0), 90.0)
        self.assertAlmostEqual(moved[0], 10.0, places=4)
        self.assertAlmostEqual(moved[1], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
