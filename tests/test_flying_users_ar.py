"""Headless tests for AR flying users (no camera window)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gamConsole.flying_users_ar import (  # noqa: E402
    ArmThrowDetector,
    FlyingUser,
    FlyingUsersARGame,
    ThrowEvent,
    ThrownBurger,
    blit_circle,
    make_burger_sprite,
    make_user_avatar,
)
from gamConsole.games_registry import games_for_menu  # noqa: E402


class FlyingUserPhysicsTest(unittest.TestCase):
    def _user(self, z: float = 1.0) -> FlyingUser:
        return FlyingUser(0.0, 0.0, z, 0.05, 1, (200, 80, 80), None)

    def test_grows_as_user_flies_closer(self) -> None:
        far = self._user(1.0)
        near = self._user(0.25)
        _fx, _fy, r_far = far.project(640, 480)
        _nx, _ny, r_near = near.project(640, 480)
        self.assertGreater(r_near, r_far)

    def test_tick_moves_toward_player(self) -> None:
        user = self._user(0.5)
        user.tick()
        self.assertLess(user.z, 0.5)

    def test_reached_player(self) -> None:
        self.assertTrue(self._user(0.05).reached_player)
        self.assertFalse(self._user(0.8).reached_player)


class ThrownBurgerPhysicsTest(unittest.TestCase):
    def test_flies_away_from_player(self) -> None:
        burger = ThrownBurger(0.0, 0.0, 0.2, 0.05, 0.0, 0.0)
        burger.tick()
        self.assertGreater(burger.z, 0.2)

    def test_shrinks_as_it_flies_away(self) -> None:
        near = ThrownBurger(0.0, 0.0, 0.2, 0.05, 0.0, 0.0)
        far = ThrownBurger(0.0, 0.0, 0.9, 0.05, 0.0, 0.0)
        _nx, _ny, r_near = near.project(640, 480)
        _fx, _fy, r_far = far.project(640, 480)
        self.assertGreater(r_near, r_far)

    def test_hits_overlapping_user(self) -> None:
        burger = ThrownBurger(0.0, 0.0, 0.4, 0.05, 0.0, 0.0)
        user = FlyingUser(0.0, 0.0, 0.4, 0.01, 1, (10, 10, 10), None)
        self.assertTrue(burger.hits(user, 640, 480))

    def test_misses_user_on_other_side(self) -> None:
        burger = ThrownBurger(-0.8, 0.0, 0.4, 0.05, 0.0, 0.0)
        user = FlyingUser(0.8, 0.0, 0.4, 0.01, 1, (10, 10, 10), None)
        self.assertFalse(burger.hits(user, 640, 480))

    def test_from_throw_starts_near_player(self) -> None:
        event = ThrowEvent(x_px=320, y_px=200, dx=-40.0, dy=10.0)
        burger = ThrownBurger.from_throw(event, 640, 480)
        self.assertLess(burger.z, 0.3)


class ArmThrowDetectorTest(unittest.TestCase):
    def test_detects_fast_move_in_right_arm_zone(self) -> None:
        detector = ArmThrowDetector()
        h, w = 240, 320
        first = np.zeros((h, w), dtype=np.uint8)
        second = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(first, (250, 70), 28, 255, -1)
        cv2.circle(second, (215, 95), 28, 255, -1)
        self.assertIsNone(detector.detect(first))
        event = detector.detect(second)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertGreater(event.x_px, int(0.60 * w))

    def test_ignores_still_mask(self) -> None:
        detector = ArmThrowDetector()
        mask = np.zeros((240, 320), dtype=np.uint8)
        self.assertIsNone(detector.detect(mask))
        self.assertIsNone(detector.detect(mask))


class FlyingUsersHelpersTest(unittest.TestCase):
    def test_avatar_is_bgr_image(self) -> None:
        img = make_user_avatar((10, 20, 30))
        self.assertEqual(img.shape, (128, 128, 3))

    def test_burger_sprite_is_bgr_image(self) -> None:
        img = make_burger_sprite()
        self.assertEqual(img.shape, (128, 128, 3))

    def test_blit_circle_writes_pixels(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        blit_circle(frame, make_user_avatar((0, 255, 0)), 100, 100, 30)
        self.assertGreater(int(frame.sum()), 0)


class FlyingUsersGameTest(unittest.TestCase):
    def test_reset_restores_lives(self) -> None:
        game = FlyingUsersARGame(camera=0)
        game.lives = 0
        game.score = 99
        game.game_over = True
        game.burgers.append(ThrownBurger(0.0, 0.0, 0.2, 0.05, 0.0, 0.0))
        game._reset()
        self.assertEqual(game.lives, 5)
        self.assertEqual(game.score, 0)
        self.assertFalse(game.game_over)
        self.assertEqual(len(game.burgers), 0)

    def test_arm_throw_spawns_burger(self) -> None:
        game = FlyingUsersARGame(camera=0)
        game.spawn_in = 999
        h, w = 480, 640
        still = np.full((h, w, 3), 40, dtype=np.uint8)
        game.on_frame(still)
        game.on_frame(still)
        first = still.copy()
        second = still.copy()
        cv2.circle(first, (500, 140), 50, (220, 220, 220), -1)
        cv2.circle(second, (430, 190), 50, (220, 220, 220), -1)
        game.on_frame(first)
        game.on_frame(second)
        self.assertGreaterEqual(len(game.burgers), 1)

    def test_on_frame_spawns_user_without_camera(self) -> None:
        game = FlyingUsersARGame(camera=0)
        game.spawn_in = 0
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        out = game.on_frame(frame)
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertGreaterEqual(len(game.users), 1)

    def test_registry_button_is_builtin(self) -> None:
        entry = next(g for g in games_for_menu() if g.id == "flying-users-ar")
        self.assertEqual(entry.launcher, "builtin")
        self.assertEqual(entry.mode_arg, "flying_users")
        self.assertEqual(entry.title, "Flyvende brugere (AR)")


if __name__ == "__main__":
    unittest.main()
