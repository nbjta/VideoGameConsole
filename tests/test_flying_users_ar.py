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
    FlyingUsersARGame,
    ThrowEvent,
    ThrownBurger,
    blit_circle,
    make_burger_sprite,
    make_kitten_sprite,
    motion_mask,
)
from gamConsole.adventure_city import AdventureCityBackdrop, StreetKitten  # noqa: E402
from gamConsole.overhead_branches import OverheadBranch  # noqa: E402
from gamConsole.games_registry import games_for_menu  # noqa: E402
from gamConsole.player_cutout import PlayerCutout  # noqa: E402


class OverheadBranchTest(unittest.TestCase):
    def test_sun_branch_is_brief(self) -> None:
        b = OverheadBranch(0.0, 0.0, enter_from_left=True)
        b.start_t = 10.0
        self.assertFalse(b.active(9.0))
        self.assertTrue(b.active(10.0))
        self.assertTrue(b.active(10.0 + OverheadBranch.DURATION - 1))
        self.assertFalse(b.active(10.0 + OverheadBranch.DURATION))
        self.assertTrue(b.passed(10.0 + OverheadBranch.DURATION, 640))


class StreetKittenMotionTest(unittest.TestCase):
    def _kitten(self) -> StreetKitten:
        return StreetKitten(0.0, 2.0, 0, 1, (40, 145, 230), (40, 45, 230))

    def test_enters_from_left_toward_center(self) -> None:
        kitten = StreetKitten(0.0, 2.0, 0, 1, (40, 145, 230), (40, 45, 230), enter_from_left=True)
        x0 = kitten.x_on(0.0, 320)
        x1 = kitten.x_on(20.0, 320)
        self.assertGreater(x1, x0)

    def test_enters_from_right_toward_center(self) -> None:
        kitten = StreetKitten(0.0, 2.0, 0, 1, (40, 145, 230), (40, 45, 230), enter_from_left=False)
        x0 = kitten.x_on(0.0, 320)
        x1 = kitten.x_on(20.0, 320)
        self.assertLess(x1, x0)

    def test_is_street_actor(self) -> None:
        from gamConsole.adventure_city import StreetActor

        self.assertIsInstance(self._kitten(), StreetActor)


class ThrownBurgerPhysicsTest(unittest.TestCase):
    def test_flies_toward_street(self) -> None:
        burger = ThrownBurger(100.0, 80.0, 2.0, 8.0)
        burger.tick()
        self.assertGreater(burger.y, 80.0)

    def test_shrinks_with_age(self) -> None:
        burger = ThrownBurger(100.0, 80.0, 0.0, 0.0)
        r0 = burger.radius()
        burger.age = 20
        self.assertLess(burger.radius(), r0)

    def test_hits_overlapping_kitten(self) -> None:
        burger = ThrownBurger(100.0, 200.0, 0.0, 0.0)
        self.assertTrue(burger.hits_xy(100, 200, 20))

    def test_misses_kitten_on_other_side(self) -> None:
        burger = ThrownBurger(40.0, 200.0, 0.0, 0.0)
        self.assertFalse(burger.hits_xy(400, 200, 20))

    def test_from_throw_moves_down(self) -> None:
        event = ThrowEvent(x_px=320, y_px=80, dx=-40.0, dy=10.0)
        burger = ThrownBurger.from_throw(event, 640, 480)
        self.assertGreater(burger.vy, 0)


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


class MotionMaskTest(unittest.TestCase):
    def test_size_change_does_not_crash(self) -> None:
        small = np.full((60, 80, 3), 40, dtype=np.uint8)
        large = np.full((120, 160, 3), 40, dtype=np.uint8)
        _mask, prev = motion_mask(small, None)
        mask, prev2 = motion_mask(large, prev)
        self.assertEqual(mask.shape, (120, 160))
        self.assertEqual(prev2.shape, (120, 160))


class FlyingUsersHelpersTest(unittest.TestCase):
    def test_kitten_sprite_is_bgr_image(self) -> None:
        img = make_kitten_sprite((40, 145, 230), (40, 45, 230))
        self.assertEqual(len(img.shape), 3)
        self.assertEqual(img.shape[2], 3)

    def test_burger_sprite_is_bgr_image(self) -> None:
        img = make_burger_sprite()
        self.assertEqual(img.shape, (128, 128, 3))

    def test_blit_circle_writes_pixels(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        blit_circle(frame, make_kitten_sprite((40, 145, 230), (40, 45, 230)), 100, 100, 30)
        self.assertGreater(int(frame.sum()), 0)


class FlyingUsersGameTest(unittest.TestCase):
    def test_reset_restores_lives(self) -> None:
        game = FlyingUsersARGame(camera=0)
        game.lives = 0
        game.score = 99
        game.game_over = True
        game.burgers.append(ThrownBurger(10.0, 10.0, 1.0, 4.0))
        game._reset()
        self.assertEqual(game.lives, 5)
        self.assertEqual(game.score, 0)
        self.assertFalse(game.game_over)
        self.assertEqual(len(game.burgers), 0)

    def test_arm_throw_spawns_burger(self) -> None:
        game = FlyingUsersARGame(camera=0)
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

    def test_on_frame_has_street_kittens(self) -> None:
        game = FlyingUsersARGame(camera=0)
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        out = game.on_frame(frame)
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertGreaterEqual(len(game.kittens), 1)
        self.assertIsInstance(game.kittens[0], StreetKitten)

    def test_registry_button_is_builtin(self) -> None:
        entry = next(g for g in games_for_menu() if g.id == "flying-users-ar")
        self.assertEqual(entry.launcher, "builtin")
        self.assertEqual(entry.mode_arg, "flying_users")
        self.assertEqual(entry.title, "Humdingers killinger (AR)")

    def test_on_frame_uses_moving_city_background(self) -> None:
        game = FlyingUsersARGame(camera=0)
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        a = game.on_frame(frame.copy())
        b = game.on_frame(frame.copy())
        self.assertEqual(a.shape, (240, 320, 3))
        self.assertFalse(np.array_equal(a, b))


class AdventureCityTest(unittest.TestCase):
    def test_render_size_and_motion(self) -> None:
        city = AdventureCityBackdrop()
        first = city.render(320, 240)
        city.tick()
        city.tick()
        second = city.render(320, 240)
        self.assertEqual(first.shape, (240, 320, 3))
        self.assertGreater(int(first.sum()), 0)
        self.assertFalse(np.array_equal(first, second))
        self.assertGreaterEqual(len(city.kittens), 1)
        x0 = city.kittens[0].x_on(city.t, 320)
        city.tick()
        city.tick()
        x1 = city.kittens[0].x_on(city.t, 320)
        self.assertNotEqual(x0, x1)


class PlayerCutoutTest(unittest.TestCase):
    def test_composite_keeps_person_pixels(self) -> None:
        cut = PlayerCutout()
        webcam = np.zeros((120, 160, 3), dtype=np.uint8)
        webcam[20:110, 40:120] = (0, 0, 255)
        background = np.full((120, 160, 3), (255, 0, 0), dtype=np.uint8)
        out = cut.composite(webcam, background)
        self.assertEqual(out.shape, webcam.shape)
        self.assertGreater(int(out[:, :, 2].sum()), 0)
        self.assertGreater(int(out[:, :, 0].sum()), 0)

    def test_corners_stay_background_not_room(self) -> None:
        cut = PlayerCutout()
        webcam = np.full((120, 160, 3), 255, dtype=np.uint8)
        webcam[25:100, 55:105] = (0, 0, 255)
        background = np.zeros((120, 160, 3), dtype=np.uint8)
        background[:] = (255, 0, 0)
        out = cut.composite(webcam, background)
        corner = out[0:12, 0:12]
        self.assertGreater(float(corner[:, :, 0].mean()), float(corner[:, :, 2].mean()))


if __name__ == "__main__":
    unittest.main()

