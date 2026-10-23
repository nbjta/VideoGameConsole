"""Headless tests for AR flying users (no camera window)."""

from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gamConsole.flying_users_ar import (  # noqa: E402
    CAR_HIT_POINTS,
    WristThrowDetector,
    CarHitFlash,
    CarHitSound,
    FlyingUsersARGame,
    ThrowEvent,
    ThrownBurger,
    blit_circle,
    kitten_cross_y,
    make_burger_sprite,
    make_kitten_sprite,
    motion_mask,
)
from gamConsole.pose_body_overlay import R_WRIST, BodyPose  # noqa: E402
from gamConsole.adventure_city import (  # noqa: E402
    ONCOMING_GAP_SECONDS,
    AdventureCityBackdrop,
    OncomingLaneCar,
    OncomingTraffic,
    StreetKitten,
    _road_half_at_y,
    _road_layout,
    draw_kitten_side,
)
from gamConsole.games_registry import games_for_menu  # noqa: E402
from gamConsole.round_score import ROUND_SECONDS, HighScoreList, ResultScreen, RoundScore  # noqa: E402
from gamConsole.player_cutout import PlayerCutout  # noqa: E402
from gamConsole.leader_back_view import player_car_box  # noqa: E402


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

    def test_only_one_kitten_is_created(self) -> None:
        city = AdventureCityBackdrop()
        self.assertEqual(len(city.kittens), 1)
        self.assertEqual(city.kittens[0].depth, 1.0)

    def test_foreground_body_is_one_fifth_of_the_frame_height(self) -> None:
        near = self._kitten()
        near.depth = 1.0
        far = self._kitten()
        far.depth = 0.0
        near_h = _drawn_kitten_height(near._body_scale(640, 480))
        far_h = _drawn_kitten_height(far._body_scale(640, 480))
        self.assertAlmostEqual(near_h, 480 / 5, delta=3)
        self.assertLess(far_h, near_h)

    def test_disappears_after_five_seconds(self) -> None:
        kitten = self._kitten()
        kitten.shown_at = 10.0
        self.assertFalse(kitten.expired(14.9))
        self.assertTrue(kitten.expired(15.0))


def _drawn_kitten_height(scale: float) -> int:
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_kitten_side(canvas, 320, 360, (40, 145, 230), (40, 45, 230), 0.0, scale=scale)
    rows = np.where(np.any(canvas > 10, axis=2))[0]
    return int(rows.max() - rows.min() + 1)


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

    def test_small_sideways_speed_is_not_forced_across_the_screen(self) -> None:
        event = ThrowEvent(x_px=500, y_px=80, dx=1.0, dy=10.0)
        burger = ThrownBurger.from_throw(event, 640, 480)
        self.assertGreater(burger.vx, 0.0)
        self.assertAlmostEqual(burger.vx / burger.vy, 0.1)

    def test_straight_down_stays_straight(self) -> None:
        event = ThrowEvent(x_px=500, y_px=80, dx=0.0, dy=10.0)
        burger = ThrownBurger.from_throw(event, 640, 480)
        self.assertEqual(burger.vx, 0.0)
        self.assertGreater(burger.vy, 0.0)

    def test_upward_throw_stays_upward(self) -> None:
        event = ThrowEvent(x_px=200, y_px=200, dx=3.0, dy=-6.0)
        burger = ThrownBurger.from_throw(event, 640, 480)
        self.assertLess(burger.vy, 0.0)
        self.assertAlmostEqual(burger.vx / burger.vy, 3.0 / -6.0)

    def test_direction_stays_constant_until_the_kitten_line(self) -> None:
        event = ThrowEvent(x_px=180, y_px=80, dx=4.0, dy=8.0)
        burger = ThrownBurger.from_throw(event, 640, 480)
        vx, vy = burger.vx, burger.vy
        cross_y = kitten_cross_y(480, 640)
        for _ in range(60):
            if burger.y >= cross_y:
                break
            burger.tick()
        self.assertEqual((burger.vx, burger.vy), (vx, vy))
        self.assertGreaterEqual(burger.y, cross_y)
        aimed_x = 180.0 + (vx / vy) * (burger.y - 80.0)
        self.assertAlmostEqual(burger.x, aimed_x)

    def test_slow_throw_is_scaled_onto_the_same_line(self) -> None:
        event = ThrowEvent(x_px=200, y_px=80, dx=0.2, dy=0.4)
        burger = ThrownBurger.from_throw(event, 640, 480)
        self.assertAlmostEqual(burger.vx / burger.vy, 0.5)
        cross_y = kitten_cross_y(480, 640)
        for _ in range(50):
            burger.tick()
        self.assertAlmostEqual(burger.y, cross_y)

    def test_starts_at_twice_the_old_radius(self) -> None:
        burger = ThrownBurger(100.0, 80.0, 0.0, 0.0)
        self.assertEqual(burger.radius(), 44)

    def test_kitten_cross_y_is_the_foreground_body_center(self) -> None:
        _horizon, road_top, _vx, _vy, _half_h, _half_n = _road_layout(480, 640)
        self.assertEqual(kitten_cross_y(480, 640), float(road_top) - 480 / 10.0)


def _wrist_pose(width: int, height: int, right: tuple[int, int]) -> BodyPose:
    points = [(-1, -1)] * 33
    points[R_WRIST] = right
    return BodyPose(width, height, tuple(points))


class WristThrowDetectorTest(unittest.TestCase):
    def test_uses_the_average_and_releases_at_the_wrist(self) -> None:
        detector = WristThrowDetector()
        points = [(400, 80), (410, 100), (420, 120), (430, 140), (440, 160)]
        event = None
        for point in points:
            event = detector.detect(_wrist_pose(640, 480, point))
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual((event.x_px, event.y_px), (440, 160))
        self.assertAlmostEqual(event.dx, 10.0)
        self.assertAlmostEqual(event.dy, 20.0)

    def test_a_hop_that_returns_does_not_throw(self) -> None:
        detector = WristThrowDetector()
        points = [(400, 100), (460, 40), (400, 100), (400, 100), (400, 100)]
        event = None
        for point in points:
            event = detector.detect(_wrist_pose(640, 480, point))
        self.assertIsNone(event)

    def test_one_frame_is_not_enough(self) -> None:
        detector = WristThrowDetector()
        self.assertIsNone(detector.detect(_wrist_pose(640, 480, (200, 80))))
        self.assertIsNone(detector.detect(_wrist_pose(640, 480, (280, 160))))

    def test_still_wrist_does_not_throw(self) -> None:
        detector = WristThrowDetector()
        event = None
        for _ in range(6):
            event = detector.detect(_wrist_pose(640, 480, (300, 120)))
        self.assertIsNone(event)


class MotionMaskTest(unittest.TestCase):
    def test_size_change_does_not_crash(self) -> None:
        small = np.full((60, 80, 3), 40, dtype=np.uint8)
        large = np.full((120, 160, 3), 40, dtype=np.uint8)
        _mask, prev = motion_mask(small, None)
        mask, prev2 = motion_mask(large, prev)
        self.assertEqual(mask.shape, (120, 160))
        self.assertEqual(prev2.shape, (120, 160))


class MenuThumbnailTest(unittest.TestCase):
    def test_thumbs_match_game_backgrounds(self) -> None:
        from gamConsole.game_thumbnails import get_game_thumbnail

        for game_id in ("gurli-gris", "flying-users-ar", "webcam-3d"):
            thumb = get_game_thumbnail(game_id, 96)
            self.assertEqual(thumb.shape, (96, 96, 3))
            self.assertGreater(int(thumb.mean()), 8)


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

    def test_three_minutes_and_no_lives_use_the_shared_score_screen(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            game = FlyingUsersARGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            self.assertIsInstance(game.match, RoundScore)
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
            game.on_frame(frame, now=1000)
            self.assertIsNone(game.match.result)
            self.assertFalse(game.accepts_name_key(ord("h")))
            game.score = 11
            game.on_frame(frame, now=1000 + ROUND_SECONDS)
            assert game.match.result is not None
            self.assertIsInstance(game.match.result, ResultScreen)
            self.assertEqual(game.match.result.score, 11)
            self.assertTrue(game.accepts_name_key(ord("h")))
            game.on_key(ord("h"))
            assert game.match.result.entry is not None
            self.assertEqual(game.match.result.entry.name, "h")
            assert game.match.result.button.rect is not None
            x1, y1, x2, y2 = game.match.result.button.rect
            game.on_click((x1 + x2) // 2, (y1 + y2) // 2)
            self.assertIsNone(game.match.result)
            self.assertEqual(game.score, 0)
            self.assertEqual(game.lives, 5)
            game.on_frame(frame, now=2000)
            game.lives = 0
            game.on_frame(frame, now=2001)
            assert game.match.result is not None
            self.assertFalse(game.match.round.over(2001))
            self.assertEqual(game.match.result.score, game.score)

    def test_motion_blob_does_not_spawn_burger(self) -> None:
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
        self.assertEqual(len(game.burgers), 0)

    def test_wrist_throw_spawns_burger_on_the_wrist_line(self) -> None:
        game = FlyingUsersARGame(camera=0)
        points = [(400, 80), (410, 100), (420, 120), (430, 140), (440, 160)]
        for point in points:
            game._release_throw(_wrist_pose(640, 480, point), 640, 480)
        self.assertEqual(len(game.burgers), 1)
        burger = game.burgers[0]
        self.assertEqual((burger.x, burger.y), (440.0, 160.0))
        self.assertAlmostEqual(burger.vx / burger.vy, 0.5)
        game._release_throw(_wrist_pose(640, 480, (450, 180)), 640, 480)
        self.assertEqual(len(game.burgers), 1)

    def test_kitten_is_replaced_after_five_seconds_without_losing_a_life(self) -> None:
        game = FlyingUsersARGame(camera=0)
        self.assertEqual(len(game.kittens), 1)
        first = game.kittens[0]
        now = 50.0
        first.shown_at = now - 5.0
        lives = game.lives
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        game.on_frame(frame, now=now)
        self.assertEqual(len(game.kittens), 1)
        self.assertIsNot(game.kittens[0], first)
        self.assertEqual(game.lives, lives)

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


class OncomingCarTest(unittest.TestCase):
    def test_starts_in_the_middle_of_the_left_lane_at_the_horizon(self) -> None:
        car = OncomingLaneCar((40, 90, 230))
        center_x, top, width = car.pose(640, 480)
        horizon, road_top, vx, _vy, half_h, half_n = _road_layout(480, 640)
        vanish_y = horizon + 8
        half = _road_half_at_y(float(vanish_y), horizon, road_top, half_h, half_n)
        car_h = max(18.0, width * 0.74)
        self.assertEqual(center_x, int(round(vx - half / 2.0)))
        self.assertLess(abs((top + car_h / 2.0) - vanish_y), 1.5)
        self.assertLess(center_x, vx)

    def test_grows_as_it_approaches(self) -> None:
        car = OncomingLaneCar((40, 90, 230))
        _x0, top0, width0 = car.pose(640, 480)
        car.progress = 0.75
        _x1, top1, width1 = car.pose(640, 480)
        self.assertGreater(width1, width0)
        self.assertGreater(top1, top0)

    def test_only_one_car_then_a_different_color_after_the_gap(self) -> None:
        traffic = OncomingTraffic()
        self.assertTrue(traffic.active)
        traffic.tick(0.0)
        first = traffic.car.color
        traffic.car.progress = 0.99
        traffic.tick(0.05)
        self.assertFalse(traffic.active)
        self.assertEqual(traffic.car.color, first)
        traffic.tick(0.05 + ONCOMING_GAP_SECONDS - 0.1)
        self.assertFalse(traffic.active)
        traffic.tick(0.05 + ONCOMING_GAP_SECONDS)
        self.assertTrue(traffic.active)
        self.assertEqual(traffic.car.progress, 0.0)
        self.assertNotEqual(traffic.car.color, first)
        self.assertEqual(traffic.car.lane, 1)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        hidden = OncomingTraffic()
        hidden.active = False
        hidden.draw(blank)
        self.assertEqual(int(blank.sum()), 0)

    def test_right_lane_is_the_mirror_of_the_left_and_never_both_at_once(self) -> None:
        left = OncomingLaneCar((40, 90, 230), 0)
        right = OncomingLaneCar((40, 200, 90), 1)
        left.progress = right.progress = 0.4
        lx, _top, _width = left.pose(640, 480)
        rx, _top, _width = right.pose(640, 480)
        _horizon, _road_top, vx, _vy, _half_h, _half_n = _road_layout(480, 640)
        self.assertLess(lx, vx)
        self.assertGreater(rx, vx)
        self.assertEqual(vx - lx, rx - vx)
        traffic = OncomingTraffic()
        self.assertEqual(traffic.car.lane, 0)
        self.assertTrue(traffic.active)
        traffic.car.progress = 1.0
        traffic.tick(1.0)
        self.assertFalse(traffic.active)
        traffic.tick(1.0 + ONCOMING_GAP_SECONDS)
        self.assertTrue(traffic.active)
        self.assertEqual(traffic.car.lane, 1)

    def test_centered_user_is_not_hit_and_the_lane_is(self) -> None:
        width, height = 960, 540
        _horizon, _road_top, vx, _vy, _half_h, _half_n = _road_layout(height, width)
        scale = (height / 480.0) * 1.05
        center = player_car_box(width, height, vx, scale)
        car = OncomingLaneCar((40, 90, 230), 0)
        lane_hits = 0
        for step in range(101):
            car.progress = step / 100.0
            car.struck = False
            self.assertFalse(car.overlaps(center, width, height))
            lane_x, _top, _car_w = car.pose(width, height)
            lane_box = player_car_box(width, height, lane_x, scale)
            if car.overlaps(lane_box, width, height):
                lane_hits += 1
                self.assertTrue(car.strike(lane_box, width, height))
                self.assertFalse(car.strike(lane_box, width, height))
        self.assertGreater(lane_hits, 0)
        right = OncomingLaneCar((40, 200, 90), 1)
        right_hits = 0
        for step in range(101):
            right.progress = step / 100.0
            right.struck = False
            self.assertFalse(right.overlaps(center, width, height))
            lane_x, _top, _car_w = right.pose(width, height)
            if right.overlaps(player_car_box(width, height, lane_x, scale), width, height):
                right_hits += 1
        self.assertGreater(right_hits, 0)


class CarHitTest(unittest.TestCase):
    def _game(self) -> FlyingUsersARGame:
        game = FlyingUsersARGame(camera=0)
        self.addCleanup(game.player.close)
        return game

    def test_overlap_subtracts_one_point_once_and_shows_the_burst(self) -> None:
        game = self._game()
        game.hit_sound.play = lambda: setattr(game, "sound_played", True)
        game.sound_played = False
        game.score = 5
        width, height = 960, 540
        car = game.city.traffic.car
        scale = game.player.leader_scale(height)
        box = None
        for step in range(101):
            car.progress = step / 100.0
            car.struck = False
            lane_x, _top, _car_w = car.pose(width, height)
            candidate = player_car_box(width, height, lane_x, scale)
            if car.overlaps(candidate, width, height):
                box = candidate
                break
        self.assertIsNotNone(box)
        assert box is not None
        car.struck = False
        game.player.car_box = lambda _width, _height: box
        game._apply_car_hit(width, height)
        self.assertEqual(game.score, 5 + CAR_HIT_POINTS)
        self.assertEqual(CAR_HIT_POINTS, -1)
        self.assertTrue(game.sound_played)
        gains = game._score_view().gains
        self.assertEqual(len(gains), 1)
        self.assertEqual(gains[0].label(), "-1")
        self.assertTrue(game.hit_flash.visible())
        game.sound_played = False
        game._apply_car_hit(width, height)
        self.assertEqual(game.score, 4)
        self.assertFalse(game.sound_played)
        self.assertEqual(len(game._score_view().gains), 1)

    def test_waiting_car_does_not_change_the_score(self) -> None:
        game = self._game()
        game.hit_sound.play = lambda: setattr(game, "sound_played", True)
        game.sound_played = False
        game.score = 5
        game.city.traffic.active = False
        game._apply_car_hit(960, 540)
        self.assertEqual(game.score, 5)
        self.assertFalse(game.sound_played)

    def test_burst_keeps_the_background_around_it(self) -> None:
        flash = CarHitFlash()
        flash.trigger(200, 200)
        canvas = np.full((400, 400, 3), (10, 20, 30), dtype=np.uint8)
        flash.draw(canvas)
        changed = np.any(canvas != (10, 20, 30), axis=2)
        self.assertGreater(int(np.count_nonzero(changed)), 100)
        self.assertTrue(np.array_equal(canvas[0, 0], np.array((10, 20, 30), dtype=np.uint8)))

    def test_hit_sound_is_a_short_wav(self) -> None:
        sound = CarHitSound()
        self.assertTrue(sound._ensure())
        with wave.open(str(sound._path), "rb") as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertGreater(handle.getnframes(), 1000)

    def test_score_is_drawn_at_the_top(self) -> None:
        game = self._game()
        game.score = 7
        frame = np.zeros((120, 320, 3), dtype=np.uint8)
        game._draw_hud(frame)
        self.assertGreater(int(frame[0:40, 100:220].sum()), 0)


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

