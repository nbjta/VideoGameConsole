#9
"""Gurli Gris i spilmenuen, uden at åbne kamera eller spilvindue."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gamConsole.flying_users_ar import WebcamARGame  # noqa: E402
from gamConsole.games_registry import games_for_menu  # noqa: E402
from gamConsole.gurli_gris import (  # noqa: E402
    BACKGROUND_PATH,
    BRANCH_PATH,
    FIGURE_PATH,
    POINT_BLUE,
    POINT_VALUES,
    WAVE_SECONDS,
    BranchSwing,
    BubbleRain,
    GurliGrisBackdrop,
    GurliGrisFigure,
    GurliGrisGame,
    SoapBubble,
    _hold_branch,
)
from gamConsole.menu_opencv import CANVAS_H, CANVAS_W, QUIT_BAR_H, _compose_menu  # noqa: E402
from gamConsole.round_score import (  # noqa: E402
    HIGHSCORE_SHOWN,
    NAME_MAX,
    ROUND_SECONDS,
    HighScoreList,
    ResultScreen,
    RoundScore,
    TimedRound,
    clock_text,
)
from gamConsole.opencv_ui import game_point_from_canvas  # noqa: E402
from gamConsole.pose_body_overlay import (  # noqa: E402
    BONE,
    BodyPose,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_SHOULDER,
    R_WRIST,
)


class GurliGrisMenuTest(unittest.TestCase):
    def test_gurli_and_humdingers_first_in_menu(self) -> None:
        from gamConsole.games_registry import games_for_menu

        menu = games_for_menu()
        self.assertGreaterEqual(len(menu), 2)
        self.assertEqual(menu[0].id, "gurli-gris")
        self.assertEqual(menu[1].id, "flying-users-ar")

    def test_registry_button_is_builtin(self) -> None:
        entry = next(g for g in games_for_menu() if g.id == "gurli-gris")
        self.assertEqual(entry.launcher, "builtin")
        self.assertEqual(entry.mode_arg, "gurli_gris")
        self.assertEqual(entry.title, "Gurli Gris")

    def test_selected_card_stays_above_quit_bar(self) -> None:
        games = games_for_menu()
        selected = next(i for i, game in enumerate(games) if game.id == "gurli-gris")
        webcam = np.zeros((480, 640, 3), dtype=np.uint8)
        _canvas, layout = _compose_menu(webcam, games, selected, None, False, 0)
        card = next(box for box in layout.card_boxes if box.game_index == selected)
        self.assertLess(card.y2, CANVAS_H - QUIT_BAR_H)
        self.assertIn(selected, {box.game_index for box in layout.card_boxes})


class GurliGrisBackdropTest(unittest.TestCase):
    def test_background_file_is_the_supplied_landscape(self) -> None:
        self.assertTrue(BACKGROUND_PATH.is_file())
        src = cv2.imread(str(BACKGROUND_PATH), cv2.IMREAD_COLOR)
        self.assertIsNotNone(src)
        assert src is not None
        self.assertEqual(src.shape[0], 148)
        self.assertEqual(src.shape[1], 270)

    def test_render_keeps_sky_and_grass(self) -> None:
        backdrop = GurliGrisBackdrop()
        frame = backdrop.render(320, 180)
        src = cv2.imread(str(backdrop.image_path), cv2.IMREAD_COLOR)
        self.assertEqual(frame.shape, (180, 320, 3))
        sky_delta = np.abs(src[:12].mean(axis=(0, 1)) - frame[:12].mean(axis=(0, 1)))
        grass_delta = np.abs(src[-12:].mean(axis=(0, 1)) - frame[-12:].mean(axis=(0, 1)))
        self.assertLess(float(sky_delta.max()), 25)
        self.assertLess(float(grass_delta.max()), 25)


def _pose(left_wrist: tuple, right_wrist: tuple) -> BodyPose:
    points = [(-1, -1)] * 33
    points[L_SHOULDER] = (120, 140)
    points[R_SHOULDER] = (200, 140)
    points[L_WRIST] = left_wrist
    points[R_WRIST] = right_wrist
    return BodyPose(320, 360, tuple(points))


def _pink(image: np.ndarray) -> np.ndarray:
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (red > 180) & (green > 80) & (green < 230) & (blue > 80) & (red > blue)


def _leaves(image: np.ndarray) -> np.ndarray:
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (green > 120) & (green > blue + 40) & (green > red)


class GurliGrisGameTest(unittest.TestCase):
    def test_game_is_webcam_ar(self) -> None:
        self.assertIsInstance(GurliGrisGame(camera=0), WebcamARGame)

    def test_figure_file_is_the_supplied_image(self) -> None:
        self.assertTrue(FIGURE_PATH.is_file())
        src = cv2.imread(str(FIGURE_PATH), cv2.IMREAD_COLOR)
        self.assertIsNotNone(src)
        assert src is not None
        self.assertEqual(src.shape[0], 414)
        self.assertEqual(src.shape[1], 738)

    def test_on_frame_shows_figure_on_landscape_not_webcam(self) -> None:
        game = GurliGrisGame(camera=0)
        try:
            webcam = np.full((360, 640, 3), (0, 0, 255), dtype=np.uint8)
            out = game.on_frame(webcam)
        finally:
            game.close()
        self.assertEqual(out.shape, (360, 640, 3))
        sky = out[0:8, 0:8]
        grass = out[-8:, 0:8]
        self.assertGreater(float(sky[:, :, 0].mean()), float(sky[:, :, 2].mean()))
        self.assertGreater(float(grass[:, :, 1].mean()), float(grass[:, :, 0].mean()))
        self.assertEqual(int(np.all(out == webcam, axis=2).sum()), 0)
        self.assertGreater(int(_pink(out).sum()), 200)

    def test_point_silhouette_is_drawn_over_the_scene(self) -> None:
        game = GurliGrisGame(camera=0)
        try:
            points = [(-1, -1)] * 33
            points[NOSE] = (40, 30)
            game.player._last_pose = BodyPose(640, 360, tuple(points))
            webcam = np.full((360, 640, 3), (20, 20, 20), dtype=np.uint8)
            out = game.player.composite(webcam, game.backdrop.render(640, 360))
        finally:
            game.close()
        self.assertEqual(tuple(int(v) for v in out[30, 40]), BONE)

    def test_arms_follow_the_point_silhouette(self) -> None:
        figure = GurliGrisFigure()
        canvas = np.full((360, 640, 3), (255, 180, 80), dtype=np.uint8)
        down = canvas.copy()
        up = canvas.copy()
        figure.draw(down, _pose((120, 260), (200, 260)))
        figure.draw(up, _pose((120, 40), (200, 40)))
        only_up = _pink(up) & ~_pink(down)
        only_down = _pink(down) & ~_pink(up)
        self.assertGreater(int(only_up.sum()), 20)
        self.assertGreater(int(only_down.sum()), 20)
        #9 Bladene følger armen: oppe når armen peger op, nede når den peger ned.
        leaves_up = _leaves(up)
        leaves_down = _leaves(down)
        self.assertGreater(int(leaves_up.sum()), 10)
        self.assertGreater(int(leaves_down.sum()), 10)
        self.assertLess(float(np.where(leaves_up)[0].mean()), float(np.where(leaves_down)[0].mean()))

    def test_both_arms_hold_the_branch_and_one_is_mirrored(self) -> None:
        self.assertTrue(BRANCH_PATH.is_file())
        arm = np.zeros((40, 16, 3), np.uint8)
        arm[10:40, 4:12] = (180, 160, 220)
        alpha = np.zeros((40, 16), np.float32)
        alpha[10:40, 4:12] = 1.0
        arm[39, 8] = (0, 0, 255)
        branch = np.zeros((20, 10, 3), np.uint8)
        branch_alpha = np.zeros((20, 10), np.float32)
        branch[0:6, 6:10] = (20, 220, 40)
        branch_alpha[0:6, 6:10] = 1.0
        branch[16:20, 0:4] = (30, 50, 90)
        branch_alpha[16:20, 0:4] = 1.0
        held, held_a, *_rest = _hold_branch(arm, alpha, branch, branch_alpha)
        mirrored, mirrored_a, *_rest = _hold_branch(arm, alpha, cv2.flip(branch, 1), cv2.flip(branch_alpha, 1))
        self.assertEqual(tuple(int(v) for v in held[held.shape[0] - 1, held.shape[1] // 2]), (0, 0, 255))
        self.assertEqual(tuple(int(v) for v in mirrored[mirrored.shape[0] - 1, mirrored.shape[1] // 2]), (0, 0, 255))

        def leaf_side(image: np.ndarray, mask: np.ndarray) -> float:
            leaves = (image[:, :, 1] > 180) & (mask > 0.5)
            xs = np.where(leaves)[1]
            self.assertGreater(len(xs), 0)
            return float(xs.mean()) - image.shape[1] / 2.0

        self.assertLess(leaf_side(held, held_a) * leaf_side(mirrored, mirrored_a), 0)
        canvas = np.full((360, 640, 3), (255, 180, 80), dtype=np.uint8)
        GurliGrisFigure().draw(canvas, _pose((40, 140), (280, 140)))
        leaves = (canvas[:, :, 1] > 120) & (canvas[:, :, 1] > canvas[:, :, 0] + 40) & (canvas[:, :, 1] > canvas[:, :, 2])
        self.assertGreater(int(leaves[:, :320].sum()), 10)
        self.assertGreater(int(leaves[:, 320:].sum()), 10)


def _rain_with_bubble(value: int) -> BubbleRain:
    rain = BubbleRain(np.random.default_rng(0))
    rain._spawn_in = 50
    rain.bubbles.append(SoapBubble(50, 40, value, 18, 0))
    return rain


def _wave(rain: BubbleRain, seconds: float, degrees: float = 8.0) -> None:
    rain.tick(200, 160, [BranchSwing((50, 80), (50, 10), 12, 0.0)], 0.0)
    step = 0.1
    steps = int(round(seconds / step))
    for index in range(1, steps + 1):
        rain.tick(200, 160, [BranchSwing((50, 80), (50, 10), 12, degrees * index)], index * step)
    assert WAVE_SECONDS == 0.5


class SoapBubbleTest(unittest.TestCase):
    def test_named_point_values_are_in_the_bubbles(self) -> None:
        self.assertIn(4, POINT_VALUES)
        self.assertIn(1, POINT_VALUES)
        self.assertIn(-1, POINT_VALUES)
        self.assertEqual(SoapBubble(0, 0, 4, 10, 1).label(), "+4")
        self.assertEqual(SoapBubble(0, 0, 1, 10, 1).label(), "+1")
        self.assertEqual(SoapBubble(0, 0, -1, 10, 1).label(), "-1")

    def test_bubble_falls_downward(self) -> None:
        rain = BubbleRain(np.random.default_rng(0))
        rain._spawn_in = 50
        rain.bubbles.append(SoapBubble(80, 10, 4, 20, 6))
        rain.tick(200, 120, [], 0.0)
        self.assertEqual(len(rain.bubbles), 1)
        self.assertAlmostEqual(rain.bubbles[0].y, 16)
        self.assertEqual(rain.score, 0)

    def test_waving_a_branch_for_half_a_second_bursts_the_bubble(self) -> None:
        rain = _rain_with_bubble(4)
        _wave(rain, 0.5)
        self.assertEqual(rain.score, 4)
        self.assertEqual(rain.bubbles, [])
        self.assertEqual(len(rain.gains), 1)
        self.assertEqual(rain.gains[0].label(), "+4")
        self.assertLess(rain.gains[0].y, rain.gains[0].origin_y)

    def test_holding_still_on_a_bubble_does_not_score(self) -> None:
        rain = _rain_with_bubble(4)
        _wave(rain, 0.5, degrees=0.0)
        self.assertEqual(rain.score, 0)
        self.assertEqual(len(rain.bubbles), 1)

    def test_less_than_half_a_second_of_waving_is_not_enough(self) -> None:
        rain = _rain_with_bubble(4)
        _wave(rain, 0.4)
        self.assertEqual(rain.score, 0)
        self.assertAlmostEqual(rain.bubbles[0].wave_time, 0.4, places=5)

    def test_bubble_stays_still_while_the_timer_runs_and_falls_when_the_branch_leaves(self) -> None:
        rain = BubbleRain(np.random.default_rng(0))
        rain._spawn_in = 50
        rain.bubbles.append(SoapBubble(50, 40, 4, 18, 6))
        rain._last_now = 0.0
        rain._prev_angles = [0.0]

        def swing(angle: float) -> BranchSwing:
            return BranchSwing((50, 80), (50, 10), 12, angle)

        rain.tick(200, 160, [swing(8.0)], 0.1)
        self.assertAlmostEqual(rain.bubbles[0].y, 40.0)
        self.assertAlmostEqual(rain.bubbles[0].wave_time, 0.1)
        rain.tick(200, 160, [swing(8.0)], 0.2)
        self.assertAlmostEqual(rain.bubbles[0].y, 40.0)
        self.assertAlmostEqual(rain.bubbles[0].wave_time, 0.1)
        rain.tick(200, 160, [BranchSwing((400, 80), (400, 10), 12, 90.0)], 0.3)
        self.assertEqual(rain.bubbles[0].wave_time, 0.0)
        self.assertAlmostEqual(rain.bubbles[0].y, 46.0)

    def test_leaving_the_bubble_resets_the_wave(self) -> None:
        rain = _rain_with_bubble(4)
        _wave(rain, 0.4)
        away = BranchSwing((400, 80), (400, 10), 12, 90)
        rain.tick(200, 160, [away], 0.5)
        self.assertEqual(rain.bubbles[0].wave_time, 0.0)
        self.assertEqual(rain.score, 0)

    def test_negative_number_lowers_the_score(self) -> None:
        rain = _rain_with_bubble(-1)
        rain.score = 3
        _wave(rain, 0.5)
        self.assertEqual(rain.score, 2)
        self.assertEqual(rain.gains[0].label(), "-1")

    def test_received_points_are_drawn_in_blue(self) -> None:
        rain = _rain_with_bubble(4)
        _wave(rain, 0.5)
        canvas = np.full((160, 200, 3), 255, dtype=np.uint8)
        rain.gains[0].draw(canvas)
        blue, _green, red = canvas[:, :, 0], canvas[:, :, 1], canvas[:, :, 2]
        self.assertTrue(np.any((blue > 180) & (blue > red + 40)))

    def test_missed_bubble_does_not_change_score(self) -> None:
        rain = BubbleRain(np.random.default_rng(0))
        rain._spawn_in = 50
        rain.bubbles.append(SoapBubble(40, 200, 4, 16, 5))
        rain.tick(120, 180, [], 0.0)
        self.assertEqual(rain.score, 0)
        self.assertEqual(rain.bubbles, [])

    def test_number_inside_bubble_is_blue(self) -> None:
        canvas = np.full((140, 140, 3), 255, dtype=np.uint8)
        SoapBubble(70, 70, 4, 30, 1).draw(canvas)
        roi = canvas[50:90, 50:90]
        blue, _green, red = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        self.assertTrue(np.any((blue > 180) & (blue > red + 40)))
        self.assertEqual(POINT_BLUE[0], 255)

    def test_score_is_drawn_at_the_top(self) -> None:
        rain = BubbleRain(np.random.default_rng(0))
        rain.score = 7
        canvas = np.zeros((90, 320, 3), dtype=np.uint8)
        rain.draw_score(canvas)
        top = canvas[:40]
        bottom = canvas[70:]
        self.assertGreater(int(top.sum()), 0)
        self.assertEqual(int(bottom.sum()), 0)


class GurliRoundTest(unittest.TestCase):
    def test_clock_shows_three_minutes_and_counts_down(self) -> None:
        self.assertEqual(ROUND_SECONDS, 180.0)
        self.assertEqual(clock_text(180), "3:00")
        self.assertEqual(clock_text(179.1), "3:00")
        self.assertEqual(clock_text(179), "2:59")
        self.assertEqual(clock_text(90), "1:30")
        self.assertEqual(clock_text(0), "0:00")

    def test_round_ends_at_three_minutes(self) -> None:
        game_round = TimedRound()
        game_round.start(10)
        self.assertFalse(game_round.over(10 + 179.9))
        self.assertTrue(game_round.over(10 + 180))
        self.assertEqual(game_round.remaining(10 + 60), 120)


class HighScoreTest(unittest.TestCase):
    def test_score_is_kept_and_highest_is_first(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scores.json"
            board = HighScoreList(path)
            board.add("Bo", 2)
            board.add("Ane", 9)
            again = HighScoreList(path)
            self.assertEqual([(entry.name, entry.score) for entry in again.entries], [("Ane", 9), ("Bo", 2)])

    def test_top_three_can_type_a_name_and_a_tie_for_third_counts(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            board = HighScoreList(Path(folder) / "scores.json")
            board.add("En", 9)
            board.add("To", 8)
            board.add("Tre", 7)
            screen = ResultScreen(board, 7, lambda: None)
            self.assertTrue(screen.can_name)
            self.assertEqual(screen.score_line(), "7")
            for key in (ord("h"), ord("q"), ord("m")):
                screen.on_key(key)
            screen.on_key(8)
            screen.on_key(ord("a"))
            self.assertEqual(screen.entry.name, "hqa")
            self.assertEqual(screen.score_line(), "7")
            self.assertIn("4 hqa 7", screen.highscore_lines())

    def test_fourth_place_cannot_type_and_eleventh_is_left_off_the_list(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            board = HighScoreList(Path(folder) / "scores.json")
            board.add("En", 9)
            board.add("To", 8)
            board.add("Tre", 7)
            fourth = ResultScreen(board, 6, lambda: None)
            fourth.on_key(ord("A"))
            self.assertFalse(fourth.can_name)
            assert fourth.entry is not None
            self.assertEqual(fourth.entry.name, "")
            self.assertEqual(fourth.score_line(), "6")
            self.assertIn("4 — 6", fourth.highscore_lines())
            full = HighScoreList(Path(folder) / "full.json")
            for score in range(20, 10, -1):
                full.add("", score)
            self.assertEqual(len(full.entries), HIGHSCORE_SHOWN)
            missed = ResultScreen(full, 1, lambda: None)
            missed.on_key(ord("A"))
            self.assertIsNone(missed.entry)
            self.assertEqual(missed.score_line(), "1")
            self.assertTrue(all(entry.score > 1 for entry in full.entries))

    def test_name_stops_at_twelve_characters(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            screen = ResultScreen(HighScoreList(Path(folder) / "scores.json"), 3, lambda: None)
            for _ in range(NAME_MAX + 3):
                screen.on_key(ord("a"))
            self.assertEqual(len(screen.entry.name), NAME_MAX)

    def test_play_button_is_below_the_score_and_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            started: list[int] = []
            screen = ResultScreen(HighScoreList(Path(folder) / "scores.json"), 4, lambda: started.append(1))
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            screen.draw(canvas)
            assert screen.button.rect is not None
            x1, y1, x2, y2 = screen.button.rect
            self.assertGreater(y1, 240)
            button = canvas[y1:y2, x1:x2]
            green = (button[:, :, 1] > 180) & (button[:, :, 1] > button[:, :, 2] + 40)
            self.assertGreater(int(green.sum()), 50)
            self.assertGreater(int(canvas[:90].sum()), 0)
            screen.on_click((x1 + x2) // 2, (y1 + y2) // 2)
            self.assertEqual(started, [1])
            screen.on_click(2, 2)
            self.assertEqual(started, [1])
            ix = 0
            iy = 0
            from gamConsole.opencv_ui import fitted_frame_rect

            x0, y0, nw, nh = fitted_frame_rect(640, 480, CANVAS_W, CANVAS_H)
            ix = x0 + int(((x1 + x2) // 2) * nw / 640)
            iy = y0 + int(((y1 + y2) // 2) * nh / 480)
            point = game_point_from_canvas(ix, iy, 640, 480, CANVAS_W, CANVAS_H)
            assert point is not None
            self.assertTrue(screen.button.contains(point[0], point[1]))


class GurliRoundPlayTest(unittest.TestCase):
    def test_three_minutes_shows_the_score_and_play_starts_over(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            game = GurliGrisGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            try:
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                self.assertIsInstance(game.match, RoundScore)
                playing = game.on_frame(frame, now=1000)
                self.assertIsNone(game.match.result)
                self.assertFalse(game.accepts_name_key(ord("h")))
                self.assertEqual(clock_text(game.match.round.remaining(1000)), "3:00")
                self.assertGreater(int(playing[:, -80:].sum()), 0)
                game.player.bubbles.score = 6
                ended = game.on_frame(frame, now=1000 + ROUND_SECONDS)
                self.assertIsNotNone(game.match.result)
                assert game.match.result is not None
                self.assertEqual(game.match.result.score, 6)
                self.assertTrue(game.match.result.can_name)
                self.assertIn("1 — 6", game.match.result.highscore_lines())
                self.assertTrue(game.accepts_name_key(ord("h")))
                game.on_key(ord("h"))
                self.assertEqual(game.match.result.entry.name, "h")
                assert game.match.result.button.rect is not None
                x1, y1, x2, y2 = game.match.result.button.rect
                self.assertGreater(int(ended.sum()), 0)
                game.on_click((x1 + x2) // 2, (y1 + y2) // 2)
                self.assertIsNone(game.match.result)
                self.assertEqual(game.player.bubbles.score, 0)
                game.on_frame(frame, now=9000)
                self.assertIsNone(game.match.result)
                self.assertEqual(game.match.round.started_at, 9000)
            finally:
                game.close()


if __name__ == "__main__":
    unittest.main()
