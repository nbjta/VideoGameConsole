#11
"""Målmand i spilmenuen, uden at åbne kamera eller spilvindue."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from gamConsole.flying_users_ar import WebcamARGame  # noqa: E402
from gamConsole.games_registry import games_for_menu  # noqa: E402
from gamConsole.head_tilt import HeadTilt  # noqa: E402
from gamConsole.maalmand import (  # noqa: E402
    FLIGHT_SECONDS,
    Football,
    GoalBackdrop,
    GoalFrame,
    Goalkeeper,
    GoalkeeperGame,
    ShotClock,
)
from gamConsole.maalmand_session import MaalmandSession  # noqa: E402
from gamConsole.menu_opencv import CANVAS_H, QUIT_BAR_H, _compose_menu  # noqa: E402
from gamConsole.pose_body_overlay import (  # noqa: E402
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
    BodyPose,
)
from gamConsole.round_score import (  # noqa: E402
    ROUND_SECONDS,
    HighScoreList,
    RoundScore,
    clock_text,
)


def _ears(left: tuple, right: tuple) -> BodyPose:
    points = [(-1, -1)] * 33
    points[7] = left
    points[8] = right
    return BodyPose(640, 360, tuple(points))


def _jersey(image: np.ndarray) -> np.ndarray:
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (red > 150) & (red > green + 40) & (red > blue + 40)


class MaalmandMenuTest(unittest.TestCase):
    def test_registry_button_is_builtin(self) -> None:
        entry = next(game for game in games_for_menu() if game.id == "maalmand")
        self.assertEqual(entry.launcher, "builtin")
        self.assertEqual(entry.mode_arg, "maalmand")
        self.assertEqual(entry.title, "Målmand")

    def test_selected_card_stays_above_quit_bar(self) -> None:
        games = games_for_menu()
        selected = next(index for index, game in enumerate(games) if game.id == "maalmand")
        webcam = np.zeros((480, 640, 3), dtype=np.uint8)
        _canvas, layout = _compose_menu(webcam, games, selected, None, False, 0)
        card = next(box for box in layout.card_boxes if box.game_index == selected)
        self.assertLess(card.y2, CANVAS_H - QUIT_BAR_H)
        self.assertIn(selected, {box.game_index for box in layout.card_boxes})

    def test_session_factory_builds_maalmand(self) -> None:
        from gamConsole.game_sessions import create_session

        entry = next(game for game in games_for_menu() if game.id == "maalmand")
        with patch("gamConsole.maalmand_session.cv2.VideoCapture") as cap:
            cap.return_value.isOpened.return_value = False
            session, message = create_session(entry, 0)
        self.assertIsInstance(session, MaalmandSession)
        self.assertIn("Målmand", message)
        assert session is not None
        session.stop()


class GoalBackdropTest(unittest.TestCase):
    def test_frame_is_a_goal(self) -> None:
        frame = GoalBackdrop().render(640, 360)
        goal = GoalFrame(640, 360)
        sky = frame[2, 2]
        self.assertGreater(int(sky[0]), int(sky[2]))
        grass = frame[-4, 8]
        self.assertGreater(int(grass[1]), int(grass[0]))
        self.assertGreater(int(grass[1]), int(grass[2]))
        post = frame[(goal.top + goal.line) // 2, goal.left_c]
        self.assertGreater(int(post.min()), 220)
        mouth = frame[goal.top + 10 : goal.line - 10, goal.inner_left + 8 : goal.inner_right - 8]
        self.assertGreater(float(mouth[:, :, 0].mean()), float(mouth[:, :, 2].mean()))

    def test_thumbnail_shows_the_goal(self) -> None:
        from gamConsole.game_thumbnails import clear_thumbnail_cache, get_game_thumbnail

        clear_thumbnail_cache()
        thumb = get_game_thumbnail("maalmand", 96)
        self.assertEqual(thumb.shape[:2], (96, 96))
        self.assertGreater(int(thumb[2, 2, 0]), int(thumb[2, 2, 2]))
        self.assertGreater(int(thumb[-3, 8, 1]), int(thumb[-3, 8, 2]))


class GoalkeeperMoveTest(unittest.TestCase):
    def test_level_head_stays_centered(self) -> None:
        keeper = Goalkeeper()
        body_w, _body_h = keeper.size(640, 360)
        origin = keeper.origin_x(_ears((200, 80), (280, 80)), 640, 360)
        self.assertEqual(origin, 640 // 2 - body_w // 2)

    def test_tilt_uses_the_same_offset_as_gurli_and_humdingers(self) -> None:
        from gamConsole.ar_player_view import HeadTilt as HumdingersTilt
        from gamConsole.gurli_gris import HeadTilt as GurliTilt

        pose = _ears((220, 78), (300, 88))
        level = _ears((220, 80), (300, 80))
        keeper = Goalkeeper()
        shift = keeper.origin_x(pose, 640, 360) - keeper.origin_x(level, 640, 360)
        shared = HeadTilt().offset_x(pose, 640)
        self.assertEqual(shift, shared)
        self.assertEqual(shared, GurliTilt().offset_x(pose, 640))
        self.assertEqual(shared, HumdingersTilt().offset_x(pose, 640))
        self.assertGreater(shift, 0)

    def test_full_tilt_stays_between_the_posts(self) -> None:
        keeper = Goalkeeper()
        goal = GoalFrame(960, 540)
        right = keeper.box(_ears((200, 40), (280, 200)), 960, 540)
        left = keeper.box(_ears((200, 200), (280, 40)), 960, 540)
        self.assertGreaterEqual(right[0], goal.inner_left)
        self.assertLessEqual(right[2], goal.inner_right)
        self.assertGreaterEqual(left[0], goal.inner_left)
        self.assertLessEqual(left[2], goal.inner_right)
        self.assertGreater(right[0], left[0])

    def test_avatar_is_cut_into_joints(self) -> None:
        keeper = Goalkeeper()
        for name in (
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
        ):
            self.assertGreater(keeper.avatar.pixels[name], 100)

    def test_raised_arms_lift_the_hands(self) -> None:
        keeper = Goalkeeper()
        down = np.zeros((360, 640, 3), np.uint8)
        up = down.copy()
        keeper.draw(down, _arms(elbow_y=280, knee_y=340))
        keeper.draw(up, _arms(elbow_y=80, knee_y=200))
        self.assertLess(_skin_y(up), _skin_y(down))

    def test_figure_shifts_right_when_the_head_tilts_right(self) -> None:
        keeper = Goalkeeper()
        level = np.zeros((360, 640, 3), np.uint8)
        tilted = level.copy()
        keeper.draw(level, _ears((220, 90), (300, 90)))
        keeper.draw(tilted, _ears((220, 70), (300, 150)))
        self.assertGreater(_mass_x(tilted), _mass_x(level))


class CatchTest(unittest.TestCase):
    def test_ball_on_the_keeper_scores(self) -> None:
        keeper = Goalkeeper()
        shots = ShotClock(np.random.default_rng(1))
        box = keeper.box(None, 640, 360)
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        shots.balls.append(Football((cx, 340.0), (cx, cy), 16, 0.0))
        shots.tick(640, 360, box, FLIGHT_SECONDS, spawn=False)
        self.assertEqual(shots.score, 1)
        self.assertEqual(shots.balls, [])

    def test_ball_beside_the_keeper_scores_nothing(self) -> None:
        keeper = Goalkeeper()
        shots = ShotClock(np.random.default_rng(1))
        box = keeper.box(None, 640, 360)
        cy = (box[1] + box[3]) / 2.0
        miss_x = box[0] - 40.0
        shots.balls.append(Football((320.0, 340.0), (miss_x, cy), 16, 0.0))
        shots.tick(640, 360, box, FLIGHT_SECONDS, spawn=False)
        self.assertEqual(shots.score, 0)
        self.assertEqual(len(shots.balls), 1)
        shots.tick(640, 360, box, FLIGHT_SECONDS + 0.5, spawn=False)
        self.assertEqual(shots.score, 0)
        self.assertEqual(shots.balls, [])

    def test_ball_is_drawn_in_the_goal(self) -> None:
        frame = GoalBackdrop().render(320, 180)
        ball = Football((160.0, 90.0), (200.0, 120.0), 16, 0.0)
        ball.draw(frame, 0.0)
        crop = frame[74:106, 144:176]
        white = (crop[:, :, 0] > 200) & (crop[:, :, 1] > 200) & (crop[:, :, 2] > 200)
        self.assertGreater(int(white.sum()), 20)


class GoalkeeperGameTest(unittest.TestCase):
    def test_game_is_webcam_ar(self) -> None:
        self.assertIsInstance(GoalkeeperGame(camera=0), WebcamARGame)

    def test_on_frame_shows_the_goal_not_the_webcam(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            game = GoalkeeperGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            try:
                webcam = np.full((360, 640, 3), (0, 0, 255), dtype=np.uint8)
                out = game.on_frame(webcam, now=10.0)
            finally:
                game.close()
        self.assertEqual(out.shape, (360, 640, 3))
        self.assertEqual(int(np.all(out == webcam, axis=2).sum()), 0)
        self.assertGreater(int(out[2, 2, 0]), int(out[2, 2, 2]))
        self.assertGreater(int(_jersey(out).sum()), 30)

    def test_webcam_is_mirrored_like_gurli(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            game = GoalkeeperGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            seen = []
            game.player.refresh_pose = lambda frame: seen.append(frame.copy())
            try:
                webcam = np.zeros((40, 80, 3), dtype=np.uint8)
                webcam[:, :8] = (0, 0, 255)
                game.on_frame(webcam, now=5.0)
            finally:
                game.close()
        self.assertEqual(len(seen), 1)
        self.assertGreater(float(seen[0][:, -8:, 2].mean()), 200)

    def test_three_minutes_shows_the_score_and_play_starts_over(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            game = GoalkeeperGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            try:
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                self.assertIsInstance(game.match, RoundScore)
                playing = game.on_frame(frame, now=1000.0)
                self.assertIsNone(game.match.result)
                self.assertEqual(clock_text(game.match.round.remaining(1000.0)), "3:00")
                self.assertGreater(int(playing[:, -80:].sum()), 0)
                game.player.shots.score = 4
                game.on_frame(frame, now=1000.0 + ROUND_SECONDS)
                self.assertIsNotNone(game.match.result)
                assert game.match.result is not None
                self.assertEqual(game.match.result.score, 4)
                self.assertIn("1 — 4", game.match.result.highscore_lines())
                assert game.match.result.button.rect is not None
                x1, y1, x2, y2 = game.match.result.button.rect
                game.on_click((x1 + x2) // 2, (y1 + y2) // 2)
                self.assertIsNone(game.match.result)
                self.assertEqual(game.player.shots.score, 0)
            finally:
                game.close()


def _arms(elbow_y: int, knee_y: int) -> BodyPose:
    points = [(-1, -1)] * 33
    points[7] = (220, 80)
    points[8] = (300, 80)
    points[L_SHOULDER] = (200, 180)
    points[R_SHOULDER] = (280, 180)
    points[L_ELBOW] = (160, elbow_y)
    points[R_ELBOW] = (320, elbow_y)
    hand = elbow_y - 40 if elbow_y < 180 else elbow_y + 40
    points[L_WRIST] = (140, hand)
    points[R_WRIST] = (340, hand)
    points[L_HIP] = (220, 260)
    points[R_HIP] = (270, 260)
    points[L_KNEE] = (220, knee_y)
    points[R_KNEE] = (300, knee_y)
    points[L_ANKLE] = (220, knee_y + 60)
    points[R_ANKLE] = (300, knee_y + 60)
    return BodyPose(640, 360, tuple(points))


def _skin_y(image: np.ndarray) -> float:
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    skin = (red > 180) & (green > 120) & (green < 210) & (blue > 90) & (blue < 200) & (red > blue + 40)
    ys = np.where(skin)[0]
    if len(ys) == 0:
        raise AssertionError("huden blev ikke tegnet")
    return float(ys.mean())


def _mass_x(image: np.ndarray) -> float:
    xs = np.where(_jersey(image))[1]
    if len(xs) == 0:
        raise AssertionError("trøjen blev ikke tegnet")
    return float(xs.mean())


if __name__ == "__main__":
    unittest.main()
