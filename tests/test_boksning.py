#12
"""Boksning i spilmenuen, uden at åbne kamera eller spilvindue."""

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

from gamConsole.boksning import (  # noqa: E402
    LIGHT_SECONDS,
    PAD_COUNT,
    Boxer,
    BoxingGame,
    MusicTarget,
    RingBackdrop,
    RingFrame,
)
from gamConsole.boksning_session import BoksningSession  # noqa: E402
from gamConsole.boxing_track import SONG_ARTIST, SONG_PATH, SONG_TITLE  # noqa: E402
from gamConsole.flying_users_ar import WebcamARGame  # noqa: E402
from gamConsole.games_registry import games_for_menu  # noqa: E402
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


def _arms(elbow_y: int, wrist_y: int, wrist_x_out: int) -> BodyPose:
    points = [(-1, -1)] * 33
    points[7] = (220, 80)
    points[8] = (300, 80)
    points[L_SHOULDER] = (180, 150)
    points[R_SHOULDER] = (320, 150)
    points[L_ELBOW] = (180 - wrist_x_out // 2, elbow_y)
    points[R_ELBOW] = (320 + wrist_x_out // 2, elbow_y)
    points[L_WRIST] = (180 - wrist_x_out, wrist_y)
    points[R_WRIST] = (320 + wrist_x_out, wrist_y)
    points[L_HIP] = (210, 240)
    points[R_HIP] = (290, 240)
    points[L_KNEE] = (210, 300)
    points[R_KNEE] = (300, 300)
    points[L_ANKLE] = (210, 350)
    points[R_ANKLE] = (300, 350)
    return BodyPose(640, 360, tuple(points))


def _shorts(image: np.ndarray) -> np.ndarray:
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (blue > 140) & (blue > red + 30) & (green < 200)


def _orange(image: np.ndarray) -> np.ndarray:
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (red > 160) & (red > green + 20) & (blue < 120)


class BoksningMenuTest(unittest.TestCase):
    def test_registry_button_is_builtin(self) -> None:
        entry = next(game for game in games_for_menu() if game.id == "boksning")
        self.assertEqual(entry.launcher, "builtin")
        self.assertEqual(entry.mode_arg, "boksning")
        self.assertEqual(entry.title, "Boksning")

    def test_selected_card_stays_above_quit_bar(self) -> None:
        games = games_for_menu()
        selected = next(index for index, game in enumerate(games) if game.id == "boksning")
        webcam = np.zeros((480, 640, 3), dtype=np.uint8)
        _canvas, layout = _compose_menu(webcam, games, selected, None, False, 0)
        card = next(box for box in layout.card_boxes if box.game_index == selected)
        self.assertLess(card.y2, CANVAS_H - QUIT_BAR_H)
        self.assertIn(selected, {box.game_index for box in layout.card_boxes})

    def test_session_factory_builds_boksning(self) -> None:
        from gamConsole.game_sessions import create_session

        entry = next(game for game in games_for_menu() if game.id == "boksning")
        with patch("gamConsole.boksning_session.cv2.VideoCapture") as cap:
            cap.return_value.isOpened.return_value = False
            session, message = create_session(entry, 0)
        self.assertIsInstance(session, BoksningSession)
        self.assertIn("Boksning", message)
        assert session is not None
        session.stop()


class RingBackdropTest(unittest.TestCase):
    def test_frame_is_the_ring_picture(self) -> None:
        import cv2

        from gamConsole.boksning import RING_PATH

        source = cv2.imread(str(RING_PATH), cv2.IMREAD_COLOR)
        self.assertIsNotNone(source)
        assert source is not None
        native = RingBackdrop().render(source.shape[1], source.shape[0])
        self.assertTrue(np.array_equal(native, source))
        frame = RingBackdrop().render(640, 360)
        self.assertEqual(frame.shape, (360, 640, 3))
        #12 Loftet i billedet er mørkt. Lærredet nede er lyst.
        top = int(frame[8, 320].mean())
        floor = int(frame[int(360 * 0.82), 320].mean())
        self.assertLess(top, floor - 40)

    def test_thumbnail_shows_the_boxer_and_a_neon_field(self) -> None:
        from gamConsole.game_thumbnails import clear_thumbnail_cache, get_game_thumbnail

        clear_thumbnail_cache()
        thumb = get_game_thumbnail("boksning", 96)
        self.assertEqual(thumb.shape[:2], (96, 96))
        self.assertGreater(int(_orange(thumb).sum()), 4)
        #12 Menu-billedet tænder felt 0, som er lilla.
        purple = (thumb[:, :, 0] > 100) & (thumb[:, :, 2] > 80) & (thumb[:, :, 1] < 140)
        self.assertGreater(int(purple.sum()), 3)


class BoxerMoveTest(unittest.TestCase):
    def test_level_head_stays_centered(self) -> None:
        boxer = Boxer()
        box = boxer.box(_ears((200, 80), (280, 80)), 640, 360)
        body_w = box[2] - box[0]
        self.assertEqual(box[0], 640 // 2 - body_w // 2)

    def test_head_tilt_does_not_move_the_boxer(self) -> None:
        pose = _ears((220, 78), (300, 88))
        level = _ears((220, 80), (300, 80))
        boxer = Boxer()
        self.assertEqual(boxer.box(pose, 640, 360)[0], boxer.box(level, 640, 360)[0])

    def test_full_tilt_stays_centered_between_the_posts(self) -> None:
        boxer = Boxer()
        ring = RingFrame(960, 540)
        right = boxer.box(_ears((200, 40), (280, 200)), 960, 540)
        left = boxer.box(_ears((200, 200), (280, 40)), 960, 540)
        self.assertEqual(left[0], right[0])
        self.assertGreaterEqual(right[0], ring.left)
        self.assertLessEqual(right[2], ring.right)

    def test_avatar_is_cut_into_joints(self) -> None:
        boxer = Boxer()
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
            self.assertGreater(boxer.avatar.pixels[name], 100)

    def test_raised_arms_lift_the_gloves(self) -> None:
        boxer = Boxer()
        down = np.zeros((360, 640, 3), np.uint8)
        up = down.copy()
        boxer.draw(down, _arms(elbow_y=220, wrist_y=280, wrist_x_out=40))
        boxer.draw(up, _arms(elbow_y=80, wrist_y=40, wrist_x_out=90))
        self.assertLess(_orange_y(up), _orange_y(down))

    def test_figure_stays_when_the_head_tilts(self) -> None:
        boxer = Boxer()
        level = np.zeros((360, 640, 3), np.uint8)
        tilted = level.copy()
        boxer.draw(level, _ears((220, 90), (300, 90)))
        boxer.draw(tilted, _ears((220, 70), (300, 150)))
        #12 Kroppen bliver stående. Hovedleddet må dreje, men figuren rykker sig ikke.
        self.assertAlmostEqual(_mass_x(tilted), _mass_x(level), delta=2.0)


class PunchTest(unittest.TestCase):
    def test_fields_spread_out_and_the_legs_are_cut(self) -> None:
        #12 Målene står længere fra hinanden, og underbenene ryger ud under kanten.
        boxer = Boxer()
        box = boxer.box(None, 640, 360)
        self.assertLess(box[1], 50)
        self.assertGreater(box[3], 360)
        target = MusicTarget(np.random.default_rng(1))
        target.place(boxer, 640, 360)
        xs = [pad.x for pad in target.pads]
        ys = [pad.y for pad in target.pads]
        self.assertGreater(max(xs) - min(xs), 300)
        self.assertGreater(max(ys) - min(ys), 140)
        for pad in target.pads:
            self.assertGreaterEqual(pad.x - pad.radius, 0)
            self.assertLessEqual(pad.x + pad.radius, 640)
            self.assertGreaterEqual(pad.y - pad.radius, 0)
            self.assertLessEqual(pad.y + pad.radius, 360)
        frame = np.zeros((360, 640, 3), np.uint8)
        boxer.draw(frame, None)
        self.assertGreater(int(_shorts(frame).sum()), 40)
        self.assertGreater(int(np.any(frame[-1] != 0, axis=1).sum()), 0)

    def test_six_fields_sit_around_the_boxer(self) -> None:
        boxer = Boxer()
        target = MusicTarget(np.random.default_rng(1))
        target.place(boxer, 640, 360)
        self.assertEqual(len(target.pads), PAD_COUNT)
        left, right = boxer.gloves(None, 640, 360)
        center_x = (left[0] + right[0]) / 2.0
        self.assertEqual(sum(pad.x < center_x for pad in target.pads), 3)
        self.assertEqual(sum(pad.x > center_x for pad in target.pads), 3)
        self.assertLess(min(pad.y for pad in target.pads), left[1])
        self.assertGreater(max(pad.y for pad in target.pads), left[1])

    def test_each_field_has_its_own_neon(self) -> None:
        boxer = Boxer()
        target = MusicTarget(np.random.default_rng(1))
        target.place(boxer, 640, 360)
        seen = set()
        for pad in target.pads:
            frame = np.zeros((360, 640, 3), np.uint8)
            target.light(pad.index, 0.0)
            target.draw_pads(frame, 0.0)
            x = int(round(pad.x + pad.radius * 0.55))
            y = int(round(pad.y))
            seen.add(tuple(int(channel) for channel in frame[y, x]))
        self.assertEqual(len(seen), PAD_COUNT)

    def test_guard_does_not_hit_a_lit_field(self) -> None:
        boxer = Boxer()
        target = MusicTarget(np.random.default_rng(1))
        target.place(boxer, 640, 360)
        for index in range(PAD_COUNT):
            target.light(index, 0.0)
            target.tick(boxer, None, 640, 360, 0.2, auto=False)
        self.assertEqual(target.score, 0)
        self.assertEqual(target.lit, PAD_COUNT - 1)

    def test_raised_glove_hits_the_lit_high_field(self) -> None:
        boxer = Boxer()
        target = MusicTarget(np.random.default_rng(1))
        target.place(boxer, 640, 360)
        target.light(0, 0.0)
        target.tick(boxer, _arms(elbow_y=70, wrist_y=30, wrist_x_out=100), 640, 360, 0.2, auto=False)
        self.assertEqual(target.score, 1)
        self.assertIsNone(target.lit)

    def test_side_glove_hits_the_lit_side_field(self) -> None:
        boxer = Boxer()
        target = MusicTarget(np.random.default_rng(1))
        target.place(boxer, 640, 360)
        target.light(5, 0.0)
        target.tick(boxer, _arms(elbow_y=150, wrist_y=150, wrist_x_out=140), 640, 360, 0.2, auto=False)
        self.assertEqual(target.score, 1)

    def test_low_glove_hits_the_lit_low_field(self) -> None:
        boxer = Boxer()
        target = MusicTarget(np.random.default_rng(1))
        target.place(boxer, 640, 360)
        target.light(4, 0.0)
        target.tick(boxer, _arms(elbow_y=230, wrist_y=280, wrist_x_out=50), 640, 360, 0.2, auto=False)
        self.assertEqual(target.score, 1)

    def test_a_dark_field_scores_nothing(self) -> None:
        boxer = Boxer()
        target = MusicTarget(np.random.default_rng(1))
        target.place(boxer, 640, 360)
        target.light(4, 0.0)
        target.tick(boxer, _arms(elbow_y=70, wrist_y=30, wrist_x_out=100), 640, 360, 0.2, auto=False)
        self.assertEqual(target.score, 0)
        self.assertEqual(target.lit, 4)

    def test_a_late_punch_scores_nothing(self) -> None:
        boxer = Boxer()
        target = MusicTarget(np.random.default_rng(1))
        target.place(boxer, 640, 360)
        target.light(0, 0.0)
        target.tick(
            boxer,
            _arms(elbow_y=70, wrist_y=30, wrist_x_out=100),
            640,
            360,
            LIGHT_SECONDS + 0.05,
            auto=False,
        )
        self.assertEqual(target.score, 0)
        self.assertIsNone(target.lit)


class BoxingGameTest(unittest.TestCase):
    def test_game_is_webcam_ar(self) -> None:
        self.assertIsInstance(BoxingGame(camera=0), WebcamARGame)

    def test_on_frame_shows_the_ring_not_the_webcam(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            game = BoxingGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            try:
                webcam = np.full((360, 640, 3), (0, 0, 255), dtype=np.uint8)
                out = game.on_frame(webcam, now=10.0)
            finally:
                game.close()
        self.assertEqual(out.shape, (360, 640, 3))
        self.assertEqual(int(np.all(out == webcam, axis=2).sum()), 0)
        self.assertLess(int(out[4, 4].mean()), 90)
        self.assertGreater(int(_shorts(out).sum()), 30)

    def test_webcam_is_mirrored_like_gurli(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            game = BoxingGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
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
            game = BoxingGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            try:
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                self.assertIsInstance(game.match, RoundScore)
                playing = game.on_frame(frame, now=1000.0)
                self.assertIsNone(game.match.result)
                self.assertEqual(clock_text(game.match.round.remaining(1000.0)), "3:00")
                self.assertGreater(int(playing[:, -80:].sum()), 0)
                game.player.target.score = 4
                game.on_frame(frame, now=1000.0 + ROUND_SECONDS)
                self.assertIsNotNone(game.match.result)
                assert game.match.result is not None
                self.assertEqual(game.match.result.score, 4)
                self.assertIn("1 — 4", game.match.result.highscore_lines())
                assert game.match.result.button.rect is not None
                x1, y1, x2, y2 = game.match.result.button.rect
                game.on_click((x1 + x2) // 2, (y1 + y2) // 2)
                self.assertIsNone(game.match.result)
                self.assertEqual(game.player.target.score, 0)
            finally:
                game.close()

    #20
    def test_action_driving_rock_plays_for_the_round_and_stops_on_the_result(self) -> None:
        self.assertEqual(SONG_TITLE, "Action Driving Rock")
        self.assertEqual(SONG_ARTIST, "Verclub_Music")
        self.assertTrue(SONG_PATH.is_file())
        self.assertGreater(SONG_PATH.stat().st_size, 1_000_000)
        with tempfile.TemporaryDirectory() as folder:
            game = BoxingGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            try:
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                self.assertFalse(game.track.started)
                game.on_frame(frame, now=1000.0)
                self.assertTrue(game.track.started)
                self.assertEqual(game.track.path, SONG_PATH)
                self.assertIsNone(game.track._sound)
                game.on_frame(frame, now=1000.0 + ROUND_SECONDS)
                self.assertFalse(game.track.started)
                assert game.match.result is not None
                assert game.match.result.button.rect is not None
                x1, y1, x2, y2 = game.match.result.button.rect
                game.on_click((x1 + x2) // 2, (y1 + y2) // 2)
                game.on_frame(frame, now=2000.0)
                self.assertTrue(game.track.started)
            finally:
                game.close()
                self.assertFalse(game.track.started)


def _orange_y(image: np.ndarray) -> float:
    ys = np.where(_orange(image))[0]
    if len(ys) == 0:
        raise AssertionError("handskerne blev ikke tegnet")
    return float(ys.mean())


def _mass_x(image: np.ndarray) -> float:
    xs = np.where(_shorts(image))[1]
    if len(xs) == 0:
        raise AssertionError("shortsene blev ikke tegnet")
    return float(xs.mean())


if __name__ == "__main__":
    unittest.main()
