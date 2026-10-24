"""Point for et rigtigt trin på sangens tidspunkt, og highscore bagefter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gamConsole.game_audio import game_audio
from gamConsole.just_dance_score import JOINTS, POINTS_PER_JOINT, MoveScore, points_for_latest
from gamConsole.just_dance_session import JustDanceSession
from gamConsole.just_dance_tempo import DanceTempo
from gamConsole.just_dance_tracks import TrackList
from gamConsole.round_score import HighScoreList


def _angles(value: float) -> dict:
    return {joint: [value] for joint in JOINTS}


class _Frames:
    def __init__(self, frame: np.ndarray, count: int) -> None:
        self._frame = frame
        self._left = count

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self._left <= 0:
            return False, None
        self._left -= 1
        return True, self._frame.copy()


class _Model:
    def store_angles(self, dest, _frame, keypoints) -> None:
        for joint in JOINTS:
            dest[joint].append(keypoints[joint])


class _SilentPlayback:
    def __init__(self) -> None:
        self.stopped = 0

    def play(self, _path) -> None:
        return None

    def stop(self) -> None:
        self.stopped += 1


def _session(board: HighScoreList) -> JustDanceSession:
    session = JustDanceSession.__new__(JustDanceSession)
    session.song = "cheapthrills"
    session.camera = 0
    session._game = None
    session._ready = False
    session._picking = True
    session._pending = None
    session._error = None
    session._stage = SimpleNamespace(composite=lambda frame: frame, close=lambda: None)
    session._playback = _SilentPlayback()
    session._tracks = TrackList()
    session._tempo = None
    session._frame1 = None
    session._last_scene = None
    session.moves = MoveScore(board)
    session.title = "Just Dance"
    return session


class LatestMoveScoreTest(unittest.TestCase):
    def test_each_matching_joint_at_this_moment_is_worth_points(self) -> None:
        video = _angles(90.0)
        camera = _angles(90.0)
        camera["left_arm"] = [140.0]
        camera["right_leg"] = [110.0]
        self.assertEqual(points_for_latest(video, camera), 6 * POINTS_PER_JOINT)

    def test_twenty_degrees_is_outside_the_vendor_threshold(self) -> None:
        video = _angles(90.0)
        camera = _angles(90.0)
        camera["left_arm"] = [110.0]
        self.assertEqual(points_for_latest(video, camera), 7 * POINTS_PER_JOINT)
        camera["left_arm"] = [109.0]
        self.assertEqual(points_for_latest(video, camera), 8 * POINTS_PER_JOINT)

    def test_only_the_latest_sample_counts(self) -> None:
        video = _angles(0.0)
        camera = _angles(90.0)
        video["left_arm"].append(10.0)
        camera["left_arm"].append(12.0)
        self.assertEqual(points_for_latest(video, camera), POINTS_PER_JOINT)

    def test_a_missing_joint_gives_nothing(self) -> None:
        self.assertEqual(points_for_latest({}, _angles(1.0)), 0)


class PointPopupTest(unittest.TestCase):
    def test_a_hit_raises_the_score_and_shows_a_rising_number(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            moves = MoveScore(HighScoreList(Path(folder) / "scores.json"))
            before = len(game_audio.played)
            gained = moves.award(_angles(40.0), _angles(40.0))
            self.assertEqual(gained, 8 * POINTS_PER_JOINT)
            self.assertEqual(moves.score, gained)
            self.assertEqual(game_audio.played[before:], ["positive"])
            gain = moves._view().gains[0]
            self.assertEqual(gain.label(), f"+{gained}")
            moves.place_latest(200.0, 140.0)
            moves.tick(10.0)
            moves.tick(10.5)
            self.assertLess(gain.y, 140.0)
            canvas = np.zeros((240, 400, 3), dtype=np.uint8)
            moves.draw(canvas)
            self.assertGreater(int(canvas[0:40].max()), 0)
            self.assertGreater(int(canvas[100:160, 170:230].max()), 200)

    def test_a_miss_does_not_show_points(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            moves = MoveScore(HighScoreList(Path(folder) / "scores.json"))
            before = len(game_audio.played)
            self.assertEqual(moves.award(_angles(0.0), _angles(90.0)), 0)
            self.assertEqual(moves.score, 0)
            self.assertEqual(game_audio.played[before:], [])
            self.assertEqual(moves._view().gains, [])


class DanceScoreSessionTest(unittest.TestCase):
    def test_a_matching_pose_on_the_song_frame_draws_points(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            session = _session(HighScoreList(Path(folder) / "scores.json"))
            frame = np.zeros((200, 160, 3), dtype=np.uint8)
            controller = SimpleNamespace(
                cap1=_Frames(frame, 40),
                cap2=_Frames(frame, 5),
                angles_video={joint: [] for joint in JOINTS},
                angles_camera={joint: [] for joint in JOINTS},
                model=_Model(),
                process_frame=lambda _image: {joint: 40.0 for joint in JOINTS},
            )
            session._game = SimpleNamespace(controller=controller)
            session._ready = True
            session._picking = False
            session._tempo = DanceTempo(30.0)
            shown = session.tick()
            self.assertIsNotNone(shown)
            self.assertEqual(session.moves.score, 8 * POINTS_PER_JOINT)
            self.assertGreater(int(shown[0:40].max()), 0)
            self.assertEqual(session.moves._view().gains[0].label(), "+80")

    def test_song_end_saves_highscore_and_play_again_returns_to_the_list(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            board = HighScoreList(Path(folder) / "scores.json")
            session = _session(board)
            session._game = SimpleNamespace(controller=SimpleNamespace(cap1=_Frames(np.zeros((8, 8, 3), np.uint8), 0)))
            session._ready = True
            session._picking = False
            session._tempo = DanceTempo(30.0)
            session.moves.score = 40
            session._last_scene = np.zeros((400, 640, 3), dtype=np.uint8)
            shown = session.tick()
            self.assertIsNotNone(shown)
            self.assertEqual(board.entries[0].score, 40)
            self.assertTrue(session.claims_key(ord("a")))
            session.on_key(ord("a"))
            self.assertEqual(session.moves.match.result.entry.name, "a")
            session.on_click(4, 4)
            self.assertIsNone(session._pending)
            self.assertEqual(session.finalize_score(), 40)
            self.assertEqual(len(board.entries), 1)
            button = session.moves.match.result.button.rect
            session.on_click((button[0] + button[2]) // 2, (button[1] + button[3]) // 2)
            self.assertTrue(session._picking)
            self.assertEqual(session.moves.score, 0)
            again = session.tick()
            self.assertEqual(again.shape[0], 540)
            self.assertEqual(len(session._tracks.buttons), 5)
            self.assertEqual(board.entries[0].score, 40)
            self.assertEqual(board.entries[0].name, "a")

    def test_leaving_mid_song_keeps_the_points_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            board = HighScoreList(Path(folder) / "scores.json")
            session = _session(board)
            self.assertIsNone(session.finalize_score())
            session._game = object()
            session.moves.score = 12
            self.assertEqual(session.finalize_score(), 12)
            self.assertEqual(session.finalize_score(), 12)
            self.assertEqual([(entry.name, entry.score) for entry in board.entries], [("", 12)])


if __name__ == "__main__":
    unittest.main()
