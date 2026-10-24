"""Sangvalg inde i Just Dance."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from gamConsole.just_dance_score import MoveScore
from gamConsole.just_dance_session import JustDanceSession
from gamConsole.round_score import HighScoreList
from gamConsole.just_dance_songs import DANCE_SONGS, resolve_song, song_title
from gamConsole.just_dance_tracks import TrackList
from gamConsole.menu_opencv import _compose_menu

ROOT = Path(__file__).resolve().parents[1]


class DanceSongListTest(unittest.TestCase):
    def test_lists_the_five_repo_songs(self) -> None:
        ids = [song.id for song in DANCE_SONGS]
        self.assertEqual(
            ids,
            ["cheapthrills", "callmemaybe", "uptownfunk", "ghungroo", "dontstartnow"],
        )
        songs = ROOT / "vendor" / "just-dance" / "songs"
        audio = ROOT / "vendor" / "just-dance" / "songs_audio"
        if songs.is_dir():
            for song in DANCE_SONGS:
                self.assertTrue((songs / f"{song.id}.mp4").is_file(), song.id)
                self.assertTrue((audio / f"{song.id}.mp3").is_file(), song.id)

    def test_unknown_song_falls_back_to_cheap_thrills(self) -> None:
        self.assertEqual(resolve_song("uptownfunk"), "uptownfunk")
        self.assertEqual(resolve_song("findes-ikke"), "cheapthrills")
        self.assertEqual(song_title("dontstartnow"), "Don't Start Now")


class TrackListTest(unittest.TestCase):
    def test_picker_has_one_button_per_song(self) -> None:
        tracks = TrackList()
        frame = tracks.render(960, 540)
        self.assertEqual(frame.shape, (540, 960, 3))
        self.assertEqual([button.song_id for button in tracks.buttons], [song.id for song in DANCE_SONGS])
        for button in tracks.buttons:
            self.assertGreaterEqual(button.x1, 0)
            self.assertLess(button.x2, frame.shape[1])
            self.assertGreaterEqual(button.y1, 0)
            self.assertLess(button.y2, frame.shape[0])
            self.assertEqual(tracks.hit((button.x1 + button.x2) // 2, (button.y1 + button.y2) // 2), button.song_id)

    def test_main_menu_does_not_list_songs(self) -> None:
        from gamConsole.games_registry import games_for_menu

        games = games_for_menu()
        selected = next(i for i, game in enumerate(games) if game.id == "just-dance")
        webcam = np.zeros((480, 640, 3), dtype=np.uint8)
        _canvas, layout = _compose_menu(webcam, games, selected, None, False, 0)
        self.assertFalse(hasattr(layout, "song_boxes"))


class _SilentPlayback:
    def __init__(self) -> None:
        self.played: list = []
        self.stopped = 0

    def play(self, path: Path) -> None:
        self.played.append(path)

    def stop(self) -> None:
        self.stopped += 1


class InGameTrackChoiceTest(unittest.TestCase):
    def test_game_opens_on_the_track_list(self) -> None:
        session = JustDanceSession(0)
        try:
            frame = session.tick()
            self.assertIsNone(session._game)
            self.assertTrue(session._picking)
            self.assertIsNotNone(frame)
            cheap = next(button for button in session._tracks.buttons if button.song_id == "cheapthrills")
            session.on_click((cheap.x1 + cheap.x2) // 2, (cheap.y1 + cheap.y2) // 2)
            self.assertEqual(session._pending, "cheapthrills")
            session.on_key(ord("2"))
            self.assertEqual(session._pending, "callmemaybe")
        finally:
            session.stop()

    def test_switch_opens_the_new_video_without_a_new_model(self) -> None:
        video = ROOT / "vendor" / "just-dance" / "songs" / "cheapthrills.mp4"
        if not video.is_file():
            self.skipTest("vendor songs mangler")
        session = JustDanceSession.__new__(JustDanceSession)
        session._playback = _SilentPlayback()
        session.song = "uptownfunk"
        session._frame1 = np.zeros((4, 4, 3), dtype=np.uint8)
        session._picking = False
        session._ready = True
        session._error = None
        first = cv2.VideoCapture(str(video))
        controller = SimpleNamespace(
            cap1=first,
            frame1_rate=1.0,
            angles_video={"left_arm": [1, 2]},
            angles_camera={"left_arm": [3]},
        )
        session._game = SimpleNamespace(controller=controller)
        try:
            with tempfile.TemporaryDirectory() as folder:
                session.moves = MoveScore(HighScoreList(Path(folder) / "scores.json"))
                session.moves.score = 30
                session._switch_video()
            self.assertEqual(session.moves.score, 0)
            self.assertIs(session._game.controller, controller)
            self.assertTrue(controller.cap1.isOpened())
            self.assertNotEqual(controller.frame1_rate, 1.0)
            self.assertEqual(controller.angles_video["left_arm"], [])
            self.assertEqual(controller.angles_camera["left_arm"], [])
            self.assertIsNone(session._frame1)
            self.assertFalse(session._picking)
            self.assertEqual(session._playback.played[-1].name, "uptownfunk.mp3")
        finally:
            if controller.cap1 is not None:
                controller.cap1.release()


if __name__ == "__main__":
    unittest.main()
