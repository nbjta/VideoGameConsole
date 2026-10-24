"""Sangen skal stoppe, når Just Dance forlades."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import List

from gamConsole.just_dance_audio import SongPlayback
from gamConsole.just_dance_session import JustDanceSession


class _FakeSound:
    def __init__(self, events: List[str], name: str) -> None:
        self._events = events
        self._name = name

    def play(self) -> None:
        self._events.append(f"play:{self._name}")

    def stop(self) -> None:
        self._events.append(f"stop:{self._name}")


class _RecordingPlayback(SongPlayback):
    def __init__(self) -> None:
        super().__init__()
        self.events: List[str] = []

    def _open(self, path: Path):
        if path.name == "missing.mp3":
            return None
        return _FakeSound(self.events, path.name)


class SongPlaybackTest(unittest.TestCase):
    def test_stop_ends_the_song(self) -> None:
        playback = _RecordingPlayback()
        playback.play(Path("cheapthrills.mp3"))
        playback.stop()
        self.assertEqual(playback.events, ["play:cheapthrills.mp3", "stop:cheapthrills.mp3"])
        playback.stop()
        self.assertEqual(playback.events, ["play:cheapthrills.mp3", "stop:cheapthrills.mp3"])

    def test_a_new_song_stops_the_previous_one(self) -> None:
        playback = _RecordingPlayback()
        playback.play(Path("cheapthrills.mp3"))
        playback.play(Path("uptownfunk.mp3"))
        self.assertEqual(
            playback.events,
            ["play:cheapthrills.mp3", "stop:cheapthrills.mp3", "play:uptownfunk.mp3"],
        )

    def test_missing_file_leaves_the_current_song_playing(self) -> None:
        playback = _RecordingPlayback()
        playback.play(Path("cheapthrills.mp3"))
        with self.assertRaises(FileNotFoundError):
            playback.play(Path("missing.mp3"))
        self.assertEqual(playback.events, ["play:cheapthrills.mp3"])
        playback.stop()
        self.assertEqual(playback.events, ["play:cheapthrills.mp3", "stop:cheapthrills.mp3"])


class _Capture:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class _Controller:
    def __init__(self) -> None:
        self.cap1 = _Capture()
        self.cap2 = _Capture()

    def release_capture(self) -> None:
        self.cap1.release()
        self.cap2.release()


class JustDanceSessionStopTest(unittest.TestCase):
    def test_leaving_the_game_stops_the_song_and_the_camera(self) -> None:
        session = JustDanceSession.__new__(JustDanceSession)
        session._playback = _RecordingPlayback()
        session._playback.play(Path("callmemaybe.mp3"))
        session._game = type("Game", (), {"controller": _Controller()})()
        session._stage = type("Stage", (), {"close": lambda self: None})()
        session.stop()
        self.assertEqual(
            session._playback.events,
            ["play:callmemaybe.mp3", "stop:callmemaybe.mp3"],
        )
        self.assertTrue(session._game.controller.cap1.released)
        self.assertTrue(session._game.controller.cap2.released)


if __name__ == "__main__":
    unittest.main()
