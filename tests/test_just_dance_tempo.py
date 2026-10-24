"""Danseren skal følge sangens tid, ikke hvor lang tid et billede tager at tegne."""

from __future__ import annotations

import unittest

from gamConsole.just_dance_session import JustDanceSession
from gamConsole.just_dance_tempo import DanceTempo


class DanceTempoTest(unittest.TestCase):
    def test_slow_loop_skips_ahead_to_the_song(self) -> None:
        tempo = DanceTempo(30.0, now=0.0)
        shown = []
        t = 0.0
        for _ in range(10):
            for _read in range(tempo.frames_to_read(t)):
                tempo.note_read()
            shown.append(tempo.shown)
            t += 0.05
        self.assertEqual(shown[0], 0)
        self.assertEqual(tempo.shown, int(0.45 * 30.0))
        self.assertEqual(tempo.due(0.45), tempo.shown)

    def test_fast_loop_holds_the_frame(self) -> None:
        tempo = DanceTempo(30.0, now=0.0)
        self.assertEqual(tempo.frames_to_read(0.0), 1)
        tempo.note_read()
        self.assertEqual(tempo.frames_to_read(0.01), 0)
        self.assertEqual(tempo.frames_to_read(0.02), 0)
        self.assertEqual(tempo.frames_to_read(1.0 / 30.0), 1)

    def test_file_fps_is_used_for_24fps_songs(self) -> None:
        tempo = DanceTempo(23.976, now=0.0)
        tempo.note_read()
        elapsed = 2.0
        while tempo.frames_to_read(elapsed):
            tempo.note_read()
        self.assertEqual(tempo.shown, int(elapsed * 23.976 + 1e-6))

    def test_score_runs_once_per_hundred_video_frames(self) -> None:
        tempo = DanceTempo(30.0, now=0.0)
        self.assertFalse(tempo.should_score())
        tempo.note_read()
        self.assertTrue(tempo.should_score())
        self.assertFalse(tempo.should_score())
        for _ in range(99):
            tempo.note_read()
        self.assertFalse(tempo.should_score())
        tempo.note_read()
        self.assertEqual(tempo.shown, 100)
        self.assertTrue(tempo.should_score())
        self.assertFalse(tempo.should_score())


class _ScriptedTempo(DanceTempo):
    def __init__(self, plan: list) -> None:
        super().__init__(30.0, now=0.0)
        self.plan = plan

    def frames_to_read(self, now: float | None = None) -> int:
        del now
        if self.plan:
            return self.plan.pop(0)
        return 0


class _FakeCapture:
    def __init__(self) -> None:
        self.reads = 0

    def read(self):
        self.reads += 1
        return True, self.reads


class PullVideoFrameTest(unittest.TestCase):
    def test_late_tick_reads_the_due_frames_then_holds(self) -> None:
        session = JustDanceSession.__new__(JustDanceSession)
        session._frame1 = None
        capture = _FakeCapture()
        tempo = _ScriptedTempo([3, 0])
        first = session._pull_video_frame(capture, tempo)
        second = session._pull_video_frame(capture, tempo)
        self.assertEqual(first, 3)
        self.assertEqual(second, 3)
        self.assertEqual(capture.reads, 3)
        self.assertEqual(tempo.shown, 2)


if __name__ == "__main__":
    unittest.main()
