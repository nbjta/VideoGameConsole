#9 #1
"""Kenney-lydene findes, og cue-navnene følger filnavnene."""

from __future__ import annotations

import sys
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamConsole.game_audio import AUDIO_DIR, GameAudio  # noqa: E402


class GameAudioTest(unittest.TestCase):
    def test_cues_are_short_mono_wavs(self) -> None:
        for name in ("click", "positive", "negative", "bubble", "car", "round"):
            path = AUDIO_DIR / f"{name}.wav"
            self.assertTrue(path.is_file(), name)
            with wave.open(str(path), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertGreater(handle.getnframes(), 1000)

    def test_points_pick_positive_or_negative_by_the_filename(self) -> None:
        audio = GameAudio(AUDIO_DIR)
        audio.points(4)
        audio.points(-1)
        audio.bubble()
        audio.click()
        audio.car()
        audio.round_over()
        self.assertEqual(audio.played, ["positive", "negative", "bubble", "click", "car", "round"])

    def test_licenses_are_cc0(self) -> None:
        for name in (
            "LICENSE-interface-sounds.txt",
            "LICENSE-impact-sounds.txt",
            "LICENSE-music-jingles.txt",
        ):
            text = (AUDIO_DIR / name).read_text(encoding="utf-8")
            self.assertIn("CC0", text)


if __name__ == "__main__":
    unittest.main()
