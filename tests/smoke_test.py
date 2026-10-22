"""Headless smoke tests for gamConsole (no GUI)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamConsole.cvgames_launcher import cvgames_binary, launch_cvgames  # noqa: E402
from gamConsole.games_registry import GAMES  # noqa: E402
from gamConsole.webcam_preview import WebcamPreview  # noqa: E402


class SmokeTest(unittest.TestCase):
    def test_game_registry(self) -> None:
        self.assertGreaterEqual(len(GAMES), 7)
        ids = {g.id for g in GAMES}
        self.assertIn("just-dance", ids)
        self.assertIn("jdwebcam", ids)
        # issue #1
        self.assertIn("flying-users-ar", ids)

    def test_webcam_preview_import(self) -> None:
        preview = WebcamPreview(0)
        self.assertTrue(hasattr(preview, "start"))

    def test_cvgames_launch(self) -> None:
        binary = cvgames_binary()
        ok, msg = launch_cvgames("debug", camera=0)
        if binary is None:
            self.assertFalse(ok)
            self.assertIn("CvGames", msg)
        else:
            self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
