#2
"""Tilt of Time i spilmenuen, uden at åbne kamera eller spilvindue."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamConsole.games_registry import games_for_menu  # noqa: E402
import numpy as np

from gamConsole.menu_opencv import _compose_menu  # noqa: E402
from gamConsole.opencv_ui import QUIT_BAR_H  # noqa: E402
from gamConsole.tilt_of_time_launcher import bind_camera_index, setup_hint  # noqa: E402


class _FakeCv2:
    def __init__(self) -> None:
        self.calls = []

    def VideoCapture(self, index=0, *args, **kwargs):
        self.calls.append(index)
        return ("cap", index)


class _Module:
    def __init__(self) -> None:
        self.cv2 = _FakeCv2()


class TiltOfTimeMenuTest(unittest.TestCase):
    def test_registry_entry(self) -> None:
        entry = next(g for g in games_for_menu() if g.id == "tilt-of-time")
        self.assertEqual(entry.launcher, "tilt_of_time")
        self.assertEqual(entry.mode_arg, "tilt")
        self.assertEqual(entry.title, "Tilt of Time")
        self.assertIn("ebinrajiiit/Tilt_of_Time", entry.source_url)

    def test_setup_hint_points_at_script(self) -> None:
        self.assertIn("setup_tilt_of_time.sh", setup_hint())

    def test_menu_card_stays_above_quit_bar(self) -> None:
        import gamConsole.menu_opencv as menu

        webcam = np.zeros((480, 640, 3), dtype=np.uint8)
        _canvas, boxes = _compose_menu(webcam, games_for_menu(), None, None, 0)
        tilt = next(box for box in boxes if box.game.id == "tilt-of-time")
        self.assertLess(tilt.y2, menu.CANVAS_H - QUIT_BAR_H)

    def test_bind_camera_replaces_only_hardcoded_zero(self) -> None:
        module = _Module()
        bind_camera_index(module, 2)
        self.assertEqual(module.cv2.VideoCapture(0), ("cap", 2))
        self.assertEqual(module.cv2.VideoCapture(1), ("cap", 1))
        self.assertEqual(module.cv2.VideoCapture("clip.mp4"), ("cap", "clip.mp4"))

    def test_bind_camera_keeps_zero_when_menu_uses_zero(self) -> None:
        module = _Module()
        bind_camera_index(module, 0)
        self.assertEqual(module.cv2.VideoCapture(0), ("cap", 0))


if __name__ == "__main__":
    unittest.main()
