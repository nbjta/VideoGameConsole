#6
"""Handora i spilmenuen, uden kamera og uden at åbne browseren."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamConsole.games_registry import games_for_menu  # noqa: E402
from gamConsole.handora_launcher import (  # noqa: E402
    PAGE_PATH,
    PORT,
    HandoraServer,
    page_url,
    port_is_open,
    setup_hint,
)
from gamConsole.handora_session import _text_frame  # noqa: E402
from gamConsole.menu_opencv import _compose_menu  # noqa: E402
from gamConsole.opencv_ui import (  # noqa: E402
    HOME_Y2,
    QUIT_BAR_H,
    fit_frame,
    set_content_size,
)


class HandoraMenuTest(unittest.TestCase):
    def test_registry_entry(self) -> None:
        entry = next(g for g in games_for_menu() if g.id == "handora")
        self.assertEqual(entry.launcher, "handora")
        self.assertEqual(entry.mode_arg, "hand_gesture")
        self.assertEqual(entry.title, "Handora")
        self.assertIn("BOSS294/Handora", entry.source_url)

    def test_setup_hint_points_at_script(self) -> None:
        self.assertIn("setup_handora.sh", setup_hint())

    def test_page_url_matches_upstream_path(self) -> None:
        self.assertEqual(PORT, 5190)
        self.assertEqual(PAGE_PATH, "/modules/hand_gesture/index.html")
        self.assertEqual(page_url(), "http://127.0.0.1:5190/modules/hand_gesture/index.html")

    def test_start_without_files_explains_setup(self) -> None:
        ok, msg = HandoraServer(root=Path("/tmp/does-not-exist-handora")).start()
        self.assertFalse(ok)
        self.assertIn("setup_handora.sh", msg)

    def test_server_serves_cloned_page(self) -> None:
        import socket

        with tempfile.TemporaryDirectory() as folder:
            page = Path(folder) / "modules" / "hand_gesture"
            page.mkdir(parents=True)
            (page / "index.html").write_text("<title>HANDORA</title>", encoding="utf-8")
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            server = HandoraServer(port=port, root=Path(folder))
            ok, msg = server.start()
            self.assertTrue(ok, msg)
            try:
                deadline = time.time() + 5
                while time.time() < deadline and not server.ready():
                    time.sleep(0.05)
                self.assertTrue(server.ready())
                with urllib.request.urlopen(server.url, timeout=1) as resp:
                    body = resp.read().decode("utf-8")
                self.assertIn("HANDORA", body)
            finally:
                server.stop()
            self.assertFalse(port_is_open(port))

    def test_menu_card_stays_above_quit_bar(self) -> None:
        import gamConsole.menu_opencv as menu

        games = games_for_menu()
        selected = next(i for i, game in enumerate(games) if game.id == "handora")
        webcam = np.zeros((480, 640, 3), dtype=np.uint8)
        _canvas, layout = _compose_menu(webcam, games, selected, None, False, 0)
        card = next(box for box in layout.card_boxes if box.game_index == selected)
        play_x1, play_y1, play_x2, play_y2 = layout.main_play
        self.assertLess(card.y2, menu.CANVAS_H - QUIT_BAR_H)
        self.assertGreater(card.y2, card.y1)
        self.assertLess(play_y2, menu.CANVAS_H - QUIT_BAR_H)
        self.assertGreater(play_x2, play_x1)
        self.assertLess(play_y1, play_y2)

    def test_carousel_composes_every_game_including_clipped_cards(self) -> None:
        #7
        games = games_for_menu()
        webcam = np.zeros((480, 640, 3), dtype=np.uint8)
        for selected in range(len(games)):
            _canvas, layout = _compose_menu(webcam, games, selected, None, False, 0)
            shown = {box.game_index for box in layout.card_boxes}
            self.assertIn(selected, shown)

    def test_status_frame_fills_area_above_quit_bar(self) -> None:
        width, height = 960, 760
        slot_h = height - QUIT_BAR_H - HOME_Y2
        set_content_size(width, slot_h)
        frame = _text_frame(["Handora er åben i gamConsole."])
        self.assertEqual(frame.shape, (slot_h, width, 3))
        canvas = fit_frame(frame, width, height)
        self.assertEqual(canvas.shape, (height, width, 3))


if __name__ == "__main__":
    unittest.main()
