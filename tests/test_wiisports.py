#3
"""Nostalgia AR i spilmenuen, uden npm og uden browser."""

from __future__ import annotations

import socket
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamConsole.games_registry import games_for_menu  # noqa: E402
from gamConsole.menu_opencv import _compose_menu  # noqa: E402
from gamConsole.opencv_ui import (  # noqa: E402
    HOME_Y2,
    QUIT_BAR_H,
    fit_frame,
    set_content_size,
    window_content_height,
)
from gamConsole.wiisports_session import _text_frame  # noqa: E402
from gamConsole.wiisports_launcher import (  # noqa: E402
    FRUIT_PORT,
    LOBBY_PORT,
    LOBBY_URL,
    PONG_PORT,
    ensure_offline_supabase,
    port_is_open,
    setup_hint,
)


class WiiSportsMenuTest(unittest.TestCase):
    def test_registry_entry(self) -> None:
        entry = next(g for g in games_for_menu() if g.id == "wiisports")
        self.assertEqual(entry.launcher, "wiisports")
        self.assertEqual(entry.mode_arg, "lobby")
        self.assertEqual(entry.title, "Nostalgia AR")
        self.assertIn("devgaddebuilds/wiisports", entry.source_url)

    def test_setup_hint_points_at_script(self) -> None:
        self.assertIn("setup_wiisports.sh", setup_hint())

    def test_offline_supabase_does_not_call_create_client_with_empty_url(self) -> None:
        target = Path(__file__).resolve().parents[1] / "gamConsole" / "nostalgia_ar_supabase.ts"
        text = target.read_text(encoding="utf-8")
        self.assertIn("offlineSupabase", text)
        self.assertNotIn('createClient(url ?? ""', text)
        with tempfile.TemporaryDirectory() as folder:
            written = ensure_offline_supabase(Path(folder) / "supabase.ts")
            self.assertIsNotNone(written)
            assert written is not None
            self.assertEqual(written.read_text(encoding="utf-8"), text)

    def test_console_browser_grants_camera_and_keeps_lobby_url(self) -> None:
        from WebKit import WKPermissionDecisionGrant

        from gamConsole.console_browser import ConsoleBrowser, _DelegateClass

        browser = ConsoleBrowser(LOBBY_URL, "gamConsole — Nostalgia AR")
        self.assertEqual(browser.url, "http://127.0.0.1:5180/")
        decisions = []
        delegate = _DelegateClass.alloc().init()
        delegate.browser = browser
        delegate.webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(
            None, None, None, 0, decisions.append
        )
        self.assertEqual(decisions, [WKPermissionDecisionGrant])

    def test_local_ports_match_upstream_dev_scripts(self) -> None:
        self.assertEqual(LOBBY_PORT, 5180)
        self.assertEqual(PONG_PORT, 5173)
        self.assertEqual(FRUIT_PORT, 5174)
        self.assertEqual(LOBBY_URL, "http://127.0.0.1:5180/")

    def test_last_menu_card_stays_above_quit_bar(self) -> None:
        import gamConsole.menu_opencv as menu

        old = menu.CANVAS_H
        try:
            menu.CANVAS_H = window_content_height(840, visible_height=796)
            webcam = np.zeros((480, 640, 3), dtype=np.uint8)
            canvas, boxes = _compose_menu(webcam, games_for_menu(), None, None, 0)
            card = next(box for box in boxes if box.game.id == "wiisports")
            self.assertEqual(canvas.shape[0], menu.CANVAS_H)
            self.assertLess(card.y2, menu.CANVAS_H - QUIT_BAR_H)
            self.assertGreater(card.y2, card.y1)
        finally:
            menu.CANVAS_H = old

    def test_window_height_fits_this_screen(self) -> None:
        self.assertEqual(window_content_height(840, visible_height=796), 760)

    def test_status_frame_fills_area_above_quit_bar(self) -> None:
        width, height = 960, 760
        slot_h = height - QUIT_BAR_H - HOME_Y2
        set_content_size(width, slot_h)
        frame = _text_frame(["Nostalgia AR er i gang i browseren."])
        self.assertEqual(frame.shape, (slot_h, width, 3))
        canvas = fit_frame(frame, width, height)
        self.assertEqual(tuple(int(v) for v in canvas[HOME_Y2, width // 2]), tuple(int(v) for v in frame[0, width // 2]))
        self.assertEqual(
            tuple(int(v) for v in canvas[height - QUIT_BAR_H - 1, width // 2]),
            tuple(int(v) for v in frame[slot_h - 1, width // 2]),
        )
        self.assertGreater(int(canvas[HOME_Y2, width // 2, 0]), 0)

    def test_port_is_open_follows_a_real_listener(self) -> None:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        try:
            self.assertTrue(port_is_open(port))
        finally:
            sock.close()
        self.assertFalse(port_is_open(port))


if __name__ == "__main__":
    unittest.main()
