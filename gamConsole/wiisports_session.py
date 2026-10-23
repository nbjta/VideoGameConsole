#3
"""Status i gamConsole mens Nostalgia AR kører i konsol-browseren.

Menuen har sluppet webcamet. Siden kalder getUserMedia uden deviceId.
WebKit-vinduet godkender kameraet. macOS kan stadig spørge første gang.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from gamConsole.console_browser import ConsoleBrowser
from gamConsole.opencv_ui import content_size, draw_text
from gamConsole.wiisports_launcher import LOBBY_URL, NostalgiaArStack


class WiiSportsSession:
    title = "Nostalgia AR"

    def __init__(self, _camera: int) -> None:
        self._stack = NostalgiaArStack()
        self._browser: Optional[ConsoleBrowser] = None
        self._error: Optional[str] = None
        self._opened = False
        self.message = ""
        ok, msg = self._stack.start()
        self.message = msg
        if not ok:
            self._error = msg

    def stop(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        self._stack.stop()

    def on_key(self, key: int) -> None:
        return

    def tick(self) -> Optional[np.ndarray]:
        if self._error:
            return _text_frame(self._error.splitlines())
        failure = self._stack.failure()
        if failure:
            self._error = failure
            return _text_frame([failure])
        if not self._opened and self._stack.ready():
            self._browser = ConsoleBrowser(LOBBY_URL, "gamConsole — Nostalgia AR")
            if self._browser.show():
                self._opened = True
            else:
                self._error = self._browser.error or f"Kunne ikke åbne {LOBBY_URL}"
                return _text_frame([self._error])
        if self._browser is not None:
            self._browser.pump()
            if self._browser.error:
                self._error = self._browser.error
                return _text_frame([self._error])
        lines = [
            "Nostalgia AR kører i konsol-browseren.",
            LOBBY_URL,
            "Bordtennis :5173    Frugt :5174",
            "Tryk Play i lobbyen og start et spil.",
            "Hjem lukker browseren og serverne.",
        ]
        if self._opened:
            lines[0] = "Nostalgia AR er åben i gamConsole."
        return _text_frame(lines)


def _text_frame(lines: List[str]) -> np.ndarray:
    width, height = content_size()
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # #1a1a2e som BGR. (26, 26, 46) blev vist som brun.
    frame[:] = (46, 26, 26)
    line_h = 46
    block = line_h * max(1, len(lines))
    y = max(56, (height - block) // 2 + 30)
    for line in lines:
        if not line:
            y += line_h
            continue
        draw_text(frame, line, (48, y), scale=0.8, color=(234, 234, 234))
        y += line_h
    return frame
