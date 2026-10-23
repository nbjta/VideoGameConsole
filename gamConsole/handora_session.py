#6
"""Status i gamConsole mens Handora kører i konsol-browseren.

Menuen har sluppet webcamet. Siden starter kameraet selv via MediaPipe Camera().
WebKit-vinduet godkender kameraet. macOS kan stadig spørge første gang.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from gamConsole.console_browser import ConsoleBrowser
from gamConsole.handora_launcher import PAGE_PATH, PORT, HandoraServer
from gamConsole.opencv_ui import content_size, draw_text


class HandoraSession:
    title = "Handora"

    def __init__(self, _camera: int) -> None:
        self._server = HandoraServer()
        self._browser: Optional[ConsoleBrowser] = None
        self._error: Optional[str] = None
        self._opened = False
        self.message = ""
        ok, msg = self._server.start()
        self.message = msg
        if not ok:
            self._error = msg

    def stop(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        self._server.stop()

    def on_key(self, key: int) -> None:
        return

    def tick(self) -> Optional[np.ndarray]:
        if self._error:
            return _text_frame(self._error.splitlines())
        failure = self._server.failure()
        if failure:
            self._error = failure
            return _text_frame([failure])
        if not self._opened and self._server.ready():
            self._browser = ConsoleBrowser(self._server.url, "gamConsole — Handora")
            if self._browser.show():
                self._opened = True
            else:
                self._error = self._browser.error or f"Kunne ikke åbne {self._server.url}"
                return _text_frame([self._error])
        if self._browser is not None:
            self._browser.pump()
            if self._browser.error:
                self._error = self._browser.error
                return _text_frame([self._error])
        lines = [
            "Handora kører i konsol-browseren.",
            f"127.0.0.1:{PORT}{PAGE_PATH}",
            "Tryk Start Camera, derefter Open Games.",
            "Laser og Hit-The-Ball ligger under Open Games.",
            "MediaPipe hentes fra jsDelivr. Internet skal være tændt.",
            "Hjem lukker browseren og serveren.",
        ]
        if self._opened:
            lines[0] = "Handora er åben i gamConsole."
        return _text_frame(lines)


def _text_frame(lines: List[str]) -> np.ndarray:
    width, height = content_size()
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (46, 26, 26)
    line_h = 46
    block = line_h * max(1, len(lines))
    y = max(56, (height - block) // 2 + 30)
    for line in lines:
        if not line:
            y += line_h
            continue
        draw_text(frame, line, (48, y), scale=0.7, color=(234, 234, 234))
        y += line_h
    return frame
