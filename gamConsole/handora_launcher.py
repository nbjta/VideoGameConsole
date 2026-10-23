#6
"""Start HANDORA (BOSS294/Handora) som en lokal statisk side.

Upstream er HTML, CSS og JavaScript uden build. MediaPipe hentes fra jsDelivr,
når siden kører. Siden starter selv kameraet via MediaPipes Camera(), som
kalder getUserMedia uden deviceId. Menuens kamera-index bruges derfor ikke.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

from gamConsole.child_processes import register_child, stop_child

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor" / "handora"
INDEX_REL = Path("modules") / "hand_gesture" / "index.html"

PORT = 5190
PAGE_PATH = "/modules/hand_gesture/index.html"


def setup_hint() -> str:
    return (
        "Kør: ./scripts/setup_handora.sh\n"
        "Kilde: https://github.com/BOSS294/Handora"
    )


def page_url(port: int = PORT) -> str:
    return f"http://127.0.0.1:{port}{PAGE_PATH}"


def port_is_open(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class HandoraServer:
    """Én python-http-server med upstream-repoet som rod."""

    def __init__(self, port: int = PORT, root: Optional[Path] = None) -> None:
        self.port = port
        self.root = root or REPO
        self.proc: Optional[subprocess.Popen] = None

    @property
    def index_file(self) -> Path:
        return self.root / INDEX_REL

    @property
    def url(self) -> str:
        return page_url(self.port)

    def installed(self) -> bool:
        return self.index_file.is_file()

    def start(self) -> Tuple[bool, str]:
        if self.proc is not None and self.proc.poll() is None:
            return True, "Handora-serveren kører allerede."
        if not self.installed():
            return False, f"Handora er ikke installeret.\n\n{setup_hint()}"
        if port_is_open(self.port):
            return False, f"Port {self.port} er allerede i brug (Handora)."
        try:
            self.proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "http.server",
                    str(self.port),
                    "--bind",
                    "127.0.0.1",
                ],
                cwd=str(self.root),
                start_new_session=True,
            )
        except OSError as exc:
            return False, f"Kunne ikke starte Handora: {exc}"
        register_child(self.proc, process_group=True)
        return True, "Handora starter i browseren. Hjem stopper serveren."

    def stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is not None:
            stop_child(proc)

    def ready(self) -> bool:
        if not port_is_open(self.port):
            return False
        try:
            with urllib.request.urlopen(self.url, timeout=0.4) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def failure(self) -> Optional[str]:
        if self.proc is None:
            return None
        code = self.proc.poll()
        if code is None:
            return None
        return f"Handora-serveren stoppede (kode {code}). Se terminalen."


def is_installed() -> bool:
    return HandoraServer().installed()


_browser = None


def launch(_camera: int = 0) -> Tuple[bool, str]:
    global _browser
    import time

    from gamConsole.console_browser import ConsoleBrowser

    server = HandoraServer()
    ok, msg = server.start()
    if not ok:
        return False, msg
    deadline = time.time() + 10
    while time.time() < deadline:
        failure = server.failure()
        if failure:
            server.stop()
            return False, failure
        if server.ready():
            _browser = ConsoleBrowser(server.url, "gamConsole — Handora")
            if not _browser.show():
                server.stop()
                return False, _browser.error or f"Kunne ikke åbne {server.url}"
            return True, "Handora kører i gamConsole-browseren. Hjem lukker den."
        time.sleep(0.05)
    server.stop()
    return False, "Handora blev ikke klar. Se terminalen."
