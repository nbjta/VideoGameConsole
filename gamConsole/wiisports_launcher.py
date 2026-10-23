#3
"""Start Nostalgia AR (devgaddebuilds/wiisports) som tre lokale Vite-servere."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

from gamConsole.child_processes import register_child, stop_child

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor" / "wiisports"

LOBBY_PORT = 5180
PONG_PORT = 5173
FRUIT_PORT = 5174
LOBBY_URL = f"http://127.0.0.1:{LOBBY_PORT}/"

APPS: Tuple[Tuple[str, int], ...] = (
    ("nostalgia-ar", LOBBY_PORT),
    ("backend_handtracking", PONG_PORT),
    ("frontend_game", FRUIT_PORT),
)


def setup_hint() -> str:
    return (
        "Kør: ./scripts/setup_wiisports.sh\n"
        "Kilde: https://github.com/devgaddebuilds/wiisports"
    )


def is_installed() -> bool:
    for folder, _port in APPS:
        vite = REPO / folder / "node_modules" / "vite" / "package.json"
        if not vite.is_file():
            return False
    env_file = REPO / "nostalgia-ar" / ".env.local"
    return env_file.is_file() and (REPO / "nostalgia-ar" / "package.json").is_file()


def ensure_offline_supabase(dest: Optional[Path] = None) -> Optional[Path]:
    #3 createClient("", "") kaster "supabaseUrl is required", og lobbyen tegner aldrig.
    target = dest or (REPO / "nostalgia-ar" / "src" / "lib" / "supabase.ts")
    if dest is None and not target.parent.is_dir():
        return None
    patch = Path(__file__).with_name("nostalgia_ar_supabase.ts").read_text(encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.read_text(encoding="utf-8") == patch:
        return target
    target.write_text(patch, encoding="utf-8")
    return target


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class ViteApp:
    """Én upstream Vite-app. Processen er gruppeleder, så Hjem kan stoppe den."""

    def __init__(self, folder: str, port: int) -> None:
        self.folder = folder
        self.port = port
        self.proc: Optional[subprocess.Popen] = None

    @property
    def cwd(self) -> Path:
        return REPO / self.folder

    def start(self) -> Optional[str]:
        if self.proc is not None and self.proc.poll() is None:
            return None
        if not (self.cwd / "package.json").is_file():
            return f"Mangler {self.cwd / 'package.json'}."
        env = os.environ.copy()
        env["VITE_PINGPONG_URL"] = f"http://127.0.0.1:{PONG_PORT}/"
        env["VITE_FRUIT_URL"] = f"http://127.0.0.1:{FRUIT_PORT}/"
        env["VITE_LOBBY_URL"] = LOBBY_URL
        try:
            self.proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(self.cwd),
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            return f"Kunne ikke starte {self.folder}: {exc}"
        register_child(self.proc, process_group=True)
        return None

    def stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is not None:
            stop_child(proc)

    def exited_with_error(self) -> Optional[int]:
        if self.proc is None:
            return None
        #3 En afsluttet dev-server er en fejl, også når koden er 0. Ellers bliver vinduet hvidt.
        code = self.proc.poll()
        if code is None:
            return None
        return code


class NostalgiaArStack:
    """Lobby plus de to spil, lobbyens PLAY-knapper linker til."""

    def __init__(self) -> None:
        self.apps: List[ViteApp] = [ViteApp(folder, port) for folder, port in APPS]

    def start(self) -> Tuple[bool, str]:
        if not is_installed():
            return False, f"Nostalgia AR er ikke installeret.\n\n{setup_hint()}"
        ensure_offline_supabase()
        for app in self.apps:
            if port_is_open(app.port):
                return False, f"Port {app.port} er allerede i brug ({app.folder})."
            error = app.start()
            if error:
                self.stop()
                return False, error
        return True, "Nostalgia AR starter i browseren. Hjem stopper serverne."

    def stop(self) -> None:
        for app in self.apps:
            app.stop()

    def ready(self) -> bool:
        return all(port_is_open(app.port) for app in self.apps)

    def failure(self) -> Optional[str]:
        for app in self.apps:
            code = app.exited_with_error()
            if code is not None:
                return f"{app.folder} stoppede (kode {code}). Se terminalen."
        return None


_browser = None


def launch(_camera: int = 0) -> Tuple[bool, str]:
    global _browser
    from gamConsole.console_browser import ConsoleBrowser

    stack = NostalgiaArStack()
    ok, msg = stack.start()
    if not ok:
        return False, msg
    deadline = time.time() + 40
    while time.time() < deadline:
        failure = stack.failure()
        if failure:
            stack.stop()
            return False, failure
        if stack.ready():
            _browser = ConsoleBrowser(LOBBY_URL, "gamConsole — Nostalgia AR")
            if not _browser.show():
                stack.stop()
                return False, _browser.error or f"Kunne ikke åbne {LOBBY_URL}"
            return True, "Nostalgia AR kører i gamConsole-browseren. Hjem lukker den."
        time.sleep(0.2)
    stack.stop()
    return False, "Nostalgia AR blev ikke klar. Se terminalen."
