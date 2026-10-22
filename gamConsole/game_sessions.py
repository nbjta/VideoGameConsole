"""Opret indlejrede spil-sessioner til gamConsole (ét vindue)."""

from __future__ import annotations

from typing import Optional, Protocol, Tuple, Union

from gamConsole.builtin_sessions import BuiltinSession
from gamConsole.flying_users_session import FlyingUsersSession
from gamConsole.games_registry import GameEntry
from gamConsole.jdwebcam_session import JDWebcamSession
from gamConsole.just_dance_session import JustDanceSession

GameSession = Union[BuiltinSession, FlyingUsersSession, JustDanceSession, JDWebcamSession]


class SessionProtocol(Protocol):
    title: str

    def tick(self): ...

    def stop(self) -> None: ...

    def on_key(self, key: int) -> None: ...


def create_session(entry: GameEntry, camera: int) -> Tuple[Optional[GameSession], str]:
    if entry.launcher == "just_dance":
        return JustDanceSession(entry.mode_arg or "cheapthrills", camera), "Just Dance klar."
    if entry.launcher == "jdwebcam":
        from gamConsole.jdwebcam_launcher import launch as launch_jdwebcam

        ok, msg = launch_jdwebcam()
        session = JDWebcamSession(camera)
        if not ok:
            return session, msg
        return session, msg + " Preview i gamConsole — Hjem for menu."
    if entry.launcher == "builtin" and entry.mode_arg == "flying_users":
        return FlyingUsersSession(camera), f"{entry.title} klar."
    if entry.launcher == "builtin" or entry.launcher == "cvgames":
        mode = entry.mode_arg or "debug"
        if entry.launcher == "cvgames":
            from gamConsole.cvgames_launcher import cvgames_binary

            if cvgames_binary() is not None:
                mode = entry.mode_arg or "debug"
        return BuiltinSession(mode, camera), f"{entry.title} (samme vindue)."
    return None, "Ukendt spiltype."
