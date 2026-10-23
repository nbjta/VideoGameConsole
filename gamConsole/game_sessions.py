"""Opret indlejrede spil-sessioner til gamConsole (ét vindue)."""

from __future__ import annotations

from typing import Optional, Protocol, Tuple, Union

from gamConsole.builtin_sessions import BuiltinSession
from gamConsole.flying_users_session import FlyingUsersSession
from gamConsole.gurli_gris_session import GurliGrisSession
from gamConsole.games_registry import GameEntry
from gamConsole.handora_session import HandoraSession
from gamConsole.just_dance_session import JustDanceSession
from gamConsole.tilt_of_time_session import TiltOfTimeSession
from gamConsole.wiisports_session import WiiSportsSession

GameSession = Union[
    BuiltinSession,
    FlyingUsersSession,
    GurliGrisSession,
    HandoraSession,
    JustDanceSession,
    TiltOfTimeSession,
    WiiSportsSession,
]


class SessionProtocol(Protocol):
    title: str

    def tick(self): ...

    def stop(self) -> None: ...

    def on_key(self, key: int) -> None: ...


def create_session(entry: GameEntry, camera: int) -> Tuple[Optional[GameSession], str]:
    #6
    if entry.launcher == "handora":
        session = HandoraSession(camera)
        return session, session.message
    #3
    if entry.launcher == "wiisports":
        session = WiiSportsSession(camera)
        return session, session.message
    if entry.launcher == "just_dance":
        return JustDanceSession(entry.mode_arg or "cheapthrills", camera), "Just Dance klar."
    #2
    if entry.launcher == "tilt_of_time":
        session = TiltOfTimeSession(camera)
        return session, session.message
    #9
    if entry.launcher == "builtin" and entry.mode_arg == "gurli_gris":
        return GurliGrisSession(camera), f"{entry.title} klar."
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
