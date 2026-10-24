"""De fem sange, der ligger i vendor/just-dance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class DanceSong:
    id: str
    title: str


DANCE_SONGS: Tuple[DanceSong, ...] = (
    DanceSong("cheapthrills", "Cheap Thrills"),
    DanceSong("callmemaybe", "Call Me Maybe"),
    DanceSong("uptownfunk", "Uptown Funk"),
    DanceSong("ghungroo", "Ghungroo"),
    DanceSong("dontstartnow", "Don't Start Now"),
)


def resolve_song(song: Optional[str]) -> str:
    for item in DANCE_SONGS:
        if item.id == song:
            return item.id
    return DANCE_SONGS[0].id


def song_title(song: Optional[str]) -> str:
    chosen = resolve_song(song)
    for item in DANCE_SONGS:
        if item.id == chosen:
            return item.title
    return DANCE_SONGS[0].title
