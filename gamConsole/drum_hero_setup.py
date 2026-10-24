#16
"""Nummer og sværhedsgrad før en Drum Hero-runde.

Filerne er målt med afinfo.
#24 Hver sang har sine egne slag i sekunder. Trommen er et gæt fra frekvensbåndet.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from gamConsole.drum_hits import HITS
from gamConsole.drum_track import SONG_ARTIST, SONG_PAGE, SONG_PATH, SONG_TITLE
from gamConsole.opencv_ui import draw_text, draw_text_centered, measure_text

FOLDER = SONG_PATH.parent
#16 Samme bane-numre som HIHAT, SNARE, TOM og CRASH.
_HIHAT, _SNARE, _TOM, _CRASH = 0, 1, 2, 3
_PATTERN = (_SNARE, _TOM, _HIHAT, _CRASH)


#24 Et afkodet slag. seconds er inde i filen, lane er hi-hat, lilletromme, tam eller crash.
class DrumHit:
    """Et slag, der kan blive en node."""

    def __init__(self, milliseconds: int, lane: int) -> None:
        self.milliseconds = milliseconds
        self.seconds = milliseconds / 1000.0
        self.lane = lane


#24 Let springer slag over, når det større træf ellers ville dække to noder.
def _spaced(hits: Sequence[DrumHit], gap: float, duration: float) -> Tuple[DrumHit, ...]:
    """Behold slag, der har luft nok til træffet, også hen over loopet."""
    last: dict = {}
    kept: List[DrumHit] = []
    for hit in hits:
        previous = last.get(hit.lane)
        if previous is not None and hit.seconds - previous < gap:
            continue
        kept.append(hit)
        last[hit.lane] = hit.seconds
    first: dict = {}
    for hit in kept:
        first.setdefault(hit.lane, hit.seconds)
    trimmed: List[DrumHit] = []
    for hit in kept:
        if hit.seconds != first[hit.lane] and first[hit.lane] + duration - hit.seconds < gap:
            continue
        trimmed.append(hit)
    return tuple(trimmed)


class DrumSong:
    """Et Pixabay-nummer, der må afspilles inde i spillet."""

    def __init__(self, song_id: str, title: str, artist: str, path: Path, page: str, seconds: float) -> None:
        self.id = song_id
        self.title = title
        self.artist = artist
        self.path = path
        self.page = page
        self.seconds = seconds
        #24 Listen er målt på netop denne fil.
        self.hits = tuple(DrumHit(ms, lane) for ms, lane in HITS[song_id])

    def line(self) -> str:
        whole = int(round(self.seconds))
        return f"{self.artist} · {whole // 60}:{whole % 60:02d}"


#16 108,048 s, 116,219 s og 102,744 s er afinfo, ikke gæt.
#17 103,393 s er afinfo. Siden siger 1:43.
#18 72,229 s er afinfo. Siden siger 1:12.
SONGS: Tuple[DrumSong, ...] = (
    DrumSong("588419", SONG_TITLE, SONG_ARTIST, SONG_PATH, SONG_PAGE, 108.048),
    DrumSong(
        "617071",
        "Upbeat - Upbeat Music",
        "NastelBom",
        FOLDER / "upbeat-music-617071.mp3",
        "https://pixabay.com/music/upbeat-upbeat-music-617071/",
        116.21875,
    ),
    DrumSong(
        "587078",
        "Upbeat",
        "Sub_Clair",
        FOLDER / "upbeat-587078.mp3",
        "https://pixabay.com/music/beats-upbeat-587078/",
        102.744,
    ),
    DrumSong(
        "614901",
        "Upbeat Funk",
        "Kulakovka",
        FOLDER / "upbeat-funk-614901.mp3",
        "https://pixabay.com/music/funk-upbeat-funk-614901/",
        103.392625,
    ),
    DrumSong(
        "622901",
        "Energetic Action Sport",
        "JonasBlakewood",
        FOLDER / "energetic-action-sport-622901.mp3",
        "https://pixabay.com/music/rock-energetic-action-sport-622901/",
        72.228562,
    ),
)


class Difficulty:
    """Hvor tæt noderne kommer, og hvor stort træffet er."""

    id = ""
    title = ""
    detail = ""
    beat = 1.15
    hit_early = 0.22
    hit_late = 0.18

    def windows(self) -> Tuple[float, float]:
        return (self.hit_early, self.hit_late)

    def select(self, hits: Sequence[DrumHit], duration: float) -> Tuple[DrumHit, ...]:
        del duration
        return tuple(hits)

    def lanes(self, index: int) -> Tuple[int, ...]:
        return (_PATTERN[index % len(_PATTERN)],)


class EasyDifficulty(Difficulty):
    id = "let"
    title = "Let"
    detail = "Færre slag og større træf"
    beat = 1.70
    hit_early = 0.34
    hit_late = 0.28

    def select(self, hits: Sequence[DrumHit], duration: float) -> Tuple[DrumHit, ...]:
        #24 Større træf kræver mere luft mellem to slag på samme tromme.
        return _spaced(hits, self.hit_early + self.hit_late + 0.02, duration)


class MediumDifficulty(Difficulty):
    id = "mellem"
    title = "Mellem"
    detail = "De slag, der kunne skelnes"


class HardDifficulty(Difficulty):
    #24 Samme slag som mellem. Svær er et mindre træf, ikke flere noder.
    id = "svaer"
    title = "Svær"
    detail = "Samme slag og mindre træf"
    beat = 0.70
    hit_early = 0.14
    hit_late = 0.10

    def lanes(self, index: int) -> Tuple[int, ...]:
        #16 Siden rammer hi-hat og crash. Slag ned rammer lilletromme og tam.
        return ((_HIHAT, _CRASH), (_SNARE, _TOM))[index % 2]


EASY = EasyDifficulty()
MEDIUM = MediumDifficulty()
HARD = HardDifficulty()
DIFFICULTIES: Tuple[Difficulty, ...] = (EASY, MEDIUM, HARD)


class SetupRow:
    """En række i listen. rect er venstre, top, højre, bund."""

    def __init__(self, kind: str, key: str, rect: Tuple[int, int, int, int]) -> None:
        self.kind = kind
        self.key = key
        self.rect = rect

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


class DrumSetup:
    """Lodret liste. Runden starter først, når der trykkes Spil."""

    def __init__(self) -> None:
        self.open = True
        self.song = SONGS[0]
        self.difficulty: Difficulty = MEDIUM
        self.rows: List[SetupRow] = []
        self.play_rect: Optional[Tuple[int, int, int, int]] = None

    def press(self, x: int, y: int) -> Optional[str]:
        if self.play_rect is not None and _contains(self.play_rect, x, y):
            return "play"
        for row in self.rows:
            if not row.contains(x, y):
                continue
            if row.kind == "song":
                self.song = _song(row.key)
            elif row.kind == "difficulty":
                self.difficulty = _difficulty(row.key)
            return row.kind
        return None

    def draw(self, width: int, height: int) -> np.ndarray:
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:] = (22, 18, 28)
        margin_x = max(12, int(width * 0.07))
        margin_y = max(8, int(height * 0.03))
        play_h = max(32, int(height * 0.10))
        label_h = max(16, int(height * 0.04))
        gap = max(3, int(height * 0.008))
        #17 Rækkerne følger antallet af numre, så Spil bliver under listen.
        row_count = len(SONGS) + len(DIFFICULTIES)
        gap_count = 2 + row_count
        fixed = margin_y * 2 + label_h * 2 + play_h + gap * gap_count
        row_h = max(18, (height - fixed) // row_count)
        y = margin_y
        draw_text_centered(frame, "Vælg nummer", width // 2, y, scale=_label_scale(height), color=(240, 236, 230))
        y += label_h + gap
        self.rows = []
        for song in SONGS:
            rect = (margin_x, y, width - margin_x, y + row_h)
            _draw_choice(frame, rect, song.title, song.line(), song.id == self.song.id, row_h)
            self.rows.append(SetupRow("song", song.id, rect))
            y += row_h + gap
        draw_text_centered(frame, "Sværhedsgrad", width // 2, y, scale=_label_scale(height), color=(240, 236, 230))
        y += label_h + gap
        for level in DIFFICULTIES:
            rect = (margin_x, y, width - margin_x, y + row_h)
            _draw_choice(frame, rect, level.title, level.detail, level.id == self.difficulty.id, row_h)
            self.rows.append(SetupRow("difficulty", level.id, rect))
            y += row_h + gap
        play_top = height - margin_y - play_h
        self.play_rect = (margin_x, play_top, width - margin_x, height - margin_y)
        x1, y1, x2, y2 = self.play_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (52, 150, 72), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (210, 245, 220), 2, cv2.LINE_AA)
        _draw_line(frame, "Spil", (x1, y1, x2, y2), 0.62, (255, 255, 255))
        return frame


def _song(song_id: str) -> DrumSong:
    for song in SONGS:
        if song.id == song_id:
            return song
    return SONGS[0]


def _difficulty(level_id: str) -> Difficulty:
    for level in DIFFICULTIES:
        if level.id == level_id:
            return level
    return MEDIUM


def _contains(rect: Tuple[int, int, int, int], x: int, y: int) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _label_scale(height: int) -> float:
    if height < 280:
        return 0.42
    return 0.62


def _draw_choice(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    title: str,
    detail: str,
    selected: bool,
    row_h: int,
) -> None:
    x1, y1, x2, y2 = rect
    fill = (46, 58, 78) if selected else (38, 34, 48)
    border = (80, 210, 255) if selected else (96, 90, 112)
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 3 if selected else 1, cv2.LINE_AA)
    if row_h < 40:
        _draw_line(frame, title, rect, 0.40, (245, 245, 245))
        return
    mid = y1 + int((y2 - y1) * 0.42)
    _draw_line(frame, title, (x1, y1 + 2, x2, mid), 0.46, (245, 245, 245))
    _draw_line(frame, detail, (x1, mid, x2, y2 - 2), 0.34, (190, 186, 200))


def _draw_line(
    frame: np.ndarray,
    text: str,
    rect: Tuple[int, int, int, int],
    scale: float,
    color: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = rect
    room = max(8, x2 - x1 - 16)
    fitted = scale
    while fitted > 0.28 and measure_text(text, fitted)[0] > room:
        fitted -= 0.02
    text_w, text_h = measure_text(text, fitted)
    x = x1 + max(8, ((x2 - x1) - text_w) // 2)
    y = y1 + max(0, ((y2 - y1) - text_h) // 2)
    draw_text(frame, text, (x, y), scale=fitted, color=color)


def song_ids() -> Sequence[str]:
    return tuple(song.id for song in SONGS)
