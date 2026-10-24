"""Sangvalg inde i Just Dance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gamConsole.disco_backdrop import DiscoBackdrop
from gamConsole.just_dance_songs import DANCE_SONGS
from gamConsole.opencv_ui import draw_text, draw_text_centered, measure_text

_FILL = (248, 220, 190)
_CHOSEN = (245, 210, 168)
_BORDER = (210, 175, 120)
_CHOSEN_BORDER = (220, 150, 60)
_TEXT = (110, 70, 30)


@dataclass
class TrackButton:
    x1: int
    y1: int
    x2: int
    y2: int
    song_id: str


class TrackList:
    def __init__(self) -> None:
        self.buttons: List[TrackButton] = []

    def render(self, width: int = 960, height: int = 540, current: Optional[str] = None) -> np.ndarray:
        backdrop = DiscoBackdrop()
        backdrop.t = 14.0
        frame = backdrop.render(width, height)
        draw_text_centered(frame, "Vælg sang", width // 2, int(height * 0.16), scale=0.9, color=(255, 255, 255))
        self._place(frame, int(height * 0.34), int(height * 0.86), current)
        return frame

    def draw_bar(self, frame: np.ndarray, current: Optional[str]) -> None:
        height, width = frame.shape[:2]
        bar_h = max(56, height // 10)
        y1 = height - bar_h
        cv2.rectangle(frame, (0, y1), (width - 1, height - 1), (36, 18, 48), -1)
        self._place(frame, y1 + 8, height - 8, current)

    def hit(self, x: int, y: int) -> Optional[str]:
        for button in self.buttons:
            if button.x1 <= x <= button.x2 and button.y1 <= y <= button.y2:
                return button.song_id
        return None

    def _place(self, frame: np.ndarray, top: int, bottom: int, current: Optional[str]) -> None:
        height, width = frame.shape[:2]
        scale = 0.46
        pad_x, pad_y, gap = 14, 8, 8
        sizes = []
        for song in DANCE_SONGS:
            text_w, text_h = measure_text(song.title, scale)
            sizes.append((text_w + pad_x * 2, text_h + pad_y * 2))
        rows = _rows_that_fit(sizes, width - 16, gap)
        row_gap = 8
        block_h = sum(max(h for _w, h in row) for row in rows) + row_gap * (len(rows) - 1)
        y = max(top, min(top, bottom - block_h))
        if y + block_h > bottom:
            y = max(0, bottom - block_h)
        self.buttons = []
        song_i = 0
        for row in rows:
            row_h = max(h for _w, h in row)
            row_w = sum(w for w, _h in row) + gap * (len(row) - 1)
            x = max(8, (width - row_w) // 2)
            for button_w, button_h in row:
                song = DANCE_SONGS[song_i]
                x1, y1 = x, y + (row_h - button_h) // 2
                x2, y2 = min(width - 1, x1 + button_w), min(height - 1, y1 + button_h)
                chosen = song.id == current
                cv2.rectangle(frame, (x1, y1), (x2, y2), _CHOSEN if chosen else _FILL, -1, cv2.LINE_AA)
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    _CHOSEN_BORDER if chosen else _BORDER,
                    2,
                    cv2.LINE_AA,
                )
                text_w, text_h = measure_text(song.title, scale)
                draw_text(
                    frame,
                    song.title,
                    (x1 + (button_w - text_w) // 2, y1 + (button_h - text_h) // 2),
                    scale=scale,
                    color=_TEXT,
                )
                self.buttons.append(TrackButton(x1, y1, x2, y2, song.id))
                x += button_w + gap
                song_i += 1
            y += row_h + row_gap


def _rows_that_fit(
    sizes: List[Tuple[int, int]],
    max_width: int,
    gap: int,
) -> List[List[Tuple[int, int]]]:
    rows: List[List[Tuple[int, int]]] = []
    current: List[Tuple[int, int]] = []
    used = 0
    for size in sizes:
        extra = size[0] if not current else gap + size[0]
        if current and used + extra > max_width:
            rows.append(current)
            current = [size]
            used = size[0]
        else:
            current.append(size)
            used += extra
    if current:
        rows.append(current)
    return rows
