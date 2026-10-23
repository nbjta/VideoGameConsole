#8
"""Værktøjskort i toppen af Husbygning — vælg pensel m.m. med hånden."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from gamConsole.house_paint_assets import blit_brush
from gamConsole.opencv_ui import draw_text

Color = Tuple[int, int, int]


@dataclass(frozen=True)
class ToolCard:
    tool_id: str
    label: str
    paint_color: Optional[Color] = None


#8
TOOL_CARDS: Tuple[ToolCard, ...] = (
    ToolCard("hand", "Tom hånd"),
    ToolCard("brush_red", "Rød pensel", (0, 0, 220)),
    ToolCard("brush_blue", "Blå pensel", (220, 120, 30)),
    ToolCard("brush_green", "Grøn pensel", (40, 180, 60)),
)


class ToolCardBar:
    """Horisontal række med cards øverst på skærmen."""

    BAR_H = 78
    MARGIN = 10
    GAP = 10

    def __init__(self) -> None:
        self.selected_id = "hand"

    def tool_at(self, px: int, py: int, width: int) -> Optional[str]:
        if py < 0 or py > self.BAR_H:
            return None
        for x1, y1, x2, y2, card in self._layout(width):
            if x1 <= px <= x2 and y1 <= py <= y2:
                return card.tool_id
        return None

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, self.BAR_H), (22, 28, 42), -1)
        cv2.line(frame, (0, self.BAR_H), (w, self.BAR_H), (90, 110, 150), 2, cv2.LINE_AA)
        for x1, y1, x2, y2, card in self._layout(w):
            selected = card.tool_id == self.selected_id
            fill = (48, 72, 120) if selected else (32, 42, 62)
            cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
            border = (120, 200, 255) if selected else (70, 85, 110)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2, cv2.LINE_AA)
            _draw_tool_icon(frame, card, x1, y1, x2, y2)
            draw_text(frame, card.label, (x1 + 6, y2 - 22), scale=0.38, color=(230, 238, 255))

    def _layout(self, width: int) -> List[Tuple[int, int, int, int, ToolCard]]:
        n = len(TOOL_CARDS)
        inner = width - 2 * self.MARGIN - self.GAP * (n - 1)
        cw = max(72, inner // n)
        y1, y2 = 8, self.BAR_H - 8
        out: List[Tuple[int, int, int, int, ToolCard]] = []
        x = self.MARGIN
        for card in TOOL_CARDS:
            x2 = min(width - self.MARGIN, x + cw)
            out.append((x, y1, x2, y2, card))
            x = x2 + self.GAP
        return out


def _draw_tool_icon(frame: np.ndarray, card: ToolCard, x1: int, y1: int, x2: int, y2: int) -> None:
    cx = (x1 + x2) // 2
    cy = y1 + (y2 - y1) // 2 - 6
    if card.tool_id == "hand":
        cv2.circle(frame, (cx, cy + 4), 10, (210, 195, 175), -1, cv2.LINE_AA)
        for dx in (-8, 0, 8):
            cv2.line(frame, (cx + dx, cy + 8), (cx + dx, cy - 10), (210, 195, 175), 3, cv2.LINE_AA)
        return
    if card.paint_color is None:
        return
    #8 Brugerens pensel-PNG (rød maling på spidsen).
    icon_h = max(36, min(52, y2 - y1 - 18))
    blit_brush(frame, cx, cy, max_height=icon_h, angle_deg=-38.0)


def paint_color_for(tool_id: str) -> Optional[Color]:
    for card in TOOL_CARDS:
        if card.tool_id == tool_id:
            return card.paint_color
    return None
