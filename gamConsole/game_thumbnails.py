"""Spil-billeder til menu-gitter (PNG i assets eller genererede ikoner)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
THUMB_DIR = ROOT / "assets" / "game_thumbs"
DEFAULT_SIZE = 128

_cache: Dict[str, np.ndarray] = {}


def clear_thumbnail_cache() -> None:
    _cache.clear()


#8
def _draw_webcam_3d(img: np.ndarray) -> None:
    cv2.rectangle(img, (18, 78), (110, 108), (150, 154, 152), -1)
    cv2.rectangle(img, (28, 70), (100, 80), (176, 178, 174), -1)
    cv2.line(img, (24, 58), (104, 46), (90, 140, 190), 8, cv2.LINE_AA)


def _draw_bubbles(img: np.ndarray) -> None:
    colors = [(255, 120, 80), (80, 220, 120), (120, 160, 255)]
    for i, (x, y, r) in enumerate([(40, 50, 22), (88, 72, 18), (70, 38, 14)]):
        cv2.circle(img, (x, y), r, colors[i], 2, cv2.LINE_AA)


def _draw_debug(img: np.ndarray) -> None:
    cv2.rectangle(img, (8, 20, 60, 108), (60, 60, 80), -1)
    cv2.rectangle(img, (68, 20, 120, 108), (40, 40, 40), -1)
    cv2.line(img, (8, 64), (120, 64), (0, 255, 120), 1)


def _draw_flying_users(img: np.ndarray) -> None:
    cv2.circle(img, (64, 48), 22, (255, 200, 120), -1, cv2.LINE_AA)
    cv2.circle(img, (56, 44), 3, (30, 30, 30), -1)
    cv2.circle(img, (72, 44), 3, (30, 30, 30), -1)
    cv2.ellipse(img, (90, 88), (18, 10), 0, 0, 360, (40, 140, 220), -1)


def _draw_just_dance(img: np.ndarray) -> None:
    cv2.line(img, (64, 30), (64, 70), (255, 180, 220), 4, cv2.LINE_AA)
    cv2.line(img, (64, 42), (40, 58), (255, 180, 220), 3, cv2.LINE_AA)
    cv2.line(img, (64, 42), (88, 50), (255, 180, 220), 3, cv2.LINE_AA)
    cv2.line(img, (64, 70), (48, 100), (255, 180, 220), 3, cv2.LINE_AA)
    cv2.line(img, (64, 70), (80, 100), (255, 180, 220), 3, cv2.LINE_AA)


def _draw_tilt(img: np.ndarray) -> None:
    cv2.circle(img, (64, 64), 40, (220, 200, 120), 2, cv2.LINE_AA)
    cv2.line(img, (64, 64), (64, 32), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(img, (64, 64), (88, 76), (255, 120, 80), 2, cv2.LINE_AA)


def _draw_wiisports(img: np.ndarray) -> None:
    cv2.rectangle(img, (20, 48, 108, 56), (240, 240, 250), -1)
    cv2.circle(img, (48, 52), 6, (255, 80, 60), -1)
    cv2.circle(img, (80, 52), 6, (255, 220, 60), -1)


#6
def _draw_handora(img: np.ndarray) -> None:
    cv2.line(img, (28, 96), (96, 32), (180, 255, 80), 3, cv2.LINE_AA)
    cv2.circle(img, (96, 32), 8, (255, 180, 80), -1, cv2.LINE_AA)
    cv2.circle(img, (44, 72), 12, (80, 120, 255), 2, cv2.LINE_AA)


_GENERATORS = {
    "webcam-3d": _draw_webcam_3d,
    "webcam-bub": _draw_bubbles,
    "webcam-debug": _draw_debug,
    "flying-users-ar": _draw_flying_users,
    "just-dance": _draw_just_dance,
    "tilt-of-time": _draw_tilt,
    "wiisports": _draw_wiisports,
    "handora": _draw_handora,
}


def _generate(game_id: str, size: int) -> np.ndarray:
    base = 128
    img = np.zeros((base, base, 3), dtype=np.uint8)
    img[:] = (52, 56, 74)
    cv2.rectangle(img, (6, 6), (base - 7, base - 7), (120, 130, 165), 2, cv2.LINE_AA)
    draw = _GENERATORS.get(game_id)
    if draw is not None:
        draw(img)
    else:
        cv2.putText(img, "?", (base // 2 - 8, base // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    if size != base:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


def get_game_thumbnail(game_id: str, size: int = DEFAULT_SIZE) -> np.ndarray:
    key = f"{game_id}:{size}"
    if key in _cache:
        return _cache[key]

    png = THUMB_DIR / f"{game_id}.png"
    if png.is_file():
        loaded = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if loaded is not None:
            out = cv2.resize(loaded, (size, size), interpolation=cv2.INTER_AREA)
            _cache[key] = out
            return out

    out = _generate(game_id, size)
    _cache[key] = out
    return out


def paste_thumbnail(
    canvas: np.ndarray,
    thumb: np.ndarray,
    x: int,
    y: int,
    box_w: int,
    box_h: int,
) -> int:
    """Placer thumbnail i kort. Returnerer bredde brugt (til tekst-offset)."""
    margin = 8
    side = min(88, box_w // 2 - 6, max(48, box_h - 38))
    small = cv2.resize(thumb, (side, side), interpolation=cv2.INTER_CUBIC)
    y0 = y + margin + max(0, (box_h - side - 34) // 2)
    x0 = x + margin
    y1, x1 = y0 + side, x0 + side
    if y1 > canvas.shape[0] or x1 > canvas.shape[1]:
        return side + margin * 2
    cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (238, 240, 248), -1)
    cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (90, 100, 130), 2, cv2.LINE_AA)
    canvas[y0:y1, x0:x1] = small
    return side + margin * 2
