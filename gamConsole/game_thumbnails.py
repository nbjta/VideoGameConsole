"""Spil-billeder til menu — miniudgaver af spillets baggrund hvor muligt."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
THUMB_DIR = ROOT / "assets" / "game_thumbs"
DEFAULT_SIZE = 128

_cache: Dict[str, np.ndarray] = {}


def clear_thumbnail_cache() -> None:
    _cache.clear()


RenderFn = Callable[[int, int], np.ndarray]


#9
def _render_gurli_gris(width: int, height: int) -> np.ndarray:
    from gamConsole.gurli_gris import GurliGrisBackdrop

    return GurliGrisBackdrop().render(width, height)


#1
def _render_flying_users(width: int, height: int) -> np.ndarray:
    from gamConsole.adventure_city import AdventureCityBackdrop

    city = AdventureCityBackdrop()
    city.t = 18.0
    return city.render(width, height)


#8
def _render_husbygning(width: int, height: int) -> np.ndarray:
    from gamConsole.house_build import HouseBuildScene

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    scene = HouseBuildScene()
    scene.beam_u = 0.38
    scene.beam_v = 0.62
    scene.draw(frame)
    return frame


def _render_bubbles(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        u = y / max(height - 1, 1)
        frame[y, :] = (int(40 + 30 * u), int(50 + 40 * u), int(70 + 50 * u))
    colors = [(255, 120, 80), (80, 220, 120), (120, 160, 255)]
    for i, (nx, ny, r) in enumerate([(0.32, 0.42, 0.18), (0.68, 0.58, 0.14), (0.55, 0.28, 0.11)]):
        cv2.circle(
            frame,
            (int(nx * width), int(ny * height)),
            max(8, int(r * min(width, height))),
            colors[i],
            2,
            cv2.LINE_AA,
        )
    return frame


def _render_debug(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, : width // 2] = (60, 60, 80)
    frame[:, width // 2 :] = (40, 40, 40)
    cv2.line(frame, (0, height // 2), (width, height // 2), (0, 255, 120), 1)
    return frame


def _render_just_dance(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 20, 30)
    cx, cy = width // 2, int(height * 0.38)
    cv2.line(frame, (cx, cy), (cx, cy + int(height * 0.32)), (255, 180, 220), 4, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + 8), (int(width * 0.3), int(height * 0.55)), (255, 180, 220), 3, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + 8), (int(width * 0.7), int(height * 0.48)), (255, 180, 220), 3, cv2.LINE_AA)
    return frame


def _render_tilt(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 35, 55)
    cv2.circle(frame, (width // 2, height // 2), min(width, height) // 3, (220, 200, 120), 2, cv2.LINE_AA)
    cv2.line(frame, (width // 2, height // 2), (width // 2, height // 4), (255, 255, 255), 3, cv2.LINE_AA)
    return frame


def _render_wiisports(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (180, 140, 90)
    cv2.rectangle(frame, (int(width * 0.15), int(height * 0.38)), (int(width * 0.85), int(height * 0.48)), (240, 240, 250), -1)
    cv2.circle(frame, (int(width * 0.35), int(height * 0.43)), max(4, width // 20), (255, 80, 60), -1)
    cv2.circle(frame, (int(width * 0.65), int(height * 0.43)), max(4, width // 20), (255, 220, 60), -1)
    return frame


#6
def _render_handora(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (25, 28, 38)
    cv2.line(
        frame,
        (int(width * 0.22), int(height * 0.78)),
        (int(width * 0.78), int(height * 0.22)),
        (180, 255, 80),
        3,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (int(width * 0.78), int(height * 0.22)), max(6, width // 16), (255, 180, 80), -1, cv2.LINE_AA)
    return frame


_SCENE_RENDERERS: Dict[str, RenderFn] = {
    "gurli-gris": _render_gurli_gris,
    "flying-users-ar": _render_flying_users,
    "webcam-3d": _render_husbygning,
    "webcam-bub": _render_bubbles,
    "webcam-debug": _render_debug,
    "just-dance": _render_just_dance,
    "tilt-of-time": _render_tilt,
    "wiisports": _render_wiisports,
    "handora": _render_handora,
}


def _generate(game_id: str, size: int) -> np.ndarray:
    render = _SCENE_RENDERERS.get(game_id)
    if render is not None:
        try:
            sample = max(size, 192)
            frame = render(sample, sample)
            if frame is not None and frame.size > 0:
                if frame.shape[0] != size or frame.shape[1] != size:
                    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                return frame
        except Exception:
            pass

    base = 128
    img = np.zeros((base, base, 3), dtype=np.uint8)
    img[:] = (52, 56, 74)
    cv2.rectangle(img, (6, 6), (base - 7, base - 7), (120, 130, 165), 2, cv2.LINE_AA)
    cv2.putText(img, "?", (base // 2 - 8, base // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    if size != base:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


def get_game_thumbnail(game_id: str, size: int = DEFAULT_SIZE) -> np.ndarray:
    key = f"{game_id}:{size}"
    if key in _cache:
        return _cache[key]

    if game_id in _SCENE_RENDERERS:
        out = _generate(game_id, size)
        _cache[key] = out
        return out

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
