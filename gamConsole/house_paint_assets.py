#8
"""Pensel-grafik til Husbygning (assets/house/brush_red.png)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BRUSH_PATH = ROOT / "assets" / "house" / "brush_red.png"

_brush_bgr: Optional[np.ndarray] = None
_brush_mask: Optional[np.ndarray] = None


def brush_sprite() -> Tuple[np.ndarray, np.ndarray]:
    """BGR + maske (255 = synlig pixel). Hvid baggrund gøres gennemsigtig."""
    global _brush_bgr, _brush_mask
    if _brush_bgr is not None and _brush_mask is not None:
        return _brush_bgr, _brush_mask
    if not BRUSH_PATH.is_file():
        raise FileNotFoundError(f"Mangler pensel-billede: {BRUSH_PATH}")
    img = cv2.imread(str(BRUSH_PATH), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Kunne ikke læse {BRUSH_PATH}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < 248).astype(np.uint8) * 255
    _brush_bgr = img
    _brush_mask = mask
    return _brush_bgr, _brush_mask


def blit_brush(
    frame: np.ndarray,
    center_x: int,
    center_y: int,
    *,
    max_height: int = 56,
    angle_deg: float = -42.0,
) -> None:
    """Tegn penslen centreret om (center_x, center_y), let roteret."""
    src, mask = brush_sprite()
    h0, w0 = src.shape[:2]
    scale = max_height / max(h0, 1)
    nw, nh = max(8, int(w0 * scale)), max(8, int(h0 * scale))
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
    rot = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(resized, rot, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mrot = cv2.warpAffine(m, rot, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    ox = center_x - nw // 2
    oy = center_y - nh // 2
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, ox), max(0, oy)
    x2, y2 = min(fw, ox + nw), min(fh, oy + nh)
    if x2 <= x1 or y2 <= y1:
        return
    sx1, sy1 = x1 - ox, y1 - oy
    patch = rotated[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)]
    alpha = (mrot[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)].astype(np.float32) / 255.0)[:, :, None]
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = (patch.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(
        np.uint8
    )
