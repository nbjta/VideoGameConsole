"""Fælles OpenCV UI-hjælpere (luk på macOS)."""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

import cv2

#3 Krydset ligger oven på billedet, så der ikke er en bjælke under spillet.
QUIT_BAR_H = 0
HOME_X1, HOME_Y1, HOME_X2, HOME_Y2 = 8, 8, 132, 44
_CLOSE_SIZE = 46
_CLOSE_MARGIN = 14
_content_wh: Tuple[int, int] = (960, 752)


def window_was_closed(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def key_requests_quit(key: int) -> bool:
    if key == 255 or key == -1:
        return False
    return key in (ord("q"), ord("Q"), 27)


def key_requests_home(key: int) -> bool:
    return key in (ord("h"), ord("H"), ord("m"), ord("M"))


def draw_home_button(frame) -> None:
    # BGR, så knappen tegnes blå. Tuplen (60, 90, 140) blev vist som brun.
    cv2.rectangle(frame, (HOME_X1, HOME_Y1), (HOME_X2, HOME_Y2), (245, 205, 150), -1)
    cv2.rectangle(frame, (HOME_X1, HOME_Y1), (HOME_X2, HOME_Y2), (220, 160, 90), 2)
    draw_text(frame, "Hjem", (HOME_X1 + 16, HOME_Y1 + 6), scale=0.62, color=(110, 70, 30))


def home_button_click(x: int, y: int) -> bool:
    return HOME_X1 <= x <= HOME_X2 and HOME_Y1 <= y <= HOME_Y2


def close_button_rect(width: int) -> Tuple[int, int, int, int]:
    """#3 Krydset sidder i øverste højre hjørne."""
    x1 = width - _CLOSE_MARGIN - _CLOSE_SIZE
    y1 = _CLOSE_MARGIN
    return x1, y1, x1 + _CLOSE_SIZE, y1 + _CLOSE_SIZE


def draw_quit_bar(frame) -> None:
    """#3 X lukker konsollen."""
    _h, width = frame.shape[:2]
    x1, y1, x2, y2 = close_button_rect(width)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 36, 170), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    pad = 12
    cv2.line(frame, (x1 + pad, y1 + pad), (x2 - pad, y2 - pad), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, (x1 + pad, y2 - pad), (x2 - pad, y1 + pad), (255, 255, 255), 3, cv2.LINE_AA)


def quit_bar_click(frame_width: int, x: int, y: int) -> bool:
    """#3 Kun selve krydset lukker."""
    x1, y1, x2, y2 = close_button_rect(frame_width)
    return x1 <= x <= x2 and y1 <= y <= y2


#4 Hershey i cv2.putText har ikke ø/æ, så UTF-8 bliver til ??.
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)
_font_cache: Dict[int, object] = {}


def _load_font(px: int):
    cached = _font_cache.get(px)
    if cached is not None:
        return cached
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if not os.path.isfile(path):
            continue
        try:
            font = ImageFont.truetype(path, px)
        except OSError:
            continue
        _font_cache[px] = font
        return font
    font = ImageFont.load_default()
    _font_cache[px] = font
    return font


def _text_bbox(text: str, scale: float) -> Tuple[int, int]:
    from PIL import Image, ImageDraw

    font = _load_font(max(12, int(round(32 * scale))))
    probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    bbox = probe.textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0] + 2)
    height = max(1, bbox[3] - bbox[1] + 2)
    return width, height


def measure_text(text: str, scale: float = 0.7) -> Tuple[int, int]:
    if not text:
        return 0, 0
    return _text_bbox(text, scale)


def draw_text_centered(
    frame,
    text: str,
    center_x: int,
    y: int,
    scale: float = 0.7,
    color: Tuple[int, int, int] = (234, 234, 234),
) -> None:
    width, _ = measure_text(text, scale)
    draw_text(frame, text, (center_x - width // 2, y), scale=scale, color=color)


def draw_text(
    frame,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.7,
    color: Tuple[int, int, int] = (234, 234, 234),
) -> None:
    """Tegn Unicode-tekst. origin er øverste venstre hjørne, color er BGR."""
    if not text:
        return
    import numpy as np
    from PIL import Image, ImageDraw

    font = _load_font(max(12, int(round(32 * scale))))
    rgb = (int(color[2]), int(color[1]), int(color[0]), 255)
    probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    bbox = probe.textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0] + 2)
    height = max(1, bbox[3] - bbox[1] + 2)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((-bbox[0], -bbox[1]), text, font=font, fill=rgb)
    sprite = np.asarray(image)
    frame_h, frame_w = frame.shape[:2]
    x, y = origin
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_w, x + sprite.shape[1]), min(frame_h, y + sprite.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    crop = sprite[y1 - y : y1 - y + (y2 - y1), x1 - x : x1 - x + (x2 - x1)]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    bgr = crop[:, :, 2::-1].astype(np.float32)
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)


def overlay_game_chrome(frame, title: str) -> None:
    draw_home_button(frame)
    draw_text(frame, title, (HOME_X2 + 10, HOME_Y1 + 6), scale=0.62, color=(240, 240, 240))


def set_content_size(width: int, height: int) -> None:
    global _content_wh
    _content_wh = (max(1, width), max(1, height))


def content_size() -> Tuple[int, int]:
    return _content_wh


def window_content_height(preferred: int, visible_height: Optional[int] = None) -> int:
    """Højde som kan stå på skærmen sammen med vinduets titelbjælke.

    Målt på denne Mac: synlig højde 796 pt, mens menuen bad om 840.
    De nederste pixels, inklusive LUK-teksten, lå derfor under Dock.
    """
    if visible_height is None:
        visible_height = _macos_visible_height()
    if not visible_height:
        return preferred
    return max(640, min(preferred, int(visible_height) - 36))


def _macos_visible_height() -> Optional[int]:
    try:
        from AppKit import NSScreen

        screen = NSScreen.mainScreen()
        if screen is None:
            return None
        return int(screen.visibleFrame().size.height)
    except Exception:
        return None


#9 Samme placering som fit_frame, så et klik på Spil igen rammer knappen i spillets billede.
def fitted_frame_rect(frame_w: int, frame_h: int, canvas_w: int, canvas_h: int) -> Tuple[int, int, int, int]:
    top = HOME_Y2
    bottom = max(top + 1, canvas_h - QUIT_BAR_H)
    avail_h = bottom - top
    if frame_w <= 0 or frame_h <= 0 or canvas_w <= 0 or avail_h <= 0:
        return 0, top, 0, 0
    scale = min(canvas_w / frame_w, avail_h / frame_h)
    nw = max(1, min(canvas_w, int(frame_w * scale)))
    nh = max(1, min(avail_h, int(frame_h * scale)))
    x0 = (canvas_w - nw) // 2
    y0 = top + (avail_h - nh) // 2
    return x0, y0, nw, nh


def game_point_from_canvas(
    ix: int, iy: int, frame_w: int, frame_h: int, canvas_w: int, canvas_h: int
) -> Optional[Tuple[int, int]]:
    x0, y0, nw, nh = fitted_frame_rect(frame_w, frame_h, canvas_w, canvas_h)
    if nw <= 0 or nh <= 0 or frame_w <= 0 or frame_h <= 0:
        return None
    if not (x0 <= ix < x0 + nw and y0 <= iy < y0 + nh):
        return None
    return (
        min(frame_w - 1, int((ix - x0) * frame_w / nw)),
        min(frame_h - 1, int((iy - y0) * frame_h / nh)),
    )


def fit_frame(frame, width: int, height: int):
    import numpy as np

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    fh, fw = frame.shape[:2]
    if fh == 0 or fw == 0:
        return canvas
    x0, y0, nw, nh = fitted_frame_rect(fw, fh, width, height)
    resized = cv2.resize(frame, (nw, nh))
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def run_window_loop(
    window_name: str,
    cap: cv2.VideoCapture,
    render: Callable[[], Optional[object]],
    on_key: Optional[Callable[[int], None]] = None,
    on_click: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Legacy fuldskærms-løkke (CLI). Foretræk gamConsole-menu med indlejrede spil."""
    stop = False
    last_w = 640
    last_h = 480

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal stop
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ix, iy = x, y
        try:
            _left, _top, win_w, win_h = cv2.getWindowImageRect(window_name)
            if win_w > 0 and win_h > 0:
                ix = int(x * last_w / win_w)
                iy = int(y * last_h / win_h)
        except cv2.error:
            pass
        if quit_bar_click(last_w, ix, iy):
            stop = True
            return
        #9 Klik i spillets eget billede, blandt andet Spil igen.
        if on_click is not None:
            on_click(ix, iy)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while not stop:
        if window_was_closed(window_name):
            break
        frame = render()
        if frame is None:
            break
        last_h, last_w = frame.shape[:2]
        draw_quit_bar(frame)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key_requests_quit(key):
            break
        if on_key is not None and key not in (255, 0):
            on_key(key)

    cap.release()
    cv2.destroyAllWindows()
