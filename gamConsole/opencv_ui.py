"""Fælles OpenCV UI-hjælpere (luk på macOS)."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import cv2

QUIT_BAR_H = 44
HOME_X1, HOME_Y1, HOME_X2, HOME_Y2 = 8, 8, 132, 44


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
    cv2.rectangle(frame, (HOME_X1, HOME_Y1), (HOME_X2, HOME_Y2), (60, 90, 140), -1)
    cv2.rectangle(frame, (HOME_X1, HOME_Y1), (HOME_X2, HOME_Y2), (120, 160, 220), 2)
    cv2.putText(
        frame,
        "Hjem",
        (HOME_X1 + 18, HOME_Y2 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def home_button_click(x: int, y: int) -> bool:
    return HOME_X1 <= x <= HOME_X2 and HOME_Y1 <= y <= HOME_Y2


def draw_quit_bar(frame) -> None:
    h, w = frame.shape[:2]
    y1 = h - QUIT_BAR_H
    cv2.rectangle(frame, (0, y1), (w, h), (50, 45, 70), -1)
    cv2.rectangle(frame, (8, y1 + 6), (w - 8, h - 6), (120, 70, 70), 2)
    cv2.putText(
        frame,
        "LUK konsol  —  klik her  eller  Q / Esc",
        (20, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 230, 230),
        1,
        cv2.LINE_AA,
    )


def quit_bar_click(frame_height: int, y: int) -> bool:
    return y >= frame_height - QUIT_BAR_H


def overlay_game_chrome(frame, title: str) -> None:
    draw_home_button(frame)
    cv2.putText(
        frame,
        title,
        (HOME_X2 + 12, HOME_Y2 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )


def fit_frame(frame, width: int, height: int):
    import numpy as np

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    fh, fw = frame.shape[:2]
    if fh == 0 or fw == 0:
        return canvas
    scale = min(width / fw, (height - QUIT_BAR_H) / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(frame, (nw, nh))
    x0 = (width - nw) // 2
    y0 = (height - QUIT_BAR_H - nh) // 2 + HOME_Y2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def run_window_loop(
    window_name: str,
    cap: cv2.VideoCapture,
    render: Callable[[], Optional[object]],
    on_key: Optional[Callable[[int], None]] = None,
) -> None:
    """Legacy fuldskærms-løkke (CLI). Foretræk gamConsole-menu med indlejrede spil."""
    stop = False
    last_h = 480

    def on_mouse(event: int, _x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal stop
        if event == cv2.EVENT_LBUTTONDOWN and quit_bar_click(last_h, y):
            stop = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while not stop:
        if window_was_closed(window_name):
            break
        frame = render()
        if frame is None:
            break
        last_h = frame.shape[0]
        draw_quit_bar(frame)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key_requests_quit(key):
            break
        if on_key is not None and key not in (255, 0):
            on_key(key)

    cap.release()
    cv2.destroyAllWindows()
