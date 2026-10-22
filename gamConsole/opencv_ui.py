"""Fælles OpenCV UI-hjælpere (luk på macOS)."""

from __future__ import annotations

from typing import Callable, Optional

import cv2

QUIT_BAR_H = 44


def window_was_closed(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def key_requests_quit(key: int) -> bool:
    if key == 255 or key == -1:
        return False
    return key in (ord("q"), ord("Q"), 27)


def draw_quit_bar(frame) -> None:
    h, w = frame.shape[:2]
    y1 = h - QUIT_BAR_H
    cv2.rectangle(frame, (0, y1), (w, h), (50, 45, 70), -1)
    cv2.rectangle(frame, (8, y1 + 6), (w - 8, h - 6), (120, 70, 70), 2)
    cv2.putText(
        frame,
        "LUK  —  klik her  eller  Q / Esc",
        (20, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 230, 230),
        1,
        cv2.LINE_AA,
    )


def quit_bar_click(frame_height: int, y: int) -> bool:
    return y >= frame_height - QUIT_BAR_H


def run_window_loop(
    window_name: str,
    cap: cv2.VideoCapture,
    render: Callable[[], Optional[object]],
) -> None:
    """Vis render() hvert frame indtil luk (klik, Q, Esc eller rødt X)."""
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

    cap.release()
    cv2.destroyAllWindows()
