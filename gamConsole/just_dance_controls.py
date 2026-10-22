"""Luk og UI-fixes til vendor just-dance (uden at redigere vendor/)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from gamConsole.opencv_ui import (
    QUIT_BAR_H,
    draw_quit_bar,
    key_requests_quit,
    quit_bar_click,
    window_was_closed,
)

if TYPE_CHECKING:
    from just_dance_controller import JustDanceController

WIN = "Just Dance"
DISPLAY_W = 1280
DISPLAY_H = 720


def _map_mouse(x: int, y: int, img_w: int, img_h: int) -> tuple[int, int]:
    try:
        _left, _top, win_w, win_h = cv2.getWindowImageRect(WIN)
    except cv2.error:
        return x, y
    if win_w <= 0 or win_h <= 0:
        return x, y
    ix = int(x * img_w / win_w)
    iy = int(y * img_h / win_h)
    return max(0, min(img_w - 1, ix)), max(0, min(img_h - 1, iy))


def _process_frames_with_quit(self: "JustDanceController") -> None:
    counter = 0
    stop = False
    frame_h = DISPLAY_H - QUIT_BAR_H

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal stop
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        _ix, iy = _map_mouse(x, y, DISPLAY_W, DISPLAY_H)
        if quit_bar_click(DISPLAY_H, iy):
            stop = True

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_W, DISPLAY_H)
    cv2.setMouseCallback(WIN, on_mouse)

    while self.cap1.isOpened() and not stop:
        if window_was_closed(WIN):
            break

        start_time = time.time()
        ok1, frame1 = self.cap1.read()
        ok2, frame2 = self.cap2.read()
        if not ok1 or frame1 is None:
            break
        if not ok2 or frame2 is None:
            frame2 = np.zeros_like(frame1)

        frame2 = cv2.flip(frame2, 1)

        if counter == 0:
            key_points_with_scores_video = self.process_frame(frame1)
            key_points_with_scores_camera = self.process_frame(frame2)
            self.model.store_angles(self.angles_video, frame2, key_points_with_scores_video)
            self.model.store_angles(self.angles_camera, frame2, key_points_with_scores_camera)

        counter = (counter + 1) % 100

        height1, _, _ = frame1.shape
        height2, width2, _ = frame2.shape
        if height1 != height2:
            scale_factor = height1 / height2
            width2 = int(width2 * scale_factor)
            frame2 = cv2.resize(frame2, (width2, height1))

        combined_frame = np.concatenate((frame1, frame2), axis=1)
        content_h = max(1, frame_h)
        combined_frame = cv2.resize(combined_frame, (DISPLAY_W, content_h))

        canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
        canvas[:content_h, :] = combined_frame
        draw_quit_bar(canvas)
        cv2.putText(
            canvas,
            "LUK nederst | Q / Esc",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key_requests_quit(key):
            break

        elapsed_time = time.time() - start_time
        rate = self.frame1_rate or 30.0
        frame_delay = max(1, int(1000 / rate) - int(elapsed_time * 1000))
        time.sleep(frame_delay / 1000.0)


def apply_just_dance_quit_patch() -> None:
    import just_dance_controller as jdc

    jdc.JustDanceController.process_frames = _process_frames_with_quit
