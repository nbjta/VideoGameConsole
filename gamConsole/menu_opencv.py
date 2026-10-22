"""OpenCV-baseret menu — ét vindue med indlejrede spil og Hjem-knap."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gamConsole.child_processes import stop_all_games
from gamConsole.cvgames_launcher import cvgames_binary, is_webcam_games_installed
from gamConsole.game_sessions import GameSession, create_session
from gamConsole.games_registry import GameEntry, games_for_menu
from gamConsole.just_dance_session import JustDanceSession
from gamConsole.opencv_ui import (
    QUIT_BAR_H,
    draw_quit_bar,
    fit_frame,
    home_button_click,
    key_requests_home,
    key_requests_quit,
    overlay_game_chrome,
    quit_bar_click,
    window_was_closed,
)

WIN = "gamConsole"
PANEL_W = 400
CAM_W = 560
WIN_H = 840
BG = (26, 26, 46)
CARD = (22, 33, 62)
CARD_HOVER = (32, 48, 88)
FG = (234, 234, 234)
MUTED = (160, 160, 184)
ACCENT = (80, 140, 255)


@dataclass
class HitBox:
    x1: int
    y1: int
    x2: int
    y2: int
    game: GameEntry


def _map_mouse_to_image(x: int, y: int, img_w: int, img_h: int) -> Tuple[int, int]:
    try:
        _left, _top, win_w, win_h = cv2.getWindowImageRect(WIN)
    except cv2.error:
        return x, y
    if win_w <= 0 or win_h <= 0:
        return x, y
    ix = int(x * img_w / win_w)
    iy = int(y * img_h / win_h)
    return max(0, min(img_w - 1, ix)), max(0, min(img_h - 1, iy))


def _draw_button(img: np.ndarray, box: HitBox, hover: bool) -> None:
    bx1, by1, bx2, by2 = PANEL_W - 96, box.y1 + 52, PANEL_W - 24, box.y1 + 80
    color = ACCENT if hover else (60, 70, 110)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), color, -1)
    cv2.putText(img, "Spil", (bx1 + 12, by2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _status_line() -> str:
    if cvgames_binary():
        return "CvGames: klar (spil kører i samme vindue som Python)"
    if is_webcam_games_installed():
        return "Spil: samme vindue + Hjem-knap"
    return "Spil: samme vindue + Hjem-knap"


def _build_panel(
    games: List[GameEntry], hover_idx: Optional[int], active_idx: Optional[int]
) -> Tuple[np.ndarray, List[HitBox]]:
    panel = np.zeros((WIN_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = BG
    cv2.putText(panel, "gamConsole", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, FG, 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        "Klik spil (1-9). I spil: Hjem eller H.",
        (16, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    boxes: List[HitBox] = []
    y = 90
    for i, game in enumerate(games):
        card_color = CARD_HOVER if i in (hover_idx, active_idx) else CARD
        cv2.rectangle(panel, (12, y), (PANEL_W - 12, y + 88), card_color, -1)
        cv2.putText(panel, f"{i + 1}. {game.title}", (24, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, FG, 1, cv2.LINE_AA)
        desc = game.description[:52] + ("…" if len(game.description) > 52 else "")
        cv2.putText(panel, desc, (24, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, MUTED, 1, cv2.LINE_AA)
        box = HitBox(12, y, PANEL_W - 12, y + 88, game)
        boxes.append(box)
        _draw_button(panel, box, hover=(hover_idx == i))
        y += 96
    cv2.putText(panel, _status_line(), (16, WIN_H - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)
    cv2.putText(panel, "Kamera: [ / ]", (16, WIN_H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)
    return panel, boxes


def run_opencv_menu() -> None:
    games = games_for_menu()
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    canvas_w = PANEL_W + CAM_W
    canvas_h = WIN_H

    session: Optional[GameSession] = None
    hover_idx: Optional[int] = None
    active_idx: Optional[int] = None
    active_frames = 0
    boxes: List[HitBox] = []
    toast = ""
    toast_frames = 0
    request_quit = False
    request_home = False

    def _go_home() -> None:
        nonlocal session, cap, toast, toast_frames
        if session is None:
            return
        if isinstance(session, JustDanceSession):
            score = session.finalize_score()
            if score is not None:
                toast = f"Just Dance score: {score}"
                toast_frames = 120
        session.stop()
        session = None
        stop_all_games()
        cap = cv2.VideoCapture(camera_index)

    def _start_game(game: GameEntry) -> None:
        nonlocal session, cap, toast, toast_frames, active_idx, active_frames
        if cap.isOpened():
            cap.release()
        toast = f"Starter {game.title}…"
        toast_frames = 90
        session, msg = create_session(game, camera_index)
        toast = msg[:120]
        toast_frames = 150
        if session is None:
            print(f"[gamConsole] {msg}", file=sys.stderr)
            cap = cv2.VideoCapture(camera_index)
            return
        for i, g in enumerate(games):
            if g.id == game.id:
                active_idx = i
                active_frames = 15
                break

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal hover_idx, request_quit, request_home
        ix, iy = _map_mouse_to_image(x, y, canvas_w, canvas_h)
        if event == cv2.EVENT_MOUSEMOVE and session is None and ix < PANEL_W:
            hover_idx = None
            for i, box in enumerate(boxes):
                if box.x1 <= ix <= box.x2 and box.y1 <= iy <= box.y2:
                    hover_idx = i
                    break
            else:
                hover_idx = None
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if quit_bar_click(canvas_h, iy):
            request_quit = True
            return
        if home_button_click(ix, iy) and session is not None:
            request_home = True
            return
        if session is not None:
            return
        if ix >= PANEL_W:
            hover_idx = None
            return
        for box in boxes:
            if box.x1 <= ix <= box.x2 and box.y1 <= iy <= box.y2:
                _start_game(box.game)
                break

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, canvas_w, canvas_h)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        if request_quit or window_was_closed(WIN):
            break
        if request_home:
            request_home = False
            _go_home()

        if session is not None:
            raw = session.tick()
            if raw is None:
                _go_home()
                continue
            canvas = fit_frame(raw, canvas_w, canvas_h)
            overlay_game_chrome(canvas, session.title)
            draw_quit_bar(canvas)
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key_requests_quit(key):
                break
            if key_requests_home(key):
                _go_home()
            elif key not in (255, 0):
                session.on_key(key)
            continue

        if active_frames > 0:
            active_frames -= 1
        else:
            active_idx = None

        panel, boxes = _build_panel(games, hover_idx, active_idx)
        if cap.isOpened():
            ok, frame = cap.read()
            frame = cv2.resize(frame, (CAM_W, WIN_H)) if ok else np.zeros((WIN_H, CAM_W, 3), dtype=np.uint8)
        else:
            frame = np.zeros((WIN_H, CAM_W, 3), dtype=np.uint8)
        cv2.putText(frame, f"Webcam #{camera_index}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        canvas = np.hstack([panel, frame])
        if toast_frames > 0:
            cv2.rectangle(canvas, (8, WIN_H - QUIT_BAR_H - 40), (PANEL_W - 8, WIN_H - QUIT_BAR_H - 4), (40, 40, 60), -1)
            cv2.putText(canvas, toast[:100], (16, WIN_H - QUIT_BAR_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
            toast_frames -= 1
        draw_quit_bar(canvas)
        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key_requests_quit(key):
            break
        if key in (ord("["), ord("-")):
            camera_index = max(0, camera_index - 1)
            cap.release()
            cap = cv2.VideoCapture(camera_index)
        if key in (ord("]"), ord("=")):
            camera_index = min(4, camera_index + 1)
            cap.release()
            cap = cv2.VideoCapture(camera_index)
        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(games):
                _start_game(games[idx])

    if session is not None:
        session.stop()
    stop_all_games()
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
