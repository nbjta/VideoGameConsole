"""OpenCV-baseret menu — webcam-baggrund med spil-karusel."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gamConsole.child_processes import stop_all_games
from gamConsole.game_sessions import GameSession, create_session
from gamConsole.game_thumbnails import get_game_thumbnail
from gamConsole.games_registry import GameEntry, games_for_menu
from gamConsole.just_dance_session import JustDanceSession
from gamConsole.opencv_ui import (
    HOME_Y2,
    QUIT_BAR_H,
    draw_quit_bar,
    draw_text,
    draw_text_centered,
    fit_frame,
    game_point_from_canvas,
    measure_text,
    home_button_click,
    key_requests_home,
    key_requests_quit,
    overlay_game_chrome,
    quit_bar_click,
    set_content_size,
    window_content_height,
    window_was_closed,
)

WIN = "gamConsole"
CANVAS_W = 960
CANVAS_H = 840
# BGR. Lyseblå kort og mørkeblå tekst, så navnet kan læses.
CARD_ALPHA = 0.84
CARD = (248, 220, 190)
CARD_SIDE = (252, 236, 214)
CARD_SELECTED = (245, 210, 168)
FG = (110, 70, 30)
ACCENT = (220, 150, 60)
CARD_BORDER = (210, 175, 120)
THUMB_FILL = (255, 248, 236)
THUMB_BORDER = (220, 180, 130)
PLAY_BLUE = (235, 170, 70)
PLAY_BLUE_HI = (255, 200, 110)
TOAST_BG = (250, 232, 205)
TITLE_SCALE = 0.52
SELECTED_CARD_W = 268
SELECTED_CARD_H = 300
SIDE_SCALE = 0.68
CARD_PITCH = 210
MAIN_PLAY_H = 80
MAIN_PLAY_ICON_HALF_H = 34


@dataclass
class HitBox:
    x1: int
    y1: int
    x2: int
    y2: int
    game_index: int


@dataclass
class MenuLayout:
    card_boxes: List[HitBox]
    main_play: Tuple[int, int, int, int]


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


def _blend_rect(canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], alpha: float) -> None:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = canvas[y1:y2, x1:x2]
    tint = np.full_like(roi, color)
    cv2.addWeighted(roi, 1.0 - alpha, tint, alpha, 0, dst=roi)


#7 Et negativt x må ikke bruges som numpy-slice. Det tælles fra højre og bliver bredde 0.
def _paste_clipped(canvas: np.ndarray, sprite: np.ndarray, x0: int, y0: int) -> None:
    height, width = canvas.shape[:2]
    sprite_h, sprite_w = sprite.shape[:2]
    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(width, x0 + sprite_w)
    dst_y1 = min(height, y0 + sprite_h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + sprite_w + 2, y0 + sprite_h + 2), THUMB_FILL, -1)
    cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + sprite_w + 2, y0 + sprite_h + 2), THUMB_BORDER, 2, cv2.LINE_AA)
    crop = sprite[src_y0 : src_y0 + (dst_y1 - dst_y0), src_x0 : src_x0 + (dst_x1 - dst_x0)]
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = crop


def _paste_thumb_in_card(
    canvas: np.ndarray,
    thumb: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    reserved_bottom: int,
) -> int:
    pad = 12
    cell_w = x2 - x1
    cell_h = y2 - y1
    side = min(cell_w - 2 * pad, cell_h - 2 * pad - reserved_bottom)
    side = max(48, side)
    small = cv2.resize(thumb, (side, side), interpolation=cv2.INTER_CUBIC)
    x0 = x1 + (cell_w - side) // 2
    y0 = y1 + pad
    _paste_clipped(canvas, small, x0, y0)
    return y0 + side


def _title_fit(title: str, max_w: int) -> str:
    if measure_text(title, TITLE_SCALE)[0] <= max_w:
        return title
    trimmed = title
    while trimmed and measure_text(trimmed + "…", TITLE_SCALE)[0] > max_w:
        trimmed = trimmed[:-1]
    return (trimmed + "…") if trimmed else title[:1]


def _draw_carousel_card_at_offset(
    canvas: np.ndarray,
    game: GameEntry,
    offset: int,
    center_y: int,
    selected: bool,
    card_hover: bool,
    game_index: int,
) -> HitBox:
    if offset == 0:
        cw, ch = SELECTED_CARD_W, SELECTED_CARD_H
    elif abs(offset) == 1:
        cw = int(SELECTED_CARD_W * SIDE_SCALE)
        ch = int(SELECTED_CARD_H * SIDE_SCALE)
    else:
        cw = int(SELECTED_CARD_W * SIDE_SCALE * 0.82)
        ch = int(SELECTED_CARD_H * SIDE_SCALE * 0.82)
    center_x = CANVAS_W // 2 + offset * CARD_PITCH
    x1 = center_x - cw // 2
    y1 = center_y - ch // 2
    x2 = x1 + cw
    y2 = y1 + ch
    fill = CARD_SELECTED if selected else (CARD if card_hover else CARD_SIDE)
    alpha = CARD_ALPHA if selected else 0.74
    _blend_rect(canvas, x1, y1, x2, y2, fill, alpha)
    border = ACCENT if selected else CARD_BORDER
    thickness = 3 if selected else 1
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border, thickness, cv2.LINE_AA)
    reserved = 34 if selected else 8
    thumb = get_game_thumbnail(game.id, size=128)
    thumb_bottom = _paste_thumb_in_card(canvas, thumb, x1, y1, x2, y2, reserved)
    if selected:
        label = _title_fit(game.title, cw - 16)
        draw_text_centered(canvas, label, center_x, thumb_bottom + 6, scale=TITLE_SCALE, color=FG)
    return HitBox(x1, y1, x2, y2, game_index)


def _draw_main_play_button(canvas: np.ndarray, y1: int, hover: bool) -> Tuple[int, int, int, int]:
    cx = CANVAS_W // 2
    cy = y1 + MAIN_PLAY_H // 2
    half_h = MAIN_PLAY_ICON_HALF_H
    inset = 14
    color = PLAY_BLUE_HI if hover else PLAY_BLUE
    tri = np.array(
        [
            [cx - inset, cy - half_h],
            [cx - inset, cy + half_h],
            [cx + half_h + inset, cy],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, tri, color, cv2.LINE_AA)
    pad = 12
    return cx - inset - pad, cy - half_h - pad, cx + half_h + inset + pad, cy + half_h + pad


def _compose_menu(
    webcam: np.ndarray,
    games: List[GameEntry],
    selected_idx: int,
    hover_card: Optional[int],
    hover_main_play: bool,
    camera_index: int,
) -> Tuple[np.ndarray, MenuLayout]:
    canvas = cv2.resize(webcam, (CANVAS_W, CANVAS_H))
    # Kamera-tallet tegnes ikke. Tasterne [ ] skifter stadig kamera.
    del camera_index
    carousel_top = 16
    main_play_y1 = CANVAS_H - QUIT_BAR_H - MAIN_PLAY_H - 20
    center_y = carousel_top + (main_play_y1 - carousel_top) // 2

    card_boxes: List[HitBox] = []
    draw_order = []
    for offset in range(-2, 3):
        idx = selected_idx + offset
        if idx < 0 or idx >= len(games):
            continue
        draw_order.append((abs(offset), offset, idx))
    draw_order.sort(key=lambda t: t[0], reverse=True)
    for _prio, offset, idx in draw_order:
        selected = offset == 0
        box = _draw_carousel_card_at_offset(
            canvas,
            games[idx],
            offset,
            center_y,
            selected=selected,
            card_hover=hover_card == idx,
            game_index=idx,
        )
        card_boxes.append(box)

    main_play = _draw_main_play_button(canvas, main_play_y1, hover_main_play)
    return canvas, MenuLayout(card_boxes=card_boxes, main_play=main_play)


def _key_select_left(key: int) -> bool:
    if key == -1 or key == 255:
        return False
    low = key & 0xFF
    return low in (81, 2, ord("a"), ord("A")) or key in (65361, 2424832)


def _key_select_right(key: int) -> bool:
    if key == -1 or key == 255:
        return False
    low = key & 0xFF
    return low in (83, 3, ord("d"), ord("D")) or key in (65363, 2424834)


def _key_start_game(key: int) -> bool:
    if key == -1 or key == 255:
        return False
    low = key & 0xFF
    return low in (13, 10, 32, ord("p"), ord("P"))


def run_opencv_menu() -> None:
    global CANVAS_H
    #3 Vinduet skal være lavere end den synlige skærm, så LUK-bjælken ikke klippes.
    CANVAS_H = window_content_height(CANVAS_H)
    games = games_for_menu()
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    canvas_w = CANVAS_W
    canvas_h = CANVAS_H

    session: Optional[GameSession] = None
    selected_idx = 0
    hover_card: Optional[int] = None
    hover_main_play = False
    layout = MenuLayout(card_boxes=[], main_play=(0, 0, 0, 0))
    toast = ""
    toast_frames = 0
    request_quit = False
    request_home = False
    game_frame: Optional[Tuple[int, int]] = None

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
        nonlocal session, cap, toast, toast_frames
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

    def _point_in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal hover_card, hover_main_play, selected_idx, request_quit, request_home
        ix, iy = _map_mouse_to_image(x, y, canvas_w, canvas_h)
        if event == cv2.EVENT_MOUSEMOVE and session is None:
            hover_card = None
            for box in layout.card_boxes:
                if box.x1 <= ix <= box.x2 and box.y1 <= iy <= box.y2:
                    hover_card = box.game_index
                    break
            hover_main_play = _point_in_rect(ix, iy, layout.main_play)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if quit_bar_click(canvas_w, ix, iy):
            request_quit = True
            return
        if home_button_click(ix, iy) and session is not None:
            request_home = True
            return
        if session is not None:
            #9 Spil igen ligger i Gurlis billede. Klikket regnes om til det billedes koordinater.
            click = getattr(session, "on_click", None)
            if click is not None and game_frame is not None:
                point = game_point_from_canvas(ix, iy, game_frame[0], game_frame[1], canvas_w, canvas_h)
                if point is not None:
                    click(point[0], point[1])
            return
        if _point_in_rect(ix, iy, layout.main_play):
            _start_game(games[selected_idx])
            return
        for box in layout.card_boxes:
            if box.x1 <= ix <= box.x2 and box.y1 <= iy <= box.y2:
                selected_idx = box.game_index
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
            set_content_size(canvas_w, canvas_h - QUIT_BAR_H - HOME_Y2)
            raw = session.tick()
            if raw is None:
                _go_home()
                continue
            canvas = fit_frame(raw, canvas_w, canvas_h)
            overlay_game_chrome(canvas, session.title)
            draw_quit_bar(canvas)
            game_frame = (raw.shape[1], raw.shape[0])
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(1) & 0xFF
            claims = getattr(session, "claims_key", None)
            #9 Under navneskrivning skal q, h og m ind i navnet og ikke hjem eller luk.
            if claims is not None and claims(key):
                session.on_key(key)
            elif key_requests_quit(key):
                break
            elif key_requests_home(key):
                _go_home()
            elif key not in (255, 0):
                session.on_key(key)
            continue

        if cap.isOpened():
            ok, frame = cap.read()
            bg = frame if ok else np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            bg = np.zeros((480, 640, 3), dtype=np.uint8)

        canvas, layout = _compose_menu(
            bg,
            games,
            selected_idx,
            hover_card,
            hover_main_play,
            camera_index,
        )

        if toast_frames > 0:
            _blend_rect(canvas, 24, CANVAS_H - QUIT_BAR_H - 88, CANVAS_W - 24, CANVAS_H - QUIT_BAR_H - 48, TOAST_BG, 0.88)
            draw_text(canvas, toast[:90], (36, CANVAS_H - QUIT_BAR_H - 72), scale=0.52, color=FG)
            toast_frames -= 1

        draw_quit_bar(canvas)
        cv2.imshow(WIN, canvas)
        raw_key = cv2.waitKey(30)
        if raw_key == -1:
            continue
        key = raw_key & 0xFF
        if key_requests_quit(key):
            break
        if _key_select_left(raw_key):
            selected_idx = (selected_idx - 1) % len(games)
        elif _key_select_right(raw_key):
            selected_idx = (selected_idx + 1) % len(games)
        elif _key_start_game(raw_key):
            _start_game(games[selected_idx])
        elif key in (ord("["), ord("-")):
            camera_index = max(0, camera_index - 1)
            cap.release()
            cap = cv2.VideoCapture(camera_index)
        elif key in (ord("]"), ord("=")):
            camera_index = min(4, camera_index + 1)
            cap.release()
            cap = cv2.VideoCapture(camera_index)

    if session is not None:
        session.stop()
    stop_all_games()
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
