#8
"""Husbygning: flyt dele på pladen til spøgelses-markeringen.

Stolper, modulvægge, toprem, døre, vinduer og tag — én opgave ad gangen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

from gamConsole.house_paint_assets import blit_brush
from gamConsole.house_tool_cards import ToolCardBar, paint_color_for
from gamConsole.opencv_ui import draw_text

if TYPE_CHECKING:
    from gamConsole.hand_pip_overlay import HandTrack

Point = Tuple[int, int]

SNAP_TRIGGER_ERR = 0.075
SNAP_DONE_ERR = 0.012
SNAP_LERP = 0.45

WOOD = (58, 108, 154)
WOOD_HI = (96, 156, 214)
WOOD_EDGE = (32, 48, 72)
GHOST = (120, 220, 255)


@dataclass(frozen=True)
class BuildStep:
    kind: Literal["post", "wall", "rim", "door", "window", "roof"]
    target_u: float
    target_v: float
    label: str
    rim_horizontal: bool = True


#8 Stolper, modulvægge, toprem, døre, vinduer og til sidst taget.
HOUSE_PLAN: List[BuildStep] = [
    BuildStep("post", 0.12, 0.14, "Hjørnestolpe 1 (bag venstre)"),
    BuildStep("post", 0.88, 0.14, "Hjørnestolpe 2 (bag højre)"),
    BuildStep("post", 0.88, 0.86, "Hjørnestolpe 3 (for højre)"),
    BuildStep("post", 0.12, 0.86, "Hjørnestolpe 4 (for venstre)"),
    BuildStep("wall", 0.5, 0.14, "Modulvæg bag", rim_horizontal=True),
    BuildStep("wall", 0.5, 0.86, "Modulvæg for", rim_horizontal=True),
    BuildStep("wall", 0.12, 0.5, "Modulvæg venstre", rim_horizontal=False),
    BuildStep("wall", 0.88, 0.5, "Modulvæg højre", rim_horizontal=False),
    BuildStep("rim", 0.5, 0.14, "Toprem bag", rim_horizontal=True),
    BuildStep("rim", 0.5, 0.86, "Toprem for", rim_horizontal=True),
    BuildStep("rim", 0.12, 0.5, "Toprem venstre", rim_horizontal=False),
    BuildStep("rim", 0.88, 0.5, "Toprem højre", rim_horizontal=False),
    BuildStep("door", 0.50, 0.86, "Fordør"),
    BuildStep("door", 0.50, 0.14, "Bagdør"),
    BuildStep("window", 0.30, 0.86, "Vindue for venstre"),
    BuildStep("window", 0.70, 0.86, "Vindue for højre"),
    BuildStep("window", 0.12, 0.48, "Vindue venstre side"),
    BuildStep("window", 0.88, 0.48, "Vindue højre side"),
    BuildStep("roof", 0.50, 0.50, "Tag"),
]


class SceneLayer:
    def paint(self, frame: np.ndarray) -> None:
        raise NotImplementedError


class ConcreteFoundation(SceneLayer):
    """Jord, forskalling og en grå betonplade set lidt skråt ovenfra."""

    def __init__(self) -> None:
        self._noise: Dict[Tuple[int, int], np.ndarray] = {}

    def paint(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = frame.shape[:2]
        #8 Blå himmel over pladen, græs på resten — fundamentet males ovenpå.
        _paint_yard(frame, int(height * 0.20))
        top, front = _foundation_quads(width, height)
        form = _outset(top, max(10, width // 48))
        cv2.fillConvexPoly(frame, form, (52, 78, 112))
        cv2.fillConvexPoly(frame, front, (118, 124, 128))
        cv2.fillConvexPoly(frame, top, (176, 178, 174))
        self._speckle(frame, top)
        self._joints(frame, top)
        self._bolts(frame, top)
        return top, front

    def _speckle(self, frame: np.ndarray, top: np.ndarray) -> None:
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, top, 255)
        noise = self._noise_for(height, width)
        painted = frame.astype(np.int16)
        painted += noise
        np.clip(painted, 0, 255, out=painted)
        use = mask[:, :, None] > 0
        frame[:] = np.where(use, painted.astype(np.uint8), frame)

    def _noise_for(self, height: int, width: int) -> np.ndarray:
        key = (height, width)
        cached = self._noise.get(key)
        if cached is not None:
            return cached
        rng = np.random.default_rng(8)
        noise = rng.integers(-16, 17, size=(height, width, 1), dtype=np.int16)
        self._noise[key] = noise
        return noise

    def _joints(self, frame: np.ndarray, top: np.ndarray) -> None:
        far_left, far_right, near_right, near_left = [tuple(p) for p in top]
        horizontal = _segment(far_left, far_right, near_left, near_right, 0.5)
        vertical = _segment(far_left, near_left, far_right, near_right, 0.5)
        cv2.line(frame, horizontal[0], horizontal[1], (150, 152, 148), 2, cv2.LINE_AA)
        cv2.line(frame, vertical[0], vertical[1], (150, 152, 148), 2, cv2.LINE_AA)

    def _bolts(self, frame: np.ndarray, top: np.ndarray) -> None:
        radius = max(4, frame.shape[1] // 90)
        for point in _inset_corners(top, 0.14):
            cv2.circle(frame, point, radius + 2, (70, 74, 78), -1, cv2.LINE_AA)
            cv2.circle(frame, point, radius, (196, 198, 200), -1, cv2.LINE_AA)
            cv2.circle(frame, point, max(1, radius // 3), (40, 40, 40), -1, cv2.LINE_AA)


@dataclass
class PlacedPiece:
    step: BuildStep


@dataclass
class PaintDot:
    u: float
    v: float
    radius: float
    color: Tuple[int, int, int]


class HouseBuildScene:
    """Progressiv bjælkebygning på pladen med spøgelses-markering og snap."""

    def __init__(self) -> None:
        self.foundation = ConcreteFoundation()
        self.steps = list(HOUSE_PLAN)
        self.step_index = 0
        self.placed: List[PlacedPiece] = []
        self.beam_u = 0.5
        self.beam_v = 0.55
        self.snap_anim = 0
        self.reward_frames = 0
        self.complete = False
        self._slab_top: Optional[np.ndarray] = None
        self._grip_px: Optional[Point] = None
        self._grip_landmarks: Optional[Sequence[Tuple[float, float]]] = None
        self._grip_holding = False
        #8 Værktøjskort + maling på materialer der allerede ligger på pladen.
        self.tool_bar = ToolCardBar()
        self.paint_dots: List[PaintDot] = []
        self._load_step(start_offset=True)

    # Bevar API til ældre tests der læser vinkler — mappes til placering.
    @property
    def angle_x(self) -> float:
        return (self.beam_v - 0.5) * 3.0

    @angle_x.setter
    def angle_x(self, value: float) -> None:
        self.beam_v = value / 3.0 + 0.5

    @property
    def angle_y(self) -> float:
        return (self.beam_u - 0.5) * 3.0

    @angle_y.setter
    def angle_y(self, value: float) -> None:
        self.beam_u = value / 3.0 + 0.5

    def _current(self) -> Optional[BuildStep]:
        if self.step_index >= len(self.steps):
            return None
        return self.steps[self.step_index]

    def _load_step(self, *, start_offset: bool) -> None:
        step = self._current()
        if step is None:
            self.complete = True
            return
        if start_offset:
            self.beam_u = 0.5
            self.beam_v = 0.55

    def select_tool(self, tool_id: str) -> None:
        self.tool_bar.selected_id = tool_id

    def brush_active(self) -> bool:
        return self.tool_bar.selected_id.startswith("brush_")

    def paint_at_pixel(self, px: int, py: int) -> None:
        if self._slab_top is None or not self.placed:
            return
        color = paint_color_for(self.tool_bar.selected_id)
        if color is None:
            return
        uv = _pixel_to_slab_uv(self._slab_top, px, py)
        if uv is None or not _uv_on_placed_material(uv, self.placed):
            return
        u, v = uv
        if self.paint_dots:
            last = self.paint_dots[-1]
            if (
                abs(last.u - u) < 0.008
                and abs(last.v - v) < 0.008
                and last.color == color
            ):
                return
        self.paint_dots.append(PaintDot(u, v, 0.018, color))

    def alignment_error(self) -> float:
        step = self._current()
        if step is None:
            return 0.0
        return math.hypot(self.beam_u - step.target_u, self.beam_v - step.target_v)

    def steer(
        self,
        frame_shape: Sequence[int],
        centroid: Optional[Tuple[int, int]],
        *,
        holding: bool = True,
        landmarks: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> None:
        self._update_snap()
        if self.complete or self.snap_anim > 0 or self.reward_frames > 0:
            return
        #8 Hånden følger palm i billedet. Kun klem flytter delen på pladen.
        if centroid is None:
            self._grip_px = None
            self._grip_landmarks = None
            self._grip_holding = False
            return
        self._grip_px = (int(centroid[0]), int(centroid[1]))
        self._grip_landmarks = landmarks
        self._grip_holding = holding
        if not holding:
            return
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if width <= 0 or height <= 0:
            return
        self.beam_u = max(0.06, min(0.94, centroid[0] / width))
        self.beam_v = max(0.06, min(0.94, centroid[1] / height))

    def _update_snap(self) -> None:
        if self.reward_frames > 0:
            self.reward_frames -= 1
            return
        step = self._current()
        if step is None:
            return
        if self.snap_anim > 0:
            self.beam_u += (step.target_u - self.beam_u) * SNAP_LERP
            self.beam_v += (step.target_v - self.beam_v) * SNAP_LERP
            self.snap_anim -= 1
            if self.snap_anim == 0 or self.alignment_error() < SNAP_DONE_ERR:
                self.beam_u = step.target_u
                self.beam_v = step.target_v
                self._complete_step()
            return
        if self.alignment_error() < SNAP_TRIGGER_ERR:
            self.snap_anim = 10

    def _complete_step(self) -> None:
        step = self._current()
        if step is None:
            return
        self.placed.append(PlacedPiece(step=step))
        self.step_index += 1
        self.reward_frames = 50
        self.snap_anim = 0
        self._load_step(start_offset=True)

    def draw(self, frame: np.ndarray) -> None:
        top, _front = self.foundation.paint(frame)
        self._slab_top = top
        for piece in self.placed:
            _paint_piece(frame, top, piece.step, piece.step.target_u, piece.step.target_v, ghost=False)
        self._draw_paint(top, frame)
        step = self._current()
        if step is not None:
            _paint_piece(frame, top, step, step.target_u, step.target_v, ghost=True)
            _paint_piece(frame, top, step, self.beam_u, self.beam_v, ghost=False)
            self._draw_target_hint(frame, step)
        self.tool_bar.draw(frame)
        if self._grip_landmarks is not None:
            h, w = frame.shape[:2]
            grab_anchor: Optional[Point] = None
            if (
                step is not None
                and self._grip_holding
                and not self.brush_active()
                and self.snap_anim == 0
                and self.reward_frames == 0
            ):
                grab_anchor = _grasp_anchor(top, step, self.beam_u, self.beam_v)
            if self.brush_active():
                _draw_brush_at_hand(frame, self._grip_landmarks, w, h)
            else:
                _draw_screen_hand(frame, self._grip_landmarks, w, h, grab_anchor=grab_anchor)
        self._draw_hud(frame)

    def _draw_paint(self, top: np.ndarray, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        for dot in self.paint_dots:
            cx, cy = _slab_point(top, dot.u, dot.v)
            r = max(3, int(dot.radius * w))
            cv2.circle(frame, (cx, cy), r, dot.color, -1, cv2.LINE_AA)

    def _draw_target_hint(self, frame: np.ndarray, step: BuildStep) -> None:
        h, w = frame.shape[:2]
        err = self.alignment_error()
        bar_w = int(w * 0.34)
        x0 = (w - bar_w) // 2
        y0 = h - 36
        fill = int((1.0 - min(1.0, err / 0.35)) * bar_w)
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 10), (40, 50, 70), -1)
        color = (80, 220, 120) if err < SNAP_TRIGGER_ERR else (80, 180, 255)
        cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + 10), color, -1)
        draw_text(frame, f"Mål: {step.label}", (x0, y0 - 22), scale=0.48, color=(220, 235, 255))
        if self._slab_top is not None:
            tgt = _slab_point(self._slab_top, step.target_u, step.target_v)
            cur = _slab_point(self._slab_top, self.beam_u, self.beam_v)
            cv2.circle(frame, tgt, 10, GHOST, 2, cv2.LINE_AA)
            cv2.line(frame, cur, tgt, (100, 200, 255), 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self.complete:
            draw_text(frame, "Hus færdigt!", (w // 2 - 90, h // 2 - 20), scale=0.85, color=(120, 255, 160))
            draw_text(
                frame,
                "Vægge, toprem, døre, vinduer og tag",
                (w // 2 - 140, h // 2 + 22),
                scale=0.52,
                color=(230, 240, 255),
            )
            return
        step = self._current()
        total = len(self.steps)
        tool_hint = (
            "Klem på pensel-kort, tegn på materialer på pladen"
            if self.brush_active()
            else "Vælg værktøj øverst — klem for at tage fat om del"
        )
        draw_text(
            frame,
            f"Opgave {self.step_index + 1} / {total} — {tool_hint}",
            (16, ToolCardBar.BAR_H + 6),
            scale=0.44,
            color=(236, 236, 236),
        )
        if self.reward_frames > 0:
            draw_text(frame, "På plads! Næste opgave …", (w // 2 - 120, int(h * 0.14)), scale=0.62, color=(100, 255, 150))
        elif self.snap_anim > 0:
            draw_text(frame, "Snap …", (w // 2 - 40, int(h * 0.14)), scale=0.55, color=(180, 240, 255))


#8
def _grasp_anchor(top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
    """Punkt på delen hvor hånden visuelt tager fat."""
    return _part(step.kind).anchor(top, step, u, v)


#8 Samme 21 punkter som MediaPipe-hånden i nederste højre hjørne. Fingre opad.
_POINT_HAND: Tuple[Point, ...] = (
    (50, 98),
    (32, 80),
    (20, 64),
    (14, 50),
    (10, 36),
    (36, 52),
    (34, 34),
    (33, 20),
    (32, 6),
    (50, 48),
    (50, 28),
    (50, 14),
    (50, 0),
    (64, 52),
    (66, 34),
    (67, 20),
    (68, 8),
    (76, 60),
    (80, 44),
    (82, 32),
    (84, 20),
)
_POINT_HAND_LINKS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
#8 Samme farver som skelettet i hand_pip_overlay.
_HAND_LINE = (255, 255, 0)
_HAND_JOINT = (255, 255, 255)


#8
def _landmarks_on_frame(landmarks: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    """MediaPipe kører på spejlvendt camera; scenen tegnes på u-spejlet frame."""
    return tuple((1.0 - float(x), float(y)) for x, y in landmarks)


def _palm_frame_px(landmarks: Sequence[Tuple[float, float]], width: int, height: int) -> Point:
    ids = (0, 5, 9, 13, 17)
    xs = [landmarks[i][0] for i in ids]
    ys = [landmarks[i][1] for i in ids]
    return int(sum(xs) / len(xs) * width), int(sum(ys) / len(ys) * height)


#8
def _draw_brush_at_hand(
    frame: np.ndarray,
    landmarks: Sequence[Tuple[float, float]],
    width: int,
    height: int,
) -> None:
    if len(landmarks) < 9:
        return
    wrist = _landmark_to_pixel(landmarks[0][0], landmarks[0][1], width, height)
    index_tip = _landmark_to_pixel(landmarks[8][0], landmarks[8][1], width, height)
    cx = (wrist[0] + index_tip[0]) // 2
    cy = (wrist[1] + index_tip[1]) // 2
    angle = math.degrees(math.atan2(index_tip[1] - wrist[1], index_tip[0] - wrist[0])) - 55.0
    size = max(48, int(min(width, height) * 0.13))
    blit_brush(frame, cx, cy, max_height=size, angle_deg=angle)


#8
def _landmark_to_pixel(x: float, y: float, width: int, height: int) -> Point:
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return int(x * width), int(y * height)
    return int(x), int(y)


#8
def _draw_screen_hand(
    frame: np.ndarray,
    landmarks: Sequence[Tuple[float, float]],
    width: int,
    height: int,
    *,
    grab_anchor: Optional[Point] = None,
) -> None:
    """Punkt-hånd frit i billedet — som PiP — valgfri streg til del ved klem."""
    source = landmarks if len(landmarks) >= 21 else _POINT_HAND
    placed: List[Point] = [_landmark_to_pixel(x, y, width, height) for x, y in source]
    thick = max(3, int(min(width, height) * 0.012))
    joint_r = max(5, thick + 2)
    max_seg = float(min(width, height) * 0.38)
    for a, b in _POINT_HAND_LINKS:
        ax, ay = placed[a]
        bx, by = placed[b]
        if math.hypot(ax - bx, ay - by) > max_seg:
            continue
        cv2.line(frame, placed[a], placed[b], _HAND_LINE, thick, cv2.LINE_AA)
    for px, py in placed:
        cv2.circle(frame, (px, py), joint_r, _HAND_JOINT, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), max(3, joint_r * 3 // 5), _HAND_LINE, -1, cv2.LINE_AA)
    if grab_anchor is not None:
        cv2.line(frame, placed[9], grab_anchor, (180, 220, 255), 2, cv2.LINE_AA)


#8
def apply_hand_input(scene: "HouseBuildScene", track: Optional["HandTrack"], camera: np.ndarray) -> None:
    """Værktøjskort, maling eller flytning af del — afhængigt af valg og klem."""
    h, w = camera.shape[:2]
    if track is None:
        scene.steer(camera.shape, None)
        return
    lm = _landmarks_on_frame(track.points)
    px, py = _palm_frame_px(lm, w, h)
    card_tool = scene.tool_bar.tool_at(px, py, w)
    if card_tool is not None and track.clenched:
        scene.select_tool(card_tool)
        scene.steer(camera.shape, (px, py), holding=False, landmarks=lm)
        return
    if scene.brush_active() and track.clenched and py > scene.tool_bar.BAR_H + 4:
        scene.paint_at_pixel(px, py)
        scene.steer(camera.shape, (px, py), holding=False, landmarks=lm)
        return
    if scene.brush_active():
        scene.steer(camera.shape, (px, py), holding=False, landmarks=lm)
        return
    scene.steer(
        camera.shape,
        (px, py),
        holding=track.clenched,
        landmarks=lm,
    )


def _uv_on_placed_material(uv: Tuple[float, float], placed: Sequence[PlacedPiece]) -> bool:
    u, v = uv
    for piece in placed:
        tu, tv = piece.step.target_u, piece.step.target_v
        if math.hypot(u - tu, v - tv) < 0.24:
            return True
    return False


def _pixel_to_slab_uv(top: np.ndarray, px: int, py: int) -> Optional[Tuple[float, float]]:
    if cv2.pointPolygonTest(top, (float(px), float(py)), False) < 0:
        return None
    best_u, best_v = 0.5, 0.5
    best_d = 1e12
    for i in range(33):
        u = i / 32.0
        for j in range(33):
            v = j / 32.0
            sx, sy = _slab_point(top, u, v)
            d = (sx - px) ** 2 + (sy - py) ** 2
            if d < best_d:
                best_d = d
                best_u, best_v = u, v
    if best_d > 55 * 55:
        return None
    return best_u, best_v


def _slab_point(top: np.ndarray, u: float, v: float) -> Point:
    fl, fr, nr, nl = [np.array(p, dtype=float) for p in top]
    along_top = fl * (1.0 - u) + fr * u
    along_bottom = nl * (1.0 - u) + nr * u
    p = along_top * (1.0 - v) + along_bottom * v
    return int(p[0]), int(p[1])


def _paint_piece(
    frame: np.ndarray,
    top: np.ndarray,
    step: BuildStep,
    u: float,
    v: float,
    *,
    ghost: bool,
) -> None:
    _part(step.kind).paint(frame, top, step, u, v, ghost=ghost)


def _paint_post(frame: np.ndarray, base: Point, depth_v: float, *, ghost: bool) -> None:
    bx, by = base
    scale = 0.7 + 0.45 * depth_v
    post_h = int(108 * scale)
    post_w = max(12, int(18 * scale))
    x1, y1 = bx - post_w // 2, by - post_h
    x2, y2 = bx + post_w // 2, by
    color = GHOST if ghost else WOOD
    if ghost:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        cv2.addWeighted(frame, 0.72, overlay, 0.28, 0, dst=frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), WOOD_HI, 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x2, y2), WOOD_EDGE, 1, cv2.LINE_AA)


def _paint_rim(
    frame: np.ndarray,
    top: np.ndarray,
    step: BuildStep,
    u: float,
    v: float,
    *,
    ghost: bool,
) -> None:
    u0, u1 = 0.18, 0.82
    v0, v1 = 0.18, 0.82
    if step.rim_horizontal:
        p0 = _slab_point(top, u0, v)
        p1 = _slab_point(top, u1, v)
    else:
        p0 = _slab_point(top, u, v0)
        p1 = _slab_point(top, u, v1)
    thickness = max(8, int(10 + v * 6))
    color = GHOST if ghost else WOOD_HI
    if ghost:
        cv2.line(frame, p0, p1, (200, 255, 255), thickness + 2, cv2.LINE_AA)
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)
    else:
        cv2.line(frame, p0, p1, WOOD, thickness + 2, cv2.LINE_AA)
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)


def _depth_scale(v: float) -> float:
    return 0.7 + 0.45 * v


class HousePart:
    """En del der kan sættes på huset."""

    def paint(
        self,
        frame: np.ndarray,
        top: np.ndarray,
        step: BuildStep,
        u: float,
        v: float,
        *,
        ghost: bool,
    ) -> None:
        raise NotImplementedError

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        return _slab_point(top, u, v)

    def span(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> int:
        return int(90 * _depth_scale(v))


class PostPart(HousePart):
    def paint(
        self,
        frame: np.ndarray,
        top: np.ndarray,
        step: BuildStep,
        u: float,
        v: float,
        *,
        ghost: bool,
    ) -> None:
        _paint_post(frame, _slab_point(top, u, v), v, ghost=ghost)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        base = _slab_point(top, u, v)
        return base[0], base[1] - int(108 * _depth_scale(v)) // 2

    def span(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> int:
        return int(108 * _depth_scale(v))


class WallPart(HousePart):
    """#8 Modulvæg mellem stolperne. En fuge deler panelet i to moduler."""

    def paint(
        self,
        frame: np.ndarray,
        top: np.ndarray,
        step: BuildStep,
        u: float,
        v: float,
        *,
        ghost: bool,
    ) -> None:
        quad = np.array(_wall_quad(top, step, u, v), np.int32)
        if ghost:
            cv2.polylines(frame, [quad], True, (200, 255, 255), 2, cv2.LINE_AA)
            return
        cv2.fillConvexPoly(frame, quad, (176, 198, 214))
        cv2.polylines(frame, [quad], True, (140, 160, 176), 2, cv2.LINE_AA)
        seam_a = (
            (quad[0][0] + quad[1][0]) // 2,
            (quad[0][1] + quad[1][1]) // 2,
        )
        seam_b = (
            (quad[2][0] + quad[3][0]) // 2,
            (quad[2][1] + quad[3][1]) // 2,
        )
        cv2.line(frame, seam_a, seam_b, (110, 128, 142), 3, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        quad = _wall_quad(top, step, u, v)
        return (
            sum(p[0] for p in quad) // 4,
            sum(p[1] for p in quad) // 4,
        )

    def span(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> int:
        quad = _wall_quad(top, step, u, v)
        width = math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1])
        height = math.hypot(quad[3][0] - quad[0][0], quad[3][1] - quad[0][1])
        return int(max(width, height))


class RimPart(HousePart):
    def paint(
        self,
        frame: np.ndarray,
        top: np.ndarray,
        step: BuildStep,
        u: float,
        v: float,
        *,
        ghost: bool,
    ) -> None:
        _paint_rim(frame, top, step, u, v, ghost=ghost)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        p0, p1 = _rim_ends(top, step, u, v)
        return (p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2

    def span(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> int:
        p0, p1 = _rim_ends(top, step, u, v)
        return int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))


class DoorPart(HousePart):
    """#8 Dør der står på pladens kant."""

    def paint(
        self,
        frame: np.ndarray,
        top: np.ndarray,
        step: BuildStep,
        u: float,
        v: float,
        *,
        ghost: bool,
    ) -> None:
        bx, by = _slab_point(top, u, v)
        scale = _depth_scale(v)
        dw = max(22, int(42 * scale))
        dh = int(98 * scale)
        x1, y1, x2, y2 = bx - dw // 2, by - dh, bx + dw // 2, by
        if ghost:
            _ghost_rect(frame, x1, y1, x2, y2, GHOST)
            return
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 72, 118), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1 + 4, y1 + 6), (x2 - 4, y2 - 8), (28, 52, 88), -1, cv2.LINE_AA)
        cv2.circle(frame, (x2 - 8, (y1 + y2) // 2), max(2, dw // 10), (190, 210, 230), -1, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        base = _slab_point(top, u, v)
        return base[0], base[1] - int(98 * _depth_scale(v)) // 2

    def span(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> int:
        return int(98 * _depth_scale(v))


class WindowPart(HousePart):
    """#8 Vindue i væggen, med karm og rude."""

    def paint(
        self,
        frame: np.ndarray,
        top: np.ndarray,
        step: BuildStep,
        u: float,
        v: float,
        *,
        ghost: bool,
    ) -> None:
        bx, by = _slab_point(top, u, v)
        scale = _depth_scale(v)
        ww = max(16, int(30 * scale))
        wh = int(36 * scale)
        sill = int(52 * scale)
        x1, y2 = bx - ww // 2, by - sill
        y1, x2 = y2 - wh, bx + ww // 2
        if ghost:
            _ghost_rect(frame, x1, y1, x2, y2, GHOST)
            return
        cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), WOOD, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (210, 186, 110), -1, cv2.LINE_AA)
        cv2.line(frame, (bx, y1), (bx, y2), WOOD_EDGE, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2), WOOD_EDGE, 2, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        base = _slab_point(top, u, v)
        scale = _depth_scale(v)
        return base[0], base[1] - int(52 * scale) - int(18 * scale)

    def span(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> int:
        return int(36 * _depth_scale(v))


class RoofPart(HousePart):
    """#8 Sadeltag over stolperne. Det flyttes med midten."""

    def paint(
        self,
        frame: np.ndarray,
        top: np.ndarray,
        step: BuildStep,
        u: float,
        v: float,
        *,
        ghost: bool,
    ) -> None:
        eaves, far_ridge, near_ridge = _roof_points(top, u, v)
        left = np.array([eaves[0], far_ridge, near_ridge, eaves[3]], np.int32)
        right = np.array([eaves[1], far_ridge, near_ridge, eaves[2]], np.int32)
        if ghost:
            cv2.polylines(frame, [left, right], True, (200, 255, 255), 2, cv2.LINE_AA)
            return
        cv2.fillConvexPoly(frame, left, (36, 44, 118))
        cv2.fillConvexPoly(frame, right, (48, 60, 156))
        cv2.line(frame, far_ridge, near_ridge, (70, 80, 180), 2, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        _eaves, far_ridge, near_ridge = _roof_points(top, u, v)
        return (far_ridge[0] + near_ridge[0]) // 2, (far_ridge[1] + near_ridge[1]) // 2

    def span(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> int:
        eaves, _far, _near = _roof_points(top, u, v)
        return int(math.hypot(eaves[1][0] - eaves[0][0], eaves[1][1] - eaves[0][1]))


_PARTS: Dict[str, HousePart] = {
    "post": PostPart(),
    "wall": WallPart(),
    "rim": RimPart(),
    "door": DoorPart(),
    "window": WindowPart(),
    "roof": RoofPart(),
}


def _part(kind: str) -> HousePart:
    return _PARTS[kind]


def _wall_quad(top: np.ndarray, step: BuildStep, u: float, v: float) -> List[Point]:
    """Bund langs kanten, top i samme højde som stolperne."""
    bottom0, bottom1 = _rim_ends(top, step, u, v)
    if step.rim_horizontal:
        lifts = (_depth_scale(v), _depth_scale(v))
    else:
        lifts = (_depth_scale(0.18), _depth_scale(0.82))
    top0 = (bottom0[0], bottom0[1] - int(108 * lifts[0]))
    top1 = (bottom1[0], bottom1[1] - int(108 * lifts[1]))
    return [bottom0, bottom1, top1, top0]


def _rim_ends(top: np.ndarray, step: BuildStep, u: float, v: float) -> Tuple[Point, Point]:
    u0, u1 = 0.18, 0.82
    v0, v1 = 0.18, 0.82
    if step.rim_horizontal:
        return _slab_point(top, u0, v), _slab_point(top, u1, v)
    return _slab_point(top, u, v0), _slab_point(top, u, v1)


def _ghost_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, fill: Tuple[int, int, int]) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
    cv2.addWeighted(frame, 0.72, overlay, 0.28, 0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 255, 255), 2, cv2.LINE_AA)


def _roof_points(top: np.ndarray, u: float, v: float) -> Tuple[List[Point], Point, Point]:
    far_left, far_right, near_right, near_left = [tuple(int(c) for c in p) for p in top]
    far_lift = int(108 * _depth_scale(0.16))
    near_lift = int(108 * _depth_scale(0.84))
    ridge = int(46 * _depth_scale(0.5))
    eaves = [
        (far_left[0], far_left[1] - far_lift),
        (far_right[0], far_right[1] - far_lift),
        (near_right[0], near_right[1] - near_lift),
        (near_left[0], near_left[1] - near_lift),
    ]
    far_ridge = ((eaves[0][0] + eaves[1][0]) // 2, eaves[0][1] - ridge)
    near_ridge = ((eaves[2][0] + eaves[3][0]) // 2, eaves[2][1] - ridge)
    dx = int((u - 0.5) * (far_right[0] - far_left[0]))
    dy = int((v - 0.5) * (near_left[1] - far_left[1]))

    def shift(point: Point) -> Point:
        return point[0] + dx, point[1] + dy

    return [shift(p) for p in eaves], shift(far_ridge), shift(near_ridge)


# Legacy — tests refererer til TimberBeam.
class TimberBeam:
    half_extents = (2.05, 0.28, 0.36)


#8
def _paint_yard(frame: np.ndarray, horizon: int) -> None:
    """Blå himmel og græs rundt om betonfundamentet."""
    height = frame.shape[0]
    horizon = max(1, min(horizon, height - 1))
    rows = np.arange(horizon, dtype=np.float32)
    fade = rows / max(horizon - 1, 1)
    blue = (214 * (1.0 - fade) + 246 * fade).astype(np.uint8)
    green = (142 * (1.0 - fade) + 198 * fade).astype(np.uint8)
    red = (64 * (1.0 - fade) + 150 * fade).astype(np.uint8)
    frame[:horizon, :, 0] = blue[:, None]
    frame[:horizon, :, 1] = green[:, None]
    frame[:horizon, :, 2] = red[:, None]
    frame[horizon:, :] = (72, 164, 92)


def _foundation_quads(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    left = int(width * 0.07)
    right = width - left
    inset = int(width * 0.11)
    far_y = int(height * 0.30)
    near_y = int(height * 0.70)
    front_y = int(height * 0.84)
    top = np.array(
        [[left + inset, far_y], [right - inset, far_y], [right, near_y], [left, near_y]],
        dtype=np.int32,
    )
    front = np.array(
        [[left, near_y], [right, near_y], [right - width // 80, front_y], [left + width // 80, front_y]],
        dtype=np.int32,
    )
    return top, front


def _outset(poly: np.ndarray, amount: float) -> np.ndarray:
    center = poly.astype(float).mean(axis=0)
    moved: List[List[int]] = []
    for point in poly.astype(float):
        direction = point - center
        length = float(np.linalg.norm(direction)) or 1.0
        moved.append((point + direction / length * amount).astype(int).tolist())
    return np.array(moved, dtype=np.int32)


def _segment(
    a0: Sequence[int],
    a1: Sequence[int],
    b0: Sequence[int],
    b1: Sequence[int],
    t: float,
) -> Tuple[Point, Point]:
    start = (int(a0[0] * (1 - t) + a1[0] * t), int(a0[1] * (1 - t) + a1[1] * t))
    end = (int(b0[0] * (1 - t) + b1[0] * t), int(b0[1] * (1 - t) + b1[1] * t))
    return start, end


def _inset_corners(poly: np.ndarray, amount: float) -> List[Point]:
    points = [tuple(int(v) for v in point) for point in poly]
    count = len(points)
    inset: List[Point] = []
    for index, point in enumerate(points):
        previous = points[(index - 1) % count]
        nxt = points[(index + 1) % count]
        inset.append(
            (
                int(point[0] + (previous[0] - point[0]) * amount + (nxt[0] - point[0]) * amount),
                int(point[1] + (previous[1] - point[1]) * amount + (nxt[1] - point[1]) * amount),
            )
        )
    return inset
