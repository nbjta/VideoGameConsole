#10
"""Mars-rover: samme bygge-loop som Husbygning, med rover-dele på Mars.

Klem, snap, værktøjskort og maling kommer fra HouseBuildScene.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from gamConsole.hand_pip_overlay import HandPipOverlay
from gamConsole.house_build import (
    BuildStep,
    ConcreteFoundation,
    GHOST,
    HouseBuildScene,
    HousePart,
    _depth_scale,
    _foundation_quads,
    _outset,
    _paint_piece,
    _slab_point,
    apply_hand_input,
)
from gamConsole.house_tool_cards import ToolCardBar
from gamConsole.opencv_ui import draw_text, draw_text_centered, run_window_loop

Point = Tuple[int, int]

#10 Samme rækkefølge som huset: først bærende dele, så udstyr.
ROVER_PLAN: List[BuildStep] = [
    BuildStep("chassis", 0.50, 0.50, "Roverkrop"),
    BuildStep("solar", 0.50, 0.50, "Solpanel"),
    BuildStep("rocker", 0.24, 0.50, "Ophæng venstre"),
    BuildStep("rocker", 0.76, 0.50, "Ophæng højre"),
    BuildStep("wheel", 0.24, 0.30, "Hjul foran venstre"),
    BuildStep("wheel", 0.24, 0.50, "Hjul midt venstre"),
    BuildStep("wheel", 0.24, 0.70, "Hjul bag venstre"),
    BuildStep("wheel", 0.76, 0.30, "Hjul foran højre"),
    BuildStep("wheel", 0.76, 0.50, "Hjul midt højre"),
    BuildStep("wheel", 0.76, 0.70, "Hjul bag højre"),
    BuildStep("mast", 0.50, 0.38, "Mast"),
    BuildStep("camera", 0.50, 0.38, "Kamera"),
    BuildStep("hazcam", 0.50, 0.30, "Forreste kameraer"),
    BuildStep("arm", 0.58, 0.44, "Robotarm"),
    BuildStep("drill", 0.64, 0.28, "Borehoved"),
    BuildStep("antenna", 0.38, 0.46, "Parabol"),
    BuildStep("power", 0.50, 0.74, "Strømkasse"),
]

_BODY = (176, 188, 214)
_BODY_SIDE = (140, 150, 172)
_GOLD = (46, 168, 226)
_TIRE = (32, 34, 40)
_HUB = (168, 174, 186)
_METAL = (62, 70, 84)
_SOLAR = (150, 92, 46)
_SOLAR_LINE = (196, 176, 120)
_LENS = (210, 170, 40)
_POWER = (232, 232, 236)


class MarsPad(ConcreteFoundation):
    """Landingsplade på Mars. Samme flade som betonpladen, så snap-koordinaterne passer."""

    def paint(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = frame.shape[:2]
        horizon = int(height * 0.22)
        _paint_mars(frame, horizon)
        top, front = _foundation_quads(width, height)
        berm = _outset(top, max(10, width // 48))
        cv2.fillConvexPoly(frame, berm, (42, 58, 118))
        cv2.fillConvexPoly(frame, front, (78, 96, 132))
        cv2.fillConvexPoly(frame, top, (158, 172, 196))
        self._speckle(frame, top)
        self._joints(frame, top)
        self._bolts(frame, top)
        return top, front


class MarsRoverBuildScene(HouseBuildScene):
    """Husbygningens snap og hånd, med rover-plan og Mars-plade."""

    def __init__(self) -> None:
        super().__init__()
        _install_rover_parts()
        self.foundation = MarsPad()
        self.steps = list(ROVER_PLAN)
        self.step_index = 0
        self.placed.clear()
        self.paint_dots.clear()
        self.complete = False
        self.snap_anim = 0
        self.reward_frames = 0
        self.preview = False
        self._load_step(start_offset=True)

    def _load_step(self, *, start_offset: bool) -> None:
        #10 Start væk fra mærket. Husets start (0,5, 0,55) ligger inden for snap af roverkroppen.
        step = self._current()
        if step is None:
            self.complete = True
            return
        if start_offset:
            self.beam_u = 0.16
            self.beam_v = 0.86

    def steer(
        self,
        frame_shape: Sequence[int],
        centroid: Optional[Tuple[int, int]],
        *,
        holding: bool = True,
        landmarks: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> None:
        super().steer(frame_shape, centroid, holding=holding, landmarks=landmarks)
        #10 Delen står stille under snap og når roveren er færdig. Hånden skal blive ved med at følge.
        if centroid is None:
            self._grip_px = None
            self._grip_landmarks = None
            self._grip_holding = False
            return
        self._grip_px = (int(centroid[0]), int(centroid[1]))
        self._grip_landmarks = landmarks
        self._grip_holding = holding

    def place_all_for_preview(self) -> None:
        """Sæt alle dele på mærkerne, uden kamera. Bruges til menu-billedet."""
        while not self.complete:
            step = self._current()
            if step is None:
                return
            self.beam_u = step.target_u
            self.beam_v = step.target_v
            self.reward_frames = 0
            self._update_snap()
            self.reward_frames = 0
            self._update_snap()

    def draw(self, frame: np.ndarray) -> None:
        if not self.preview:
            super().draw(frame)
            return
        top, _front = self.foundation.paint(frame)
        self._slab_top = top
        for piece in self.placed:
            _paint_piece(
                frame,
                top,
                piece.step,
                piece.step.target_u,
                piece.step.target_v,
                ghost=False,
            )

    def _draw_hud(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if self.complete:
            #10 Banner i bunden. Teksten midt i billedet blev dækket af hånden og roveren.
            _draw_done_banner(frame)
            return
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
            color=(236, 228, 220),
        )
        if self.reward_frames > 0:
            draw_text(
                frame,
                "På plads! Næste opgave …",
                (width // 2 - 120, int(height * 0.14)),
                scale=0.62,
                color=(120, 220, 255),
            )
        elif self.snap_anim > 0:
            draw_text(frame, "Snap …", (width // 2 - 40, int(height * 0.14)), scale=0.55, color=(180, 220, 255))


def _install_rover_parts() -> None:
    from gamConsole import house_build

    house_build._PARTS.update(_ROVER_PARTS)


class _BoxPart(HousePart):
    du = 0.14
    dv = 0.16
    lift = 48.0
    top_color = _BODY
    side_color = _BODY_SIDE
    edge_color = _GOLD

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
        uv, bottom, roof = _box_points(top, u, v, self.du, self.dv, self.lift)
        del uv
        if ghost:
            _ghost_closed(frame, roof)
            _ghost_closed(frame, bottom)
            return
        _fill_box(frame, bottom, roof, self.top_color, self.side_color, self.edge_color)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        _uv, _bottom, roof = _box_points(top, u, v, self.du, self.dv, self.lift)
        return _mean_point(roof)


class ChassisPart(_BoxPart):
    """Roverens krop. Bred kasse med guldkant, større end solpanelet."""

    du = 0.17
    dv = 0.18
    lift = 70.0
    top_color = (198, 210, 232)
    side_color = (124, 136, 160)


class SolarPart(HousePart):
    """Fladt solpanel oven på kroppen."""

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
        _uv, _bottom, roof = _box_points(top, u, v, 0.09, 0.10, 78.0)
        if ghost:
            _ghost_closed(frame, roof)
            return
        poly = np.array(roof, np.int32)
        cv2.fillConvexPoly(frame, poly, _SOLAR)
        cv2.polylines(frame, [poly], True, _SOLAR_LINE, 2, cv2.LINE_AA)
        cv2.line(frame, roof[0], roof[2], _SOLAR_LINE, 1, cv2.LINE_AA)
        cv2.line(frame, roof[1], roof[3], _SOLAR_LINE, 1, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        _uv, _bottom, roof = _box_points(top, u, v, 0.09, 0.10, 78.0)
        return _mean_point(roof)


class PowerPart(_BoxPart):
    """Strømkasse bag på roveren, med køleribber. Den rager ud bag kroppen."""

    du = 0.09
    dv = 0.07
    lift = 58.0
    top_color = _POWER
    side_color = (190, 192, 198)
    edge_color = (90, 96, 110)

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
        super().paint(frame, top, step, u, v, ghost=ghost)
        if ghost:
            return
        _uv, bottom, roof = _box_points(top, u, v, self.du, self.dv, self.lift)
        near_left, near_right = bottom[3], bottom[2]
        for t in (0.3, 0.5, 0.7):
            x = int(near_left[0] * (1 - t) + near_right[0] * t)
            y0 = int(near_left[1] * (1 - t) + near_right[1] * t)
            y1 = int(roof[3][1] * (1 - t) + roof[2][1] * t)
            cv2.line(frame, (x, y0), (x, y1), (120, 126, 136), 2, cv2.LINE_AA)


class RockerPart(HousePart):
    """Ophæng langs den ene side. Det følger hånden i begge retninger."""

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
        span = 0.20
        front = _wheel_center(top, u, max(0.08, v - span))
        rear = _wheel_center(top, u, min(0.92, v + span))
        mid = _wheel_center(top, u, v)
        scale = _depth_scale(v)
        pivot = (mid[0], mid[1] - int(26 * scale))
        inward = 0.13 if u < 0.5 else -0.13
        body = _wheel_center(top, min(0.94, max(0.06, u + inward)), v)
        body = (body[0], body[1] - int(40 * scale))
        color = (200, 255, 255) if ghost else _METAL
        thick = 2 if ghost else max(7, int(10 * scale))
        cv2.line(frame, front, pivot, color, thick, cv2.LINE_AA)
        cv2.line(frame, pivot, rear, color, thick, cv2.LINE_AA)
        cv2.line(frame, pivot, body, color, thick, cv2.LINE_AA)
        radius = max(4, int(7 * scale))
        for point in (front, pivot, rear, body):
            cv2.circle(frame, point, radius, color, 1 if ghost else -1, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        return _wheel_center(top, u, v)


class WheelPart(HousePart):
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
        cx, cy = _wheel_center(top, u, v)
        radius = _wheel_radius(v)
        if ghost:
            cv2.circle(frame, (cx, cy), radius, (200, 255, 255), 2, cv2.LINE_AA)
            return
        cv2.circle(frame, (cx, cy), radius, _TIRE, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), max(3, radius - 3), (70, 74, 82), 2, cv2.LINE_AA)
        for spoke in range(5):
            angle = spoke * math.pi / 2.5
            tip = (
                cx + int(math.cos(angle) * radius * 0.72),
                cy + int(math.sin(angle) * radius * 0.72),
            )
            cv2.line(frame, (cx, cy), tip, (88, 92, 100), 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), max(3, radius // 3), _HUB, -1, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        return _wheel_center(top, u, v)


class MastPart(HousePart):
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
        x1, y1, x2, y2 = _mast_rect(top, u, v)
        if ghost:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 255, 255), 2, cv2.LINE_AA)
            return
        cv2.rectangle(frame, (x1 - 6, y2 - 4), (x2 + 6, y2 + 3), _METAL, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (176, 184, 196), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _GOLD, 1, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        x1, y1, x2, y2 = _mast_rect(top, u, v)
        return (x1 + x2) // 2, (y1 + y2) // 2


class CameraPart(HousePart):
    """Kamerahovedet sidder i masthøjde, så det lander oven på masten."""

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
        x1, y1, x2, y2 = _camera_rect(top, u, v)
        if ghost:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 255, 255), 2, cv2.LINE_AA)
            return
        cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 38, 44), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _GOLD, 1, cv2.LINE_AA)
        lens_r = max(3, (y2 - y1) // 3)
        cy = (y1 + y2) // 2
        for side in (-1, 1):
            cv2.circle(frame, (x1 + (x2 - x1) // 2 + side * (lens_r + 3), cy), lens_r, _LENS, -1, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        x1, y1, x2, y2 = _camera_rect(top, u, v)
        return (x1 + x2) // 2, (y1 + y2) // 2


class ArmPart(HousePart):
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
        shoulder, elbow, tip = _arm_points(top, u, v)
        color = (200, 255, 255) if ghost else _METAL
        thick = 2 if ghost else max(4, int(6 * _depth_scale(v)))
        cv2.line(frame, shoulder, elbow, color, thick, cv2.LINE_AA)
        cv2.line(frame, elbow, tip, color, thick, cv2.LINE_AA)
        cv2.circle(frame, elbow, max(3, thick), color, 1 if ghost else -1, cv2.LINE_AA)
        cv2.circle(frame, tip, max(4, thick), color, 1 if ghost else -1, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        _shoulder, elbow, _tip = _arm_points(top, u, v)
        return elbow


class HazcamPart(HousePart):
    """To små kameraer på forkanten af kroppen."""

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
        base = _slab_point(top, u, v)
        scale = _depth_scale(v)
        color = (200, 255, 255) if ghost else (28, 30, 34)
        for side in (-1, 1):
            cx = base[0] + side * int(16 * scale)
            cy = base[1] - int(18 * scale)
            half_w = max(6, int(9 * scale))
            half_h = max(5, int(7 * scale))
            x1, y1, x2, y2 = cx - half_w, cy - half_h, cx + half_w, cy + half_h
            if ghost:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), _GOLD, 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), max(2, half_h // 2), _LENS, -1, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        base = _slab_point(top, u, v)
        return base[0], base[1] - int(18 * _depth_scale(v))


class DrillPart(HousePart):
    """Borehoved for enden af armen."""

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
        hw = max(8, int(12 * scale))
        hh = max(8, int(14 * scale))
        x1, y1, x2, y2 = bx - hw, by - hh - int(16 * scale), bx + hw, by - int(16 * scale)
        color = (200, 255, 255) if ghost else (90, 98, 110)
        if ghost:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.line(frame, (bx, y2), (bx, by), color, 2, cv2.LINE_AA)
            return
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _GOLD, 1, cv2.LINE_AA)
        cv2.line(frame, (bx, y2), (bx, by), _GOLD, max(2, int(3 * scale)), cv2.LINE_AA)
        cv2.circle(frame, (bx, by), max(3, int(4 * scale)), (40, 44, 52), -1, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        bx, by = _slab_point(top, u, v)
        return bx, by - int(24 * _depth_scale(v))


class AntennaPart(HousePart):
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
        base = _slab_point(top, u, v)
        scale = _depth_scale(v)
        head = (base[0], base[1] - int(92 * scale))
        pole = (200, 255, 255) if ghost else _METAL
        dish = (200, 255, 255) if ghost else (48, 52, 60)
        cv2.line(frame, base, head, pole, 3, cv2.LINE_AA)
        axes = (max(14, int(28 * scale)), max(7, int(14 * scale)))
        cv2.ellipse(frame, head, axes, -35, 0, 360, dish, -1, cv2.LINE_AA)
        cv2.ellipse(frame, head, axes, -35, 0, 360, _GOLD if not ghost else GHOST, 2, cv2.LINE_AA)

    def anchor(self, top: np.ndarray, step: BuildStep, u: float, v: float) -> Point:
        base = _slab_point(top, u, v)
        return base[0], base[1] - int(92 * _depth_scale(v))


_ROVER_PARTS = {
    "chassis": ChassisPart(),
    "solar": SolarPart(),
    "rocker": RockerPart(),
    "wheel": WheelPart(),
    "mast": MastPart(),
    "camera": CameraPart(),
    "hazcam": HazcamPart(),
    "arm": ArmPart(),
    "drill": DrillPart(),
    "antenna": AntennaPart(),
    "power": PowerPart(),
}


def _wheel_radius(v: float) -> int:
    return max(8, int(16 * _depth_scale(v)))


def _wheel_center(top: np.ndarray, u: float, v: float) -> Point:
    bx, by = _slab_point(top, u, v)
    return bx, by - _wheel_radius(v) // 3


def _mast_rect(top: np.ndarray, u: float, v: float) -> Tuple[int, int, int, int]:
    bx, by = _slab_point(top, u, v)
    scale = _depth_scale(v)
    half = max(4, int(5 * scale))
    height = int(112 * scale)
    return bx - half, by - height, bx + half, by


def _camera_rect(top: np.ndarray, u: float, v: float) -> Tuple[int, int, int, int]:
    _x1, mast_top, _x2, _base = _mast_rect(top, u, v)
    bx = (_x1 + _x2) // 2
    scale = _depth_scale(v)
    half_w = max(10, int(16 * scale))
    height = max(10, int(16 * scale))
    return bx - half_w, mast_top - height, bx + half_w, mast_top


def _arm_points(top: np.ndarray, u: float, v: float) -> Tuple[Point, Point, Point]:
    scale = _depth_scale(v)
    shoulder = _slab_point(top, u, v)
    shoulder = (shoulder[0], shoulder[1] - int(30 * scale))
    elbow = _slab_point(top, min(0.94, u + 0.04), max(0.06, v - 0.08))
    elbow = (elbow[0], elbow[1] - int(46 * scale))
    tip = _slab_point(top, min(0.96, u + 0.06), max(0.05, v - 0.16))
    tip = (tip[0], tip[1] - int(22 * scale))
    return shoulder, elbow, tip


def _box_points(
    top: np.ndarray,
    u: float,
    v: float,
    du: float,
    dv: float,
    lift: float,
) -> Tuple[Sequence[Tuple[float, float]], List[Point], List[Point]]:
    uv = (
        (u - du, max(0.02, v - dv)),
        (u + du, max(0.02, v - dv)),
        (u + du, min(0.98, v + dv)),
        (u - du, min(0.98, v + dv)),
    )
    bottom = [_slab_point(top, cu, cv) for cu, cv in uv]
    roof = [
        (point[0], point[1] - int(lift * _depth_scale(cv)))
        for point, (_cu, cv) in zip(bottom, uv)
    ]
    return uv, bottom, roof


def _fill_box(
    frame: np.ndarray,
    bottom: Sequence[Point],
    roof: Sequence[Point],
    top_color: Tuple[int, int, int],
    side_color: Tuple[int, int, int],
    edge_color: Tuple[int, int, int],
) -> None:
    left = np.array([bottom[0], bottom[3], roof[3], roof[0]], np.int32)
    right = np.array([bottom[1], bottom[2], roof[2], roof[1]], np.int32)
    near = np.array([bottom[3], bottom[2], roof[2], roof[3]], np.int32)
    cv2.fillConvexPoly(frame, left, side_color)
    cv2.fillConvexPoly(frame, right, side_color)
    cv2.fillConvexPoly(frame, near, side_color)
    roof_poly = np.array(roof, np.int32)
    cv2.fillConvexPoly(frame, roof_poly, top_color)
    cv2.polylines(frame, [roof_poly], True, edge_color, 2, cv2.LINE_AA)


def _draw_done_banner(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    y1 = height - 58
    cv2.rectangle(frame, (0, y1), (width, height), (24, 28, 36), -1)
    cv2.line(frame, (0, y1), (width, y1), (80, 180, 230), 2, cv2.LINE_AA)
    draw_text_centered(frame, "Roveren er færdig!", width // 2, y1 + 12, scale=0.82, color=(140, 255, 190))


def _ghost_closed(frame: np.ndarray, points: Sequence[Point]) -> None:
    cv2.polylines(frame, [np.array(points, np.int32)], True, GHOST, 2, cv2.LINE_AA)


def _mean_point(points: Sequence[Point]) -> Point:
    return (
        sum(point[0] for point in points) // len(points),
        sum(point[1] for point in points) // len(points),
    )


def _paint_mars(frame: np.ndarray, horizon: int) -> None:
    """Støvet rød himmel, højderyg og rustfarvet jord."""
    height, width = frame.shape[:2]
    horizon = max(1, min(horizon, height - 1))
    rows = np.arange(horizon, dtype=np.float32)
    fade = rows / max(horizon - 1, 1)
    frame[:horizon, :, 0] = (46 * (1.0 - fade) + 78 * fade).astype(np.uint8)[:, None]
    frame[:horizon, :, 1] = (58 * (1.0 - fade) + 118 * fade).astype(np.uint8)[:, None]
    frame[:horizon, :, 2] = (108 * (1.0 - fade) + 198 * fade).astype(np.uint8)[:, None]
    ground = np.arange(height - horizon, dtype=np.float32)
    gfade = ground / max(height - horizon - 1, 1)
    frame[horizon:, :, 0] = (48 * (1.0 - gfade) + 30 * gfade).astype(np.uint8)[:, None]
    frame[horizon:, :, 1] = (64 * (1.0 - gfade) + 42 * gfade).astype(np.uint8)[:, None]
    frame[horizon:, :, 2] = (158 * (1.0 - gfade) + 112 * gfade).astype(np.uint8)[:, None]
    _horizon_ridge(frame, horizon)
    sun_center = (int(width * 0.80), int(horizon * 0.42))
    sun_r = max(8, width // 22)
    cv2.circle(frame, sun_center, sun_r, (150, 196, 236), -1, cv2.LINE_AA)
    cv2.circle(frame, sun_center, max(3, sun_r // 2), (190, 220, 250), -1, cv2.LINE_AA)
    _craters_and_rocks(frame)


def _horizon_ridge(frame: np.ndarray, horizon: int) -> None:
    height, width = frame.shape[:2]
    peaks = (0.0, -0.035, -0.012, -0.07, -0.02, -0.055, -0.018, -0.04, 0.0)
    count = len(peaks)
    points = [
        [int(index * (width - 1) / (count - 1)), int(horizon + peaks[index] * height)]
        for index in range(count)
    ]
    points.append([width - 1, horizon + 4])
    points.append([0, horizon + 4])
    cv2.fillPoly(frame, [np.array(points, np.int32)], (28, 40, 96))


def _craters_and_rocks(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    for nx, ny, rx, ry in (
        (0.18, 0.80, 0.055, 0.022),
        (0.86, 0.76, 0.04, 0.018),
    ):
        center = (int(nx * width), int(ny * height))
        axes = (max(6, int(rx * width)), max(4, int(ry * height)))
        cv2.ellipse(frame, center, axes, 0, 0, 360, (36, 48, 96), -1, cv2.LINE_AA)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (78, 102, 176), 2, cv2.LINE_AA)
    for nx, ny, rx, ry in (
        (0.06, 0.88, 0.028, 0.02),
        (0.14, 0.78, 0.016, 0.012),
        (0.93, 0.86, 0.03, 0.02),
        (0.84, 0.93, 0.02, 0.014),
        (0.32, 0.94, 0.024, 0.016),
    ):
        center = (int(nx * width), int(ny * height))
        axes = (max(4, int(rx * width)), max(3, int(ry * height)))
        cv2.ellipse(frame, center, axes, 0, 0, 360, (26, 34, 72), -1, cv2.LINE_AA)
        cv2.ellipse(frame, center, (axes[0] - 1, max(2, axes[1] // 2)), 0, 200, 340, (70, 90, 150), 1, cv2.LINE_AA)


#10 Samme vindue og samme klem-styring som Husbygning.
def main() -> None:
    parser = argparse.ArgumentParser(description="gamConsole: Mars-rover")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Kunne ikke åbne kamera {args.camera}", file=sys.stderr)
        sys.exit(1)
    scene = MarsRoverBuildScene()
    hand_pip = HandPipOverlay()

    def render():
        ok, frame = cap.read()
        if not ok:
            return None
        #10 Spejlvendt webcam — samme styring som Husbygning.
        camera = cv2.flip(frame, 1)
        track = hand_pip.read(camera)
        apply_hand_input(scene, track, camera)
        scene.draw(frame)
        hand_pip.draw(frame, camera=camera)
        return frame

    try:
        run_window_loop("gamConsole: mars-rover", cap, render)
    finally:
        hand_pip.close()


if __name__ == "__main__":
    main()
