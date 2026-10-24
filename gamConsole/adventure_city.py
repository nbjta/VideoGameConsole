"""Bevægelig cartoon-by i Adventure City-stil.

Original tegning — ikke officiel PAW Patrol-kunst eller -figurer.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]


def _road_layout(h: int, w: int) -> Tuple[int, int, int, int, float, float]:
    """Horizon, road_top, vanish_x, vanish_y, half_bredde ved horisont og ved spiller."""
    #1 Zoomet ind: mere vej i billedet, snævrere horisont.
    horizon = int(h * 0.26)
    road_top = int(h * 0.80)
    vx = w // 2
    vy = horizon + 8
    half_horizon = max(88.0, w * 0.19)
    half_near = max(228.0, w * 0.54)
    return horizon, road_top, vx, vy, half_horizon, half_near


def _road_half_at_y(
    y: float,
    horizon: int,
    road_top: int,
    half_horizon: float,
    half_near: float,
) -> float:
    if y <= horizon:
        return half_horizon
    if y >= road_top:
        return half_near
    t = (y - horizon) / max(road_top - horizon, 1)
    return half_horizon + t * (half_near - half_horizon)


class StreetActor:
    """Noget der kører sidelæns i byen, ligesom bilerne."""

    def __init__(self, phase: float, speed: float, lane: int, start_t: float = 0.0) -> None:
        self.phase = phase
        self.speed = speed
        self.lane = lane
        self.start_t = start_t
        self._prev_x: Optional[int] = None

    def x_on(self, t: float, width: int, margin: int = 90) -> int:
        span = width + margin
        elapsed = max(0.0, t - self.start_t)
        return int((elapsed * self.speed + self.phase * span) % span) - margin // 2

    def wrapped(self, t: float, width: int) -> bool:
        x = self.x_on(t, width)
        prev = self._prev_x
        self._prev_x = x
        if prev is None:
            return False
        return prev > width * 0.72 and x < width * 0.28

    def draw(self, canvas: np.ndarray, t: float) -> None:
        raise NotImplementedError

    def hitbox(self, t: float, width: int, height: int) -> Tuple[int, int, int]:
        raise NotImplementedError


def _tone(color: Color, factor: float) -> Color:
    blue, green, red = color
    return (
        int(max(0, min(255, blue * factor))),
        int(max(0, min(255, green * factor))),
        int(max(0, min(255, red * factor))),
    )


def _fill_round_rect(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Color,
    radius: int,
) -> None:
    if x2 <= x1 or y2 <= y1:
        return
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
    for cx, cy in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(canvas, (cx, cy), radius, color, -1, cv2.LINE_AA)


class CityCar(StreetActor):
    """#1 Modkørende trafik i venstre bane (højre mod venstre)."""

    WHEEL = (96, 98, 102)
    LIGHT = (250, 250, 252)

    def __init__(self, phase: float, speed: float, lane: int, color: Color, length: int) -> None:
        super().__init__(phase, speed, lane)
        self.color = color
        self.length = length

    def x_on(self, t: float, width: int, margin: int = 90) -> int:
        span = width + margin
        elapsed = max(0.0, t - self.start_t)
        return int(width + margin // 2 - (elapsed * self.speed + self.phase * span) % span)

    def _lane_pose(self, t: float, w: int, h: int) -> Tuple[int, int, int]:
        horizon, road_top, vx, vy, half_h, half_n = _road_layout(h, w)
        margin = 90
        prog = float(np.clip((self.x_on(t, w) + margin) / max(w + margin, 1), 0.0, 1.0))
        y = int(road_top - 10 - prog * (road_top - vy) * 0.22)
        half = _road_half_at_y(float(y), horizon, road_top, half_h, half_n)
        lane_shift = half * (0.34 + self.lane * 0.12)
        x = int(vx - lane_shift)
        car_len = max(28, int(self.length * (0.72 + 0.28 * prog)))
        return x, y, car_len

    def draw(self, canvas: np.ndarray, t: float) -> None:
        h, w = canvas.shape[:2]
        front_x, y, car_len = self._lane_pose(t, w, h)
        #1 Frontvendt bil i bilens egen farve. Samme form som reference-ikonet.
        self._paint_front(canvas, front_x - car_len // 2, y, max(40, car_len))

    def _paint_front(self, canvas: np.ndarray, cx: int, top: int, width: int) -> None:
        width = max(24, int(width))
        height = max(18, int(width * 0.74))
        x1 = cx - width // 2
        y1 = top
        x2 = x1 + width
        y2 = top + height
        wheel_w = max(4, width // 6)
        wheel_h = max(4, height // 7)
        cv2.rectangle(canvas, (x1 - wheel_w // 5, y2 - wheel_h), (x1 + wheel_w, y2 + wheel_h // 3), self.WHEEL, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x2 - wheel_w, y2 - wheel_h), (x2 + wheel_w // 5, y2 + wheel_h // 3), self.WHEEL, -1, cv2.LINE_AA)
        body = self.color
        shade = _tone(body, 0.78)
        _fill_round_rect(canvas, x1, y1, x2, y2, body, max(4, width // 7))
        mid = x1 + width // 2
        cv2.rectangle(canvas, (mid, y1 + width // 7), (x2 - width // 7, y2 - width // 7), shade, -1, cv2.LINE_AA)
        glass_top = y1 + max(3, height // 9)
        glass_bottom = y1 + int(height * 0.46)
        glass_left = x1 + width // 7
        glass_right = x2 - width // 7
        _fill_round_rect(canvas, glass_left, glass_top, glass_right, glass_bottom, _tone(body, 0.72), max(3, width // 10))
        cv2.rectangle(
            canvas,
            (mid, glass_top + 2),
            (glass_right - 2, glass_bottom - 2),
            _tone(body, 0.55),
            -1,
            cv2.LINE_AA,
        )
        mirror_y = y1 + int(height * 0.40)
        mirror_rx = max(3, width // 11)
        mirror_ry = max(3, height // 9)
        cv2.ellipse(canvas, (x1, mirror_y), (mirror_rx, mirror_ry), 0, 0, 360, body, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (x2, mirror_y), (mirror_rx, mirror_ry), 0, 0, 360, shade, -1, cv2.LINE_AA)
        light_y = y1 + int(height * 0.66)
        light_r = max(2, width // 11)
        cv2.circle(canvas, (x1 + width // 4, light_y), light_r, self.LIGHT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x2 - width // 4, light_y), light_r, self.LIGHT, -1, cv2.LINE_AA)
        smile_top = light_y + light_r
        smile_bottom = min(y2 - max(2, height // 12), smile_top + max(3, height // 8))
        _fill_round_rect(
            canvas,
            x1 + width // 4,
            smile_top,
            x2 - width // 4,
            smile_bottom,
            _tone(body, 0.58),
            max(2, height // 16),
        )

    def hitbox(self, t: float, width: int, height: int) -> Tuple[int, int, int]:
        x, y, car_len = self._lane_pose(t, width, height)
        return x - car_len // 2, y + 6, 20


#1 Én bil ad gangen. 4 sekunder fra horisonten og ud. 2 sekunder før næste farve.
ONCOMING_APPROACH_SECONDS = 4.0
ONCOMING_GAP_SECONDS = 2.0
ONCOMING_COLORS: Tuple[Color, ...] = (
    (40, 90, 230),
    (40, 200, 90),
    (200, 80, 255),
    (30, 210, 230),
)


class OncomingLaneCar(CityCar):
    """#1 Starter i horisonten, midt i sin bane, og vokser mod kameraet."""

    def __init__(self, color: Color, lane: int = 0) -> None:
        super().__init__(0.0, 0.0, lane, color, 46)
        self.progress = 0.0
        #1 Hver bil kan kun trække point fra én gang.
        self.struck = False

    def pose(self, width: int, height: int) -> Tuple[int, int, int]:
        horizon, road_top, vx, _vy, half_h, half_n = _road_layout(height, width)
        vanish_y = horizon + 8
        near_width = half_n * 0.42
        far_width = max(8.0, half_h * 0.12)
        car_w = far_width + self.progress * (near_width - far_width)
        car_h = max(18.0, car_w * 0.74)
        end_y = height + car_h
        y = vanish_y + self.progress * (end_y - vanish_y)
        lane_y = min(max(y, float(horizon)), float(road_top))
        half = _road_half_at_y(lane_y, horizon, road_top, half_h, half_n)
        #1 Midten af banen er halvvejs mellem midtstriben og kanten. 0 er venstre, 1 er højre.
        side = -1.0 if self.lane == 0 else 1.0
        center_x = int(round(vx + side * half / 2.0))
        top = int(round(y - car_h / 2.0))
        return center_x, top, max(8, int(round(car_w)))

    def bounds(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """#1 Kroppen, som forsiden tegnes i."""
        center_x, top, car_w = self.pose(width, height)
        car_h = max(18, int(car_w * 0.74))
        x1 = center_x - car_w // 2
        return x1, top, x1 + car_w, top + car_h

    def overlaps(self, box: Tuple[int, int, int, int], width: int, height: int) -> bool:
        return _rects_overlap(self.bounds(width, height), box)

    def strike(self, box: Tuple[int, int, int, int], width: int, height: int) -> bool:
        """#1 True kun første gang bilen overlapper brugeren."""
        if self.struck or not self.overlaps(box, width, height):
            return False
        self.struck = True
        return True

    def draw(self, canvas: np.ndarray, t: float) -> None:
        del t
        height, width = canvas.shape[:2]
        center_x, top, car_w = self.pose(width, height)
        self._paint_front(canvas, center_x, top, car_w)


def _rects_overlap(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


class OncomingTraffic:
    """#1 Højst én modkørende bil. Banerne skiftes, så de aldrig kører samtidig."""

    def __init__(self) -> None:
        self._color_i = 0
        self._lane = 0
        self.car = OncomingLaneCar(ONCOMING_COLORS[0], self._lane)
        self.active = True
        self._wait_until: Optional[float] = None
        self._last_now: Optional[float] = None

    def tick(self, now: float) -> None:
        if self._last_now is None:
            self._last_now = now
        dt = max(0.0, min(0.1, now - self._last_now))
        self._last_now = now
        if not self.active:
            if self._wait_until is not None and now >= self._wait_until:
                self._start_next()
            return
        self.car.progress += dt / ONCOMING_APPROACH_SECONDS
        if self.car.progress >= 1.0:
            self.active = False
            self._wait_until = now + ONCOMING_GAP_SECONDS

    def _start_next(self) -> None:
        self._color_i = (self._color_i + 1) % len(ONCOMING_COLORS)
        previous = self.car.color
        #1 Næste bil kører i den anden bane. Den forrige er allerede væk.
        self._lane = 1 - self._lane
        self.car = OncomingLaneCar(ONCOMING_COLORS[self._color_i], self._lane)
        if self.car.color == previous:
            self._color_i = (self._color_i + 1) % len(ONCOMING_COLORS)
            self.car = OncomingLaneCar(ONCOMING_COLORS[self._color_i], self._lane)
        self.active = True
        self._wait_until = None

    def draw(self, canvas: np.ndarray) -> None:
        if self.active:
            self.car.draw(canvas, 0.0)


#1 Tegningen ved scale 1: ører til fødder, 47 px høj og 60 px bred.
KITTEN_BODY_HEIGHT = 47
KITTEN_BODY_WIDTH = 60


class StreetKitten(StreetActor):
    """Killing der løber ind fra venstre eller højre kant — original tegning."""

    FUR = ((40, 145, 230), (170, 170, 175), (210, 235, 250), (55, 70, 90))
    #1 Fem sekunder på uret, ikke på antal billeder.
    VISIBLE_SECONDS = 5.0

    def __init__(
        self,
        phase: float,
        speed: float,
        lane: int,
        kitten_id: int,
        fur: Color,
        *,
        enter_from_left: bool = True,
        depth: float = 1.0,
        shown_at: Optional[float] = None,
    ) -> None:
        super().__init__(phase, speed, lane)
        self.kitten_id = kitten_id
        self.fur = fur
        self.enter_from_left = enter_from_left
        #1 1 er forgrunden. 0 er horisonten.
        self.depth = float(np.clip(depth, 0.0, 1.0))
        self.shown_at = time.monotonic() if shown_at is None else shown_at
        self.name = f"Killing {kitten_id}"
        self._entry_delay = phase * 48.0

    def expired(self, now: float) -> bool:
        """#1 Efter 5 sekunder er killingen væk."""
        return now - self.shown_at >= self.VISIBLE_SECONDS

    def x_on(self, t: float, width: int, margin: int = 90) -> int:
        elapsed = max(0.0, t - self.start_t - self._entry_delay)
        travel = elapsed * (self.speed * 3.5)
        if self.enter_from_left:
            return int(-margin + travel)
        return int(width + margin - travel)

    def wrapped(self, t: float, width: int, height: int) -> bool:
        """True når killingen er løbet helt over og forsvinder på modsatte side."""
        x = self._road_x(t, width, height)
        prev = self._prev_x
        self._prev_x = x
        if prev is None:
            return False
        _horizon, _road_top, vx, _vy, half_h, half_n = _road_layout(height, width)
        half = _road_half_at_y(float(self._feet_y(height, width)), _horizon, _road_top, half_h, half_n)
        if self.enter_from_left:
            far = vx + half
            return prev < far and x >= far
        near = vx - half
        return prev > near and x <= near

    def _feet_y(self, height: int, width: int) -> int:
        horizon, road_top, _vx, _vy, _half_h, _half_n = _road_layout(height, width)
        return int(round(horizon + self.depth * (road_top - horizon)))

    def _body_scale(self, height: int, width: int) -> float:
        """#1 Forgrunden er 1/5 af billedhøjden. Tættere på horisonten følger vejens bredde."""
        horizon, road_top, _vx, _vy, half_h, half_n = _road_layout(height, width)
        half = _road_half_at_y(float(self._feet_y(height, width)), horizon, road_top, half_h, half_n)
        foreground = (height / 5.0) / KITTEN_BODY_HEIGHT
        return foreground * (half / max(half_n, 1.0))

    def _road_x(self, t: float, width: int, height: int) -> int:
        elapsed = max(0.0, t - self.start_t - self._entry_delay)
        travel = elapsed * (self.speed * 3.5)
        horizon, road_top, vx, _vy, half_h, half_n = _road_layout(height, width)
        half = _road_half_at_y(float(self._feet_y(height, width)), horizon, road_top, half_h, half_n)
        margin = 90
        if self.enter_from_left:
            return int(vx - half - margin + travel)
        return int(vx + half + margin - travel)

    def _depth(self, t: float, width: int, height: int) -> Tuple[int, int, float]:
        return self._road_x(t, width, height), self._feet_y(height, width), self._body_scale(height, width)

    def draw(self, canvas: np.ndarray, t: float) -> None:
        h, w = canvas.shape[:2]
        feet_x, feet_y, scale = self._depth(t, w, h)
        flip = not self.enter_from_left
        draw_kitten_side(canvas, feet_x, feet_y, self.fur, t, flip=flip, scale=scale)

    def hitbox(self, t: float, width: int, height: int) -> Tuple[int, int, int]:
        x, feet_y, scale = self._depth(t, width, height)
        #1 Træffet dækker den tegnede krop, ikke et fast punkt over fødderne.
        y = int(round(feet_y - KITTEN_BODY_HEIGHT * scale / 2.0))
        radius = max(8, int(round(max(KITTEN_BODY_WIDTH, KITTEN_BODY_HEIGHT) * scale / 2.0)))
        return x, y, radius


def _draw_kitten_side_core(
    canvas: np.ndarray,
    feet_x: int,
    feet_y: int,
    fur: Color,
    t: float,
) -> None:
    walk = 5 if int(t / 5) % 2 == 0 else -5
    bx, by = feet_x, feet_y - 16
    cv2.ellipse(canvas, (bx - 20, by - 6), (11, 5), -50, 0, 360, fur, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (bx, by), (18, 11), 0, 0, 360, fur, -1, cv2.LINE_AA)
    hx, hy = bx + 16, by - 10
    cv2.circle(canvas, (hx, hy), 12, fur, -1, cv2.LINE_AA)
    cv2.fillConvexPoly(canvas, np.array([[hx - 7, hy - 6], [hx - 3, hy - 18], [hx + 1, hy - 6]]), fur)
    cv2.fillConvexPoly(canvas, np.array([[hx + 2, hy - 6], [hx + 7, hy - 18], [hx + 10, hy - 6]]), fur)
    #1 Ingen hjelm.
    cv2.circle(canvas, (hx + 5, hy - 1), 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (hx + 6, hy - 1), 2, (25, 25, 25), -1, cv2.LINE_AA)
    cv2.circle(canvas, (hx + 8, hy + 4), 2, (90, 140, 200), -1, cv2.LINE_AA)
    cv2.line(canvas, (bx - 8, by + 6), (bx - 8 - walk, feet_y), fur, 3, cv2.LINE_AA)
    cv2.line(canvas, (bx - 2, by + 6), (bx - 2 + walk, feet_y), fur, 3, cv2.LINE_AA)
    cv2.line(canvas, (bx + 8, by + 6), (bx + 8 + walk, feet_y), fur, 3, cv2.LINE_AA)
    cv2.line(canvas, (bx + 13, by + 6), (bx + 13 - walk, feet_y), fur, 3, cv2.LINE_AA)


def draw_kitten_side(
    canvas: np.ndarray,
    feet_x: int,
    feet_y: int,
    fur: Color,
    t: float,
    *,
    flip: bool = False,
    scale: float = 1.0,
) -> None:
    pad_w, pad_h = 110, 100
    patch = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    local_x, local_y = pad_w // 2, pad_h - 8
    _draw_kitten_side_core(patch, local_x, local_y, fur, t)
    if flip:
        patch = cv2.flip(patch, 1)
    if abs(scale - 1.0) > 0.04:
        nw = max(24, int(pad_w * scale))
        nh = max(24, int(pad_h * scale))
        patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
        pad_w, pad_h = nw, nh
    ox = feet_x - pad_w // 2
    oy = feet_y - pad_h + 8
    h, w = canvas.shape[:2]
    x1, y1 = max(0, ox), max(0, oy)
    x2, y2 = min(w, ox + pad_w), min(h, oy + pad_h)
    if x2 <= x1 or y2 <= y1:
        return
    sx1, sy1 = x1 - ox, y1 - oy
    src = patch[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)]
    roi = canvas[y1:y2, x1:x2]
    mask = np.any(src > 10, axis=2)
    roi[mask] = src[mask]


class CityLayer:
    """Et lag i by-baggrunden."""

    def draw(self, canvas: np.ndarray, t: float) -> None:
        raise NotImplementedError


class SkyLayer(CityLayer):
    def draw(self, canvas: np.ndarray, t: float) -> None:
        h, w = canvas.shape[:2]
        for y in range(h):
            u = y / max(h - 1, 1)
            canvas[y, :] = (
                int(255 * (1 - u) + 245 * u),
                int(195 * (1 - u) + 215 * u),
                int(95 * (1 - u) + 175 * u),
            )
        sun = (int(w * 0.82), int(h * 0.16))
        cv2.circle(canvas, sun, int(h * 0.07), (80, 230, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, sun, int(h * 0.09), (140, 240, 255), 2, cv2.LINE_AA)


class CloudLayer(CityLayer):
    def draw(self, canvas: np.ndarray, t: float) -> None:
        h, w = canvas.shape[:2]
        clouds = ((0.12, 0.12, 70), (0.45, 0.08, 90), (0.78, 0.16, 60), (0.30, 0.20, 50))
        for i, (nx, ny, rad) in enumerate(clouds):
            x = int((nx * w + t * (0.55 + 0.18 * i)) % (w + 160) - 80)
            y = int(ny * h)
            color = (255, 250, 245)
            cv2.ellipse(canvas, (x, y), (rad, rad // 2), 0, 0, 360, color, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (x + rad // 2, y + 4), (rad // 2 + 8, rad // 3), 0, 0, 360, color, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (x - rad // 2, y + 6), (rad // 2, rad // 3), 0, 0, 360, color, -1, cv2.LINE_AA)


class RoadLayer(CityLayer):
    """#1 Kun græs under perspektiv-vejen — ingen flad horisontal asfalt."""

    def draw(self, canvas: np.ndarray, t: float) -> None:
        del t
        h, w = canvas.shape[:2]
        _horizon, road_top, _vx, _vy, _half_h, _half_n = _road_layout(h, w)
        cv2.rectangle(canvas, (0, road_top), (w, h), (58, 105, 62), -1)


class PerspectiveRunLayer(CityLayer):
    """Perspektiv-gade der streamer mod kameraet — løb gennem byen."""

    def draw(self, canvas: np.ndarray, t: float) -> None:
        h, w = canvas.shape[:2]
        _horizon, road_top, vx, vy, half_h, half_n = _road_layout(h, w)
        asphalt = (62, 66, 78)
        xl_n, xr_n = int(vx - half_n), int(vx + half_n)
        xl_h, xr_h = int(vx - half_h), int(vx + half_h)
        left = np.array([[0, road_top], [xl_n, road_top], [xl_h, vy], [0, vy + 4]], np.int32)
        right = np.array([[w, road_top], [xr_n, road_top], [xr_h, vy], [w, vy + 4]], np.int32)
        center = np.array([[xl_n, road_top], [xr_n, road_top], [xr_h, vy], [xl_h, vy]], np.int32)
        cv2.fillConvexPoly(canvas, left, (88, 118, 72))
        cv2.fillConvexPoly(canvas, right, (88, 118, 72))
        cv2.fillConvexPoly(canvas, center, asphalt)
        cv2.line(canvas, (xl_n, road_top), (xl_h, vy), (98, 100, 112), 2, cv2.LINE_AA)
        cv2.line(canvas, (xr_n, road_top), (xr_h, vy), (98, 100, 112), 2, cv2.LINE_AA)
        #1 Zoomet: bredere, længere midtstriber med tydeligt mellemrum.
        phase = (t * 0.14) % 1.0
        bands = 6
        span = float(road_top - vy - 6)
        slot = span / bands
        dash_frac = 0.62
        for i in range(bands):
            u = (i + phase) / bands
            if u >= 0.96:
                continue
            y0 = int(vy + u * span)
            y1 = int(min(road_top - 2, y0 + slot * dash_frac))
            if y1 <= y0 + 3:
                continue
            thick = max(4, int(4 + u * 11))
            cv2.line(canvas, (vx, y0), (vx, y1), (220, 222, 235), thick, cv2.LINE_AA)


class HillLayer(CityLayer):
    """#1 Lysegrønne bakketoppe kun på horisonten. De står stille."""

    HILL = (168, 228, 186)
    HILL_SHADE = (148, 210, 164)

    def draw(self, canvas: np.ndarray, t: float) -> None:
        del t
        h, w = canvas.shape[:2]
        horizon, _road_top, _vx, _vy, _half_h, _half_near = _road_layout(h, w)
        #1 Fast række på horisonten. Ingen dybde og intet tidsled, så de ikke kører med vejen.
        specs = (
            (0.06, 0.11, self.HILL),
            (0.20, 0.15, self.HILL_SHADE),
            (0.36, 0.10, self.HILL),
            (0.52, 0.13, self.HILL_SHADE),
            (0.68, 0.16, self.HILL),
            (0.84, 0.12, self.HILL_SHADE),
            (0.98, 0.10, self.HILL),
        )
        base_y = horizon + max(6, int(h * 0.02))
        for nx, nrx, color in specs:
            rx = max(36, int(nrx * w))
            ry = max(18, int(rx * 0.42))
            self._hill(canvas, int(nx * w), base_y, rx, ry, color)

    @staticmethod
    def _hill(canvas: np.ndarray, cx: int, base_y: int, rx: int, ry: int, color: Color) -> None:
        # OpenCV måler med uret: 180→360 er den øverste halvbue.
        cv2.ellipse(canvas, (cx, base_y), (rx, ry), 0, 180, 360, color, -1, cv2.LINE_AA)


class AdventureCityBackdrop:
    """#1 Cartoon-landskab: himmel, bakker og perspektiv-vej — uden bygninger."""

    def __init__(self) -> None:
        self.t = 0.0
        self._sky = SkyLayer()
        self._clouds = CloudLayer()
        self._run = PerspectiveRunLayer()
        self._hills = HillLayer()
        self._road = RoadLayer()
        #1 Kun én modkørende bil synlig ad gangen.
        self.traffic = OncomingTraffic()
        self.kittens: List[StreetKitten] = []
        self.reset_kittens()

    def reset_kittens(self) -> None:
        #1 Kun én killing. Den første står i forgrunden.
        self.kittens = [
            StreetKitten(
                phase=0.0,
                speed=1.8,
                lane=0,
                kitten_id=1,
                fur=StreetKitten.FUR[0],
                enter_from_left=True,
                depth=1.0,
            )
        ]

    def tick(self) -> None:
        self.t += 1.0
        #1 Bilen følger sekunder, ikke hvor hurtigt billedet tegnes.
        self.traffic.tick(time.monotonic())

    def render(self, width: int, height: int) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._sky.draw(canvas, self.t)
        self._clouds.draw(canvas, self.t)
        #1 Bakker bag vejen, så kun toppen ses over horisonten.
        self._hills.draw(canvas, self.t)
        self._run.draw(canvas, self.t)
        self._road.draw(canvas, self.t)
        self.traffic.draw(canvas)
        for kitten in self.kittens:
            kitten.draw(canvas, self.t)
        return canvas
