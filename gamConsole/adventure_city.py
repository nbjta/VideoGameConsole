"""Bevægelig cartoon-by i Adventure City-stil.

Original tegning — ikke officiel PAW Patrol-kunst eller -figurer.
"""

from __future__ import annotations

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


class CityCar(StreetActor):
    """#1 Modkørende trafik i venstre bane (højre mod venstre)."""

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
        x0 = front_x - car_len
        body_h = max(12, int(14 + car_len * 0.06))
        cv2.rectangle(canvas, (x0, y), (front_x, y + body_h), self.color, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0 + 6, y - 8), (front_x - 8, y + 2), (240, 245, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x0 + 10, y + body_h), 5, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(canvas, (front_x - 10, y + body_h), 5, (30, 30, 30), -1, cv2.LINE_AA)

    def hitbox(self, t: float, width: int, height: int) -> Tuple[int, int, int]:
        x, y, car_len = self._lane_pose(t, width, height)
        return x - car_len // 2, y + 6, 20


class StreetKitten(StreetActor):
    """Killing der løber ind fra venstre eller højre kant — original tegning."""

    FUR = ((40, 145, 230), (170, 170, 175), (210, 235, 250), (55, 70, 90))
    HELMETS = ((40, 45, 230), (20, 210, 255), (230, 130, 40), (50, 175, 70))

    def __init__(
        self,
        phase: float,
        speed: float,
        lane: int,
        kitten_id: int,
        fur: Color,
        helmet: Color,
        *,
        enter_from_left: bool = True,
    ) -> None:
        super().__init__(phase, speed, lane)
        self.kitten_id = kitten_id
        self.fur = fur
        self.helmet = helmet
        self.enter_from_left = enter_from_left
        self.name = f"Killing {kitten_id}"
        self._entry_delay = phase * 48.0

    def x_on(self, t: float, width: int, margin: int = 90) -> int:
        elapsed = max(0.0, t - self.start_t - self._entry_delay)
        travel = elapsed * (self.speed * 3.5)
        if self.enter_from_left:
            return int(-margin + travel)
        return int(width + margin - travel)

    def wrapped(self, t: float, width: int) -> bool:
        """True når killingen er løbet helt over og forsvinder på modsatte side."""
        x = self.x_on(t, width)
        prev = self._prev_x
        self._prev_x = x
        if prev is None:
            return False
        if self.enter_from_left:
            return prev < width * 0.84 and x >= width * 0.84
        return prev > width * 0.16 and x <= width * 0.16

    def _depth(self, t: float, width: int, height: int) -> Tuple[int, int, float]:
        x = self.x_on(t, width)
        if self.enter_from_left:
            prog = (x + 90) / max(width * 0.58, 1)
        else:
            prog = (width + 90 - x) / max(width * 0.58, 1)
        prog = float(np.clip(prog, 0.0, 1.0))
        _hz, road_top, _vx, _vy, _hh, _hn = _road_layout(height, width)
        base_y = road_top - 6 - self.lane * 8
        feet_y = base_y - int(prog * 14)
        scale = 1.08 + 0.38 * prog
        return x, feet_y, scale

    def draw(self, canvas: np.ndarray, t: float) -> None:
        h, w = canvas.shape[:2]
        feet_x, feet_y, scale = self._depth(t, w, h)
        flip = not self.enter_from_left
        draw_kitten_side(canvas, feet_x, feet_y, self.fur, self.helmet, t, flip=flip, scale=scale)

    def hitbox(self, t: float, width: int, height: int) -> Tuple[int, int, int]:
        x, feet_y, scale = self._depth(t, width, height)
        y = feet_y - 18
        return x, y, max(18, int(26 * scale))


def _draw_kitten_side_core(
    canvas: np.ndarray,
    feet_x: int,
    feet_y: int,
    fur: Color,
    helmet: Color,
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
    cv2.ellipse(canvas, (hx, hy - 9), (13, 7), 0, 0, 360, helmet, -1, cv2.LINE_AA)
    shine = tuple(min(255, c + 45) for c in helmet)
    cv2.ellipse(canvas, (hx, hy - 12), (8, 4), 0, 0, 360, shine, -1, cv2.LINE_AA)
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
    helmet: Color,
    t: float,
    *,
    flip: bool = False,
    scale: float = 1.0,
) -> None:
    pad_w, pad_h = 110, 100
    patch = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    local_x, local_y = pad_w // 2, pad_h - 8
    _draw_kitten_side_core(patch, local_x, local_y, fur, helmet, t)
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
    """#1 Lysegrønne runde bakketoppe langs vejen — uden træer."""

    HILL = (168, 228, 186)
    HILL_SHADE = (148, 210, 164)

    def draw(self, canvas: np.ndarray, t: float) -> None:
        del t
        h, w = canvas.shape[:2]
        horizon, road_top, vx, _vy, half_h, half_near = _road_layout(h, w)
        specs = (
            (0.06, -1, 1.05),
            (0.28, -1, 1.35),
            (0.55, -1, 1.65),
            (0.82, -1, 1.95),
            (0.14, 1, 1.10),
            (0.40, 1, 1.45),
            (0.68, 1, 1.75),
            (0.90, 1, 2.05),
        )
        for depth, side, scale in specs:
            base_y = int(horizon + 8 + depth * (road_top - horizon - 18))
            half = _road_half_at_y(float(base_y), horizon, road_top, half_h, half_near)
            rx = max(32, int((34 + depth * 32) * scale))
            ry = max(16, int(rx * 0.48))
            #1 Hele bakken ligger i grøft/græs — indre kant rører ikke kørebanen.
            verge = 10 + int(depth * 16)
            if side < 0:
                inner_x = int(vx - half)
                cx = inner_x - verge - rx
                cx = max(rx + 2, cx)
                if cx + rx >= inner_x - 4:
                    continue
            else:
                inner_x = int(vx + half)
                cx = inner_x + verge + rx
                cx = min(w - rx - 2, cx)
                if cx - rx <= inner_x + 4:
                    continue
            color = self.HILL if side < 0 else self.HILL_SHADE
            self._hill(canvas, cx, base_y, rx, ry, color)

    @staticmethod
    def _hill(canvas: np.ndarray, cx: int, base_y: int, rx: int, ry: int, color: Color) -> None:
        # OpenCV måler med uret: 180→360 er den øverste halvbue.
        cv2.ellipse(canvas, (cx, base_y), (rx, ry), 0, 180, 360, color, -1, cv2.LINE_AA)


class StreetPropsLayer(CityLayer):
    """Lygtepæle langs vejkanten — følger vejens perspektiv."""

    def draw(self, canvas: np.ndarray, t: float) -> None:
        h, w = canvas.shape[:2]
        horizon, road_top, vx, _vy, half_h, half_near = _road_layout(h, w)
        #1 Færre pæle og større fase-skridt, så der er længere mellem dem.
        count = 3
        gap = 1.0 / count
        for side in (-1, 1):
            for i in range(count):
                phase = (t * 0.055 + i * gap) % 1.0
                foot_y = int(horizon + 18 + phase * (road_top - horizon - 10))
                half = _road_half_at_y(float(foot_y), horizon, road_top, half_h, half_near)
                margin = 10 + phase * 6
                side_x = int(vx + side * (half + margin))
                pole_h = int(16 + phase * 48)
                top_y = foot_y - pole_h
                if top_y < horizon:
                    continue
                cv2.line(canvas, (side_x, foot_y), (side_x, top_y), (82, 88, 98), max(2, int(2 + phase * 2)), cv2.LINE_AA)
                lamp_r = max(4, int(4 + phase * 5))
                cv2.circle(canvas, (side_x, top_y), lamp_r, (110, 200, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (side_x, top_y), max(2, lamp_r // 2), (240, 250, 255), -1, cv2.LINE_AA)


class AdventureCityBackdrop:
    """#1 Cartoon-landskab: himmel, bakker og perspektiv-vej — uden bygninger."""

    def __init__(self) -> None:
        self.t = 0.0
        self._sky = SkyLayer()
        self._clouds = CloudLayer()
        self._run = PerspectiveRunLayer()
        self._hills = HillLayer()
        self._props = StreetPropsLayer()
        self._road = RoadLayer()
        self.cars: List[CityCar] = [
            CityCar(0.0, 3.2, 0, (40, 90, 230), 46),
            CityCar(0.33, 3.6, 1, (40, 200, 90), 52),
            CityCar(0.62, 3.0, 0, (200, 80, 255), 48),
            CityCar(0.85, 3.8, 1, (30, 210, 230), 42),
        ]
        self.kittens: List[StreetKitten] = []
        self.reset_kittens()

    def reset_kittens(self) -> None:
        self.kittens = []
        for i in range(4):
            self.kittens.append(
                StreetKitten(
                    phase=0.08 + i * 0.16,
                    speed=1.55 + 0.22 * i,
                    lane=i % 2,
                    kitten_id=i + 1,
                    fur=StreetKitten.FUR[i % len(StreetKitten.FUR)],
                    helmet=StreetKitten.HELMETS[i % len(StreetKitten.HELMETS)],
                    enter_from_left=(i % 2 == 0),
                )
            )

    def tick(self) -> None:
        self.t += 1.0

    def render(self, width: int, height: int) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._sky.draw(canvas, self.t)
        self._clouds.draw(canvas, self.t)
        #1 Ingen parallax-bygninger eller facader — åbent landskab.
        self._run.draw(canvas, self.t)
        self._hills.draw(canvas, self.t)
        self._props.draw(canvas, self.t)
        self._road.draw(canvas, self.t)
        for car in self.cars:
            car.draw(canvas, self.t)
        for kitten in self.kittens:
            kitten.draw(canvas, self.t)
        return canvas
