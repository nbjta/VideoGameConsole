"""Webcam-AR: kast burgere efter flyvende killinger i en cartoon-by."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gamConsole.adventure_city import AdventureCityBackdrop, StreetKitten, _road_layout
from gamConsole.opencv_ui import run_window_loop
from gamConsole.overhead_branches import OverheadBranch, draw_branch_warning, reset_branches
from gamConsole.ar_player_view import ARPlayerView


@dataclass
class ThrowEvent:
    x_px: int
    y_px: int
    dx: float
    dy: float


class ArmThrowDetector:
    """Finder et armekast som hurtig bevægelse i venstre/højre arm-zone."""

    def __init__(self) -> None:
        self._prev: List[Optional[Tuple[int, int]]] = [None, None]
        self._prev_energy = [0.0, 0.0]
        self.cooldown = 0

    def reset(self) -> None:
        self._prev = [None, None]
        self._prev_energy = [0.0, 0.0]
        self.cooldown = 0

    def detect(self, mask: np.ndarray) -> Optional[ThrowEvent]:
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        h, w = mask.shape[:2]
        y0, y1 = int(0.05 * h), int(0.72 * h)
        zones = ((0, int(0.40 * w)), (int(0.60 * w), w))
        best: Optional[ThrowEvent] = None
        best_score = 0.0
        min_move = 0.055 * min(w, h)
        for i, (x0, x1) in enumerate(zones):
            center, energy = self._zone(mask, x0, x1, y0, y1)
            prev = self._prev[i]
            prev_e = self._prev_energy[i]
            self._prev[i] = center
            self._prev_energy[i] = energy
            if center is None or prev is None:
                continue
            dx = float(center[0] - prev[0])
            dy = float(center[1] - prev[1])
            dist = math.hypot(dx, dy)
            energy_spike = energy - prev_e
            throw = dist >= min_move or (energy >= 0.10 and energy_spike >= 0.06)
            if not throw:
                continue
            score = dist + energy_spike * min(w, h)
            if score > best_score:
                best_score = score
                best = ThrowEvent(center[0], center[1], dx, dy)
        if best is not None:
            self.cooldown = 14
        return best

    @staticmethod
    def _zone(
        mask: np.ndarray, x0: int, x1: int, y0: int, y1: int
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        roi = mask[y0:y1, x0:x1]
        if roi.size == 0:
            return None, 0.0
        energy = float(np.count_nonzero(roi)) / float(roi.size)
        moments = cv2.moments(roi)
        if moments["m00"] < max(600.0, 0.015 * roi.size):
            return None, energy
        cx = x0 + int(moments["m10"] / moments["m00"])
        cy = y0 + int(moments["m01"] / moments["m00"])
        return (cx, cy), energy


class ThrownBurger:
    """Burger der flyver hen mod gaden, hvor killingerne løber."""

    def __init__(self, x: float, y: float, vx: float, vy: float) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.age = 0

    @classmethod
    def from_throw(cls, event: ThrowEvent, frame_w: int, frame_h: int) -> ThrownBurger:
        _horizon, road_top, _vx, _vy, _hh, _hn = _road_layout(frame_h, frame_w)
        road_y = float(road_top)
        vx = event.dx * 0.28
        if abs(vx) < 5:
            vx = -9.0 if event.x_px > frame_w / 2 else 9.0
        vy = max(7.0, (road_y - event.y_px) / 16.0)
        return cls(float(event.x_px), float(event.y_px), vx, vy)

    def tick(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.age += 1

    def missed(self, frame_w: int, frame_h: int) -> bool:
        return self.y > frame_h * 0.84 or self.x < -50 or self.x > frame_w + 50 or self.age > 55

    def radius(self) -> int:
        return max(10, int(22 - self.age * 0.22))

    def hits_xy(self, cx: int, cy: int, radius: int) -> bool:
        return math.hypot(self.x - cx, self.y - cy) < (self.radius() + radius) * 0.78

    def draw(self, frame: np.ndarray) -> None:
        blit_circle(frame, burger_sprite(), int(self.x), int(self.y), self.radius())
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius(), (40, 90, 200), 2, cv2.LINE_AA)


class WebcamARGame:
    """Fælles løkke for webcam-baserede AR-spil."""

    window_name = "gamConsole: AR"

    def __init__(self, camera: int) -> None:
        self.camera = camera
        self.cap: Optional[cv2.VideoCapture] = None

    def on_frame(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def run(self) -> None:
        self.cap = cv2.VideoCapture(self.camera)
        if not self.cap.isOpened():
            print(f"Kunne ikke åbne kamera {self.camera}", file=sys.stderr)
            sys.exit(1)

        def render() -> Optional[np.ndarray]:
            assert self.cap is not None
            ok, frame = self.cap.read()
            if not ok:
                return None
            return self.on_frame(frame)

        run_window_loop(self.window_name, self.cap, render, on_key=self.on_key)

    def on_key(self, key: int) -> None:
        return


class FlyingUsersARGame(WebcamARGame):
    """Kast burgere efter Humdingers killinger i webcam-AR."""

    window_name = "gamConsole: Humdingers killinger"

    def __init__(self, camera: int) -> None:
        super().__init__(camera)
        self.rng = np.random.default_rng()
        self.burgers: List[ThrownBurger] = []
        self.throw_detector = ArmThrowDetector()
        self.city = AdventureCityBackdrop()
        self.player = ARPlayerView()
        self.prev_gray: Optional[np.ndarray] = None
        self.next_kitten_id = 5
        self.score = 0
        self.lives = 5
        self.hit_flashes: List[Tuple[int, int, int, int]] = []
        self.branches: List[OverheadBranch] = reset_branches(0)
        self.branch_spawn_cooldown = 90
        self.game_over = False

    def on_key(self, key: int) -> None:
        if self.game_over and key in (ord("r"), ord("R")):
            self._reset()

    def _reset(self) -> None:
        self.burgers = []
        self.throw_detector.reset()
        self.city.reset_kittens()
        self.score = 0
        self.lives = 5
        self.hit_flashes = []
        self.game_over = False
        self.next_kitten_id = 5
        self.branches = reset_branches(0)
        self.branch_spawn_cooldown = 90

    @property
    def kittens(self) -> List[StreetKitten]:
        return self.city.kittens

    def on_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, 1)
        if self.game_over:
            self.city.tick()
            scene = self.player.composite(frame, self.city.render(frame.shape[1], frame.shape[0]))
            return self._draw_game_over(scene)

        mask, self.prev_gray = motion_mask(frame, self.prev_gray)
        throw = self.throw_detector.detect(mask)
        if throw is not None and len(self.burgers) < 8:
            burger = ThrownBurger.from_throw(throw, frame.shape[1], frame.shape[0])
            self.burgers.append(burger)
            self.hit_flashes.append((throw.x_px, throw.y_px, 18, 8))

        h, w = frame.shape[:2]
        self.player.refresh_pose(frame)
        self.city.tick()
        self._tick_branches(frame, w, h)
        for burger in self.burgers:
            burger.tick()
        self.burgers = [b for b in self.burgers if not b.missed(w, h)]

        for kitten in list(self.city.kittens):
            if kitten.wrapped(self.city.t, w):
                cx, cy, radius = kitten.hitbox(self.city.t, w, h)
                self.lives -= 1
                self.hit_flashes.append((cx, cy, radius, 10))
                if self.lives <= 0:
                    self.game_over = True

        for burger in list(self.burgers):
            hit = None
            for kitten in self.city.kittens:
                cx, cy, radius = kitten.hitbox(self.city.t, w, h)
                if burger.hits_xy(cx, cy, radius):
                    hit = kitten
                    break
            if hit is None:
                continue
            cx, cy, radius = hit.hitbox(self.city.t, w, h)
            self.score += 10
            self.hit_flashes.append((cx, cy, radius, 12))
            self.burgers = [b for b in self.burgers if b is not burger]
            self.city.kittens = [k for k in self.city.kittens if k is not hit]
            self.city.kittens.append(self._spawn_kitten())

        scene = self.player.composite(frame, self.city.render(w, h))
        for burger in self.burgers:
            burger.draw(scene)
        self._draw_branches(scene)
        self._draw_flashes(scene)
        self._draw_hud(scene)
        return scene

    def _tick_branches(self, frame: np.ndarray, w: int, h: int) -> None:
        t = self.city.t
        for branch in list(self.branches):
            if branch.threatens_center(t, w) and not branch.hit:
                clearance = branch.clearance_y(h)
                if not self.player.is_ducked_under(frame, clearance):
                    branch.hit = True
                    self.lives -= 1
                    self.hit_flashes.append((w // 2, clearance + 20, 40, 14))
                    if self.lives <= 0:
                        self.game_over = True
        self.branches = [b for b in self.branches if not b.passed(t, w)]
        if self.branch_spawn_cooldown > 0:
            self.branch_spawn_cooldown -= 1
        elif len(self.branches) < 1:
            self.branches.append(self._spawn_branch())
            self.branch_spawn_cooldown = int(self.rng.integers(110, 190))

    def _spawn_branch(self) -> OverheadBranch:
        branch = OverheadBranch(
            phase=0.0,
            speed=0.0,
            enter_from_left=True,
            low_y_ratio=float(self.rng.uniform(0.20, 0.26)),
        )
        branch.start_t = self.city.t
        return branch

    def _draw_branches(self, scene: np.ndarray) -> None:
        t = self.city.t
        w = scene.shape[1]
        for branch in self.branches:
            if branch.threatens_center(t, w):
                draw_branch_warning(scene, branch.clearance_y(scene.shape[0]))
            branch.draw(scene, t)

    def _spawn_kitten(self) -> StreetKitten:
        idx = self.next_kitten_id - 1
        enter_from_left = bool(self.rng.integers(0, 2))
        kitten = StreetKitten(
            phase=0.0,
            speed=float(self.rng.uniform(1.5, 2.3)),
            lane=idx % 2,
            kitten_id=self.next_kitten_id,
            fur=StreetKitten.FUR[idx % len(StreetKitten.FUR)],
            helmet=StreetKitten.HELMETS[idx % len(StreetKitten.HELMETS)],
            enter_from_left=enter_from_left,
        )
        kitten.start_t = self.city.t
        self.next_kitten_id += 1
        return kitten

    def _draw_flashes(self, frame: np.ndarray) -> None:
        next_flashes = []
        for x, y, r, life in self.hit_flashes:
            cv2.circle(frame, (x, y), r + (12 - life) * 3, (80, 220, 255), 2, cv2.LINE_AA)
            if life > 1:
                next_flashes.append((x, y, r, life - 1))
        self.hit_flashes = next_flashes

    def _draw_hud(self, frame: np.ndarray) -> None:
        cv2.putText(
            frame,
            "Humdingers killinger",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Score {self.score}    Liv {self.lives}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 230, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Kast burgere - buk dig under grenen",
            (12, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    def _draw_game_over(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (20, 10, 30), -1)
        frame = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)
        cv2.putText(
            frame,
            "Game over",
            (40, frame.shape[0] // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Score {self.score}   —   R starter forfra   Q lukker",
            (40, frame.shape[0] // 2 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 230),
            2,
            cv2.LINE_AA,
        )
        return frame


def make_burger_sprite() -> np.ndarray:
    img = np.full((128, 128, 3), (20, 20, 28), dtype=np.uint8)
    cv2.ellipse(img, (64, 86), (46, 16), 0, 0, 360, (42, 92, 196), -1, cv2.LINE_AA)
    cv2.ellipse(img, (64, 74), (48, 14), 0, 0, 360, (36, 160, 48), -1, cv2.LINE_AA)
    cv2.ellipse(img, (64, 64), (50, 12), 0, 0, 360, (28, 48, 120), -1, cv2.LINE_AA)
    cv2.ellipse(img, (64, 48), (50, 28), 0, 180, 360, (40, 140, 220), -1, cv2.LINE_AA)
    cv2.ellipse(img, (64, 50), (50, 14), 0, 0, 360, (40, 140, 220), -1, cv2.LINE_AA)
    cv2.circle(img, (48, 44), 4, (20, 40, 200), -1, cv2.LINE_AA)
    cv2.circle(img, (72, 38), 4, (20, 40, 200), -1, cv2.LINE_AA)
    cv2.circle(img, (90, 48), 3, (20, 40, 200), -1, cv2.LINE_AA)
    return img


_BURGER_SPRITE: Optional[np.ndarray] = None


def burger_sprite() -> np.ndarray:
    global _BURGER_SPRITE
    if _BURGER_SPRITE is None:
        _BURGER_SPRITE = make_burger_sprite()
    return _BURGER_SPRITE


def make_kitten_sprite(fur: Tuple[int, int, int], helmet: Tuple[int, int, int]) -> np.ndarray:
    from gamConsole.adventure_city import draw_kitten_side

    img = np.full((80, 96, 3), (22, 22, 30), dtype=np.uint8)
    draw_kitten_side(img, 48, 72, fur, helmet, 0.0)
    return img


def blit_circle(dst: np.ndarray, src: np.ndarray, cx: int, cy: int, radius: int) -> None:
    if radius < 4:
        return
    size = max(8, radius * 2)
    face = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2 - 1, 255, -1)
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]
    x1 = cx - size // 2
    y1 = cy - size // 2
    x2, y2 = x1 + size, y1 + size
    h, w = dst.shape[:2]
    dx1, dy1 = max(0, x1), max(0, y1)
    dx2, dy2 = min(w, x2), min(h, y2)
    if dx1 >= dx2 or dy1 >= dy2:
        return
    sx1, sy1 = dx1 - x1, dy1 - y1
    roi = dst[dy1:dy2, dx1:dx2]
    patch = face[sy1 : sy1 + (dy2 - dy1), sx1 : sx1 + (dx2 - dx1)]
    m = mask_f[sy1 : sy1 + (dy2 - dy1), sx1 : sx1 + (dx2 - dx1)]
    dst[dy1:dy2, dx1:dx2] = (patch * m + roi.astype(np.float32) * (1.0 - m)).astype(np.uint8)


def motion_mask(frame: np.ndarray, prev: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev is None or prev.shape != gray.shape:
        return np.zeros_like(gray), gray
    diff = cv2.absdiff(gray, prev)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return cv2.dilate(mask, None, iterations=2), gray


def main() -> None:
    parser = argparse.ArgumentParser(description="gamConsole AR: burgere vs killinger")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    FlyingUsersARGame(args.camera).run()


if __name__ == "__main__":
    main()
