"""Webcam-AR: brugere flyver imod spilleren i live-kameraet."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gamConsole.opencv_ui import run_window_loop

HAAR_FACE = "haarcascade_frontalface_default.xml"


class AREntity:
    """Objekt i AR-rum foran webcam: z=1 er langt væk, z→0 er tæt på spilleren."""

    def __init__(self, x_n: float, y_n: float, z: float, speed: float) -> None:
        self.x_n = x_n
        self.y_n = y_n
        self.z = z
        self.speed = speed

    def tick(self) -> None:
        self.z -= self.speed

    @property
    def reached_player(self) -> bool:
        return self.z <= 0.10

    def project(self, frame_w: int, frame_h: int) -> Tuple[int, int, int]:
        z = max(self.z, 0.06)
        px = int(frame_w * 0.5 + self.x_n * (frame_w * 0.42) / z)
        py = int(frame_h * 0.42 + self.y_n * (frame_h * 0.30) / z)
        radius = int(min(frame_w, frame_h) * 0.11 / z)
        return px, py, max(6, radius)

    def draw(self, frame: np.ndarray) -> None:
        raise NotImplementedError


class FlyingUser(AREntity):
    """En bruger der flyver imod webcam-spilleren."""

    def __init__(
        self,
        x_n: float,
        y_n: float,
        z: float,
        speed: float,
        user_id: int,
        color: Tuple[int, int, int],
        face: Optional[np.ndarray],
    ) -> None:
        super().__init__(x_n, y_n, z, speed)
        self.user_id = user_id
        self.color = color
        self.face = face
        self.name = f"User {user_id}"

    def draw(self, frame: np.ndarray) -> None:
        px, py, radius = self.project(frame.shape[1], frame.shape[0])
        sprite = self.face if self.face is not None else make_user_avatar(self.color)
        blit_circle(frame, sprite, px, py, radius)
        cv2.circle(frame, (px, py), radius, self.color, 2, cv2.LINE_AA)
        label_y = min(frame.shape[0] - 8, py + radius + 18)
        cv2.putText(
            frame,
            self.name,
            (max(8, px - radius), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


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


class ThrownBurger(AREntity):
    """Burger kastet af spilleren, flyver væk fra kameraet imod brugerne."""

    def __init__(
        self,
        x_n: float,
        y_n: float,
        z: float,
        speed: float,
        vx: float,
        vy: float,
    ) -> None:
        super().__init__(x_n, y_n, z, speed)
        self.vx = vx
        self.vy = vy

    @classmethod
    def from_throw(cls, event: ThrowEvent, frame_w: int, frame_h: int) -> ThrownBurger:
        z = 0.16
        x_n = (event.x_px - frame_w * 0.5) * z / (frame_w * 0.42)
        y_n = (event.y_px - frame_h * 0.42) * z / max(frame_h * 0.30, 1.0)
        return cls(
            x_n=x_n,
            y_n=y_n,
            z=z,
            speed=0.038,
            vx=event.dx / frame_w * 0.10,
            vy=event.dy / frame_h * 0.10,
        )

    def tick(self) -> None:
        self.z += self.speed
        self.x_n += self.vx
        self.y_n += self.vy

    @property
    def missed(self) -> bool:
        return self.z >= 1.15

    def project(self, frame_w: int, frame_h: int) -> Tuple[int, int, int]:
        z = max(self.z, 0.06)
        px = int(frame_w * 0.5 + self.x_n * (frame_w * 0.42) / z)
        py = int(frame_h * 0.42 + self.y_n * (frame_h * 0.30) / z)
        radius = int(min(frame_w, frame_h) * 0.055 / z)
        return px, py, max(8, radius)

    def hits(self, user: FlyingUser, frame_w: int, frame_h: int) -> bool:
        bx, by, br = self.project(frame_w, frame_h)
        ux, uy, ur = user.project(frame_w, frame_h)
        return math.hypot(bx - ux, by - uy) < (br + ur) * 0.72

    def draw(self, frame: np.ndarray) -> None:
        px, py, radius = self.project(frame.shape[1], frame.shape[0])
        blit_circle(frame, burger_sprite(), px, py, radius)
        cv2.circle(frame, (px, py), radius, (40, 90, 200), 2, cv2.LINE_AA)


class FaceBank:
    """Gemmer ansigter fra webcam, så flyvende brugere ligner rigtige personer."""

    def __init__(self) -> None:
        path = getattr(cv2, "data", None)
        cascade_path = f"{path.haarcascades}{HAAR_FACE}" if path is not None else HAAR_FACE
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self._faces: List[np.ndarray] = []
        self._frame_i = 0

    def sample(self, frame: np.ndarray) -> None:
        self._frame_i += 1
        if self._frame_i % 8 != 0:
            return
        if self._cascade.empty():
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self._cascade.detectMultiScale(gray, 1.2, 4, minSize=(48, 48))
        for x, y, w, h in boxes:
            pad = int(0.12 * max(w, h))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            self._faces.append(crop.copy())
        self._faces = self._faces[-6:]

    def pick(self, rng: np.random.Generator) -> Optional[np.ndarray]:
        if not self._faces:
            return None
        return self._faces[int(rng.integers(0, len(self._faces)))].copy()


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
    """Live webcam som AR-verden: brugere flyver imod spilleren."""

    window_name = "gamConsole: flyvende brugere"

    def __init__(self, camera: int) -> None:
        super().__init__(camera)
        self.rng = np.random.default_rng()
        self.faces = FaceBank()
        self.users: List[FlyingUser] = []
        self.burgers: List[ThrownBurger] = []
        self.throw_detector = ArmThrowDetector()
        self.prev_gray: Optional[np.ndarray] = None
        self.next_user_id = 1
        self.score = 0
        self.lives = 5
        self.spawn_in = 20
        self.hit_flashes: List[Tuple[int, int, int, int]] = []
        self.game_over = False

    def on_key(self, key: int) -> None:
        if self.game_over and key in (ord("r"), ord("R")):
            self._reset()

    def _reset(self) -> None:
        self.users = []
        self.burgers = []
        self.throw_detector.reset()
        self.score = 0
        self.lives = 5
        self.spawn_in = 20
        self.hit_flashes = []
        self.game_over = False
        self.next_user_id = 1

    def on_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, 1)
        if self.game_over:
            return self._draw_game_over(frame)

        self.faces.sample(frame)
        mask, self.prev_gray = motion_mask(frame, self.prev_gray)
        throw = self.throw_detector.detect(mask)
        if throw is not None and len(self.burgers) < 8:
            burger = ThrownBurger.from_throw(throw, frame.shape[1], frame.shape[0])
            self.burgers.append(burger)
            self.hit_flashes.append((throw.x_px, throw.y_px, 18, 8))

        self.spawn_in -= 1
        if self.spawn_in <= 0 and len(self.users) < 8:
            self.users.append(self._spawn_user())
            self.spawn_in = max(18, 48 - self.score // 40)

        h, w = frame.shape[:2]
        for burger in self.burgers:
            burger.tick()
        self.burgers = [b for b in self.burgers if not b.missed]

        alive_users: List[FlyingUser] = []
        for user in self.users:
            user.tick()
            px, py, radius = user.project(w, h)
            if user.reached_player:
                self.lives -= 1
                self.hit_flashes.append((px, py, radius, 10))
                if self.lives <= 0:
                    self.game_over = True
                continue
            hit_by = next((b for b in self.burgers if b.hits(user, w, h)), None)
            if hit_by is not None:
                self.score += 10 + int(40 * (1.0 - user.z))
                self.hit_flashes.append((px, py, radius, 12))
                self.burgers = [b for b in self.burgers if b is not hit_by]
                continue
            alive_users.append(user)
        self.users = alive_users

        drawables: List[AREntity] = list(self.users) + list(self.burgers)
        drawables.sort(key=lambda e: e.z, reverse=True)
        for entity in drawables:
            entity.draw(frame)
        self._draw_flashes(frame)
        self._draw_hud(frame)
        return frame

    def _spawn_user(self) -> FlyingUser:
        color = tuple(int(v) for v in self.rng.integers(70, 255, size=3))
        user = FlyingUser(
            x_n=float(self.rng.uniform(-0.55, 0.55)),
            y_n=float(self.rng.uniform(-0.35, 0.45)),
            z=1.0,
            speed=float(self.rng.uniform(0.007, 0.016)),
            user_id=self.next_user_id,
            color=color,
            face=self.faces.pick(self.rng),
        )
        self.next_user_id += 1
        return user

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
            "AR  —  flyvende brugere",
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
            "Kast med armen for at kaste burgere",
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


def make_user_avatar(color: Tuple[int, int, int]) -> np.ndarray:
    img = np.full((128, 128, 3), (28, 28, 40), dtype=np.uint8)
    cv2.circle(img, (64, 46), 28, color, -1, cv2.LINE_AA)
    cv2.circle(img, (54, 42), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (74, 42), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (54, 43), 2, (25, 25, 25), -1, cv2.LINE_AA)
    cv2.circle(img, (74, 43), 2, (25, 25, 25), -1, cv2.LINE_AA)
    cv2.ellipse(img, (64, 54), (10, 6), 0, 15, 165, (40, 40, 50), 2, cv2.LINE_AA)
    cv2.ellipse(img, (64, 112), (38, 30), 0, 180, 360, color, -1, cv2.LINE_AA)
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
    if prev is None:
        return np.zeros_like(gray), gray
    diff = cv2.absdiff(gray, prev)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return cv2.dilate(mask, None, iterations=2), gray


def main() -> None:
    parser = argparse.ArgumentParser(description="gamConsole AR: flyvende brugere")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    FlyingUsersARGame(args.camera).run()


if __name__ == "__main__":
    main()
