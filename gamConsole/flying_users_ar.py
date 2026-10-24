"""Webcam-AR: kast burgere efter flyvende killinger i en cartoon-by."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gamConsole.adventure_city import AdventureCityBackdrop, StreetKitten, _road_layout
from gamConsole.leader_back_view import ChaseInCar, _cut_white_background
from gamConsole.drum_track import DrumTrack
from gamConsole.game_audio import AUDIO_DIR, game_audio
from gamConsole.opencv_ui import run_window_loop
from gamConsole.ar_player_view import ARPlayerView
from gamConsole.pose_body_overlay import L_WRIST, R_WRIST, BodyPose
from gamConsole.round_score import HighScoreList, RoundScore, RoundScoreGame, highscore_file

#1 Gurli Gris bruger allerede −1 som minuspoint. Der er ikke nævnt et andet tal.
CAR_HIT_POINTS = -1
_HIT_IMAGE = Path(__file__).resolve().parents[1] / "assets" / "user_hit.png"
_HIT_WAV = AUDIO_DIR / "car.wav"
_KITTEN_SPLASH = Path(__file__).resolve().parents[1] / "assets" / "kitten_burger_splash.png"
#21 Dirty Rock af Kulakovka, Pixabay 614878. Filen er målt med afinfo:
# 81,215 sekunder, 256 kbps, 44,1 kHz, stereo.
SONG_PATH = Path(__file__).resolve().parents[1] / "assets" / "humdingers" / "kulakovka-dirty-rock-614878.mp3"


@dataclass
class ThrowEvent:
    x_px: int
    y_px: int
    dx: float
    dy: float


#1 Fem billeder. Retningen er snittet af de fire skridt imellem dem.
_WRIST_FRAMES = 5
#1 missed() fjerner burgeren ved age > 55. Den skal nå linjen før det.
_KITTEN_ARRIVE_FRAMES = 50


def kitten_cross_y(frame_h: int, frame_w: int) -> float:
    """#1 Midten af forgrundens killing. Den er 1/5 af billedhøjden."""
    _horizon, road_top, _vx, _vy, _half_h, _half_n = _road_layout(frame_h, frame_w)
    return float(road_top) - frame_h / 10.0


class WristThrowDetector:
    """#1 Slipper burgeren fra håndleddet. Farten er snittet af de sidste billeder."""

    def __init__(self) -> None:
        self._wrists: List[List[Optional[Tuple[int, int]]]] = [[], []]
        self.cooldown = 0

    def reset(self) -> None:
        self._wrists = [[], []]
        self.cooldown = 0

    def detect(self, pose: Optional[BodyPose]) -> Optional[ThrowEvent]:
        self._store(pose)
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        if pose is None:
            return None
        best: Optional[ThrowEvent] = None
        best_travel = 0.0
        limit = 0.055 * float(min(pose.width, pose.height))
        for track in self._wrists:
            event = self._throw(track, limit)
            if event is None:
                continue
            travel = math.hypot(event.dx, event.dy) * (_WRIST_FRAMES - 1)
            if travel > best_travel:
                best_travel = travel
                best = event
        if best is not None:
            self.cooldown = 14
        return best

    def _store(self, pose: Optional[BodyPose]) -> None:
        for index, joint in enumerate((L_WRIST, R_WRIST)):
            point = None if pose is None else pose.get(joint)
            self._wrists[index].append(point)
            extra = len(self._wrists[index]) - _WRIST_FRAMES
            if extra > 0:
                del self._wrists[index][:extra]

    @staticmethod
    def _throw(track: List[Optional[Tuple[int, int]]], limit: float) -> Optional[ThrowEvent]:
        if len(track) < _WRIST_FRAMES:
            return None
        points: List[Tuple[int, int]] = []
        for point in track:
            if point is None:
                return None
            points.append(point)
        steps = len(points) - 1
        dx = sum(points[i + 1][0] - points[i][0] for i in range(steps)) / steps
        dy = sum(points[i + 1][1] - points[i][1] for i in range(steps)) / steps
        if math.hypot(dx, dy) * steps < limit:
            return None
        last = points[-1]
        return ThrowEvent(last[0], last[1], float(dx), float(dy))


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
        """#1 Retningen er håndens. Farten skaleres kun, så linjen når killingernes højde."""
        vx = event.dx
        vy = event.dy
        cross_y = kitten_cross_y(frame_h, frame_w)
        y0 = float(event.y_px)
        if vy > 0.0 and y0 < cross_y:
            frames = (cross_y - y0) / vy
            if frames > _KITTEN_ARRIVE_FRAMES:
                scale = frames / _KITTEN_ARRIVE_FRAMES
                vx *= scale
                vy *= scale
        return cls(float(event.x_px), y0, vx, vy)

    def tick(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.age += 1

    def missed(self, frame_w: int, frame_h: int) -> bool:
        return self.y > frame_h * 0.84 or self.x < -50 or self.x > frame_w + 50 or self.age > 55

    def radius(self) -> int:
        #1 Dobbelt så stor som de gamle 22 px, og den skrumper med den samme andel.
        return max(20, int(44 - self.age * 0.44))

    def hits_xy(self, cx: int, cy: int, radius: int) -> bool:
        return math.hypot(self.x - cx, self.y - cy) < (self.radius() + radius) * 0.78

    def draw(self, frame: np.ndarray) -> None:
        blit_circle(frame, burger_sprite(), int(self.x), int(self.y), self.radius())
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius(), (40, 90, 200), 2, cv2.LINE_AA)


class CarHitSound:
    """#1 Kenney impactMetal_medium_000, når en modkørende bil rammer."""

    def __init__(self, path: Path = _HIT_WAV) -> None:
        self._path = path

    def _ensure(self) -> bool:
        return self._path.is_file()

    def play(self) -> None:
        if self._ensure():
            game_audio.car()


class CarHitFlash:
    """#1 Eksplosionen vises over brugeren, når bilen rammer."""

    def __init__(self, path: Path = _HIT_IMAGE) -> None:
        self._path = path
        self._bgr: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._loaded = False
        self.until = 0.0
        self.x = 0
        self.y = 0

    def clear(self) -> None:
        self.until = 0.0

    def visible(self) -> bool:
        return time.monotonic() < self.until

    def trigger(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.until = time.monotonic() + 1.05

    def draw(self, canvas: np.ndarray) -> None:
        if not self.visible():
            return
        self._ensure()
        if self._bgr is None or self._alpha is None:
            return
        height, width = canvas.shape[:2]
        target = max(80, int(min(width, height) * 0.42))
        sprite_h, sprite_w = self._bgr.shape[:2]
        scale = target / max(sprite_h, sprite_w, 1)
        draw_w = max(1, int(sprite_w * scale))
        draw_h = max(1, int(sprite_h * scale))
        sprite = cv2.resize(self._bgr, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(self._alpha, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
        ChaseInCar._blit(canvas, sprite, alpha, self.x - draw_w // 2, self.y - draw_h // 2)

    def _ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        image = cv2.imread(str(self._path), cv2.IMREAD_COLOR)
        if image is None:
            return
        self._bgr, self._alpha = _cut_white_background(image, spread=12)


class KittenBurgerSplash:
    """#1 Det gule splash, når en burger rammer en killing."""

    def __init__(self, path: Path = _KITTEN_SPLASH) -> None:
        self._path = path
        self._bgr: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._loaded = False
        self._pops: List[Tuple[int, int, int, float]] = []

    def clear(self) -> None:
        self._pops.clear()

    def trigger(self, x: int, y: int, radius: int, now: float) -> None:
        #1 Splashet dækker killingen. 0,7 sekund, målt på uret.
        side = max(72, radius * 3)
        self._pops.append((x, y, side, now + 0.7))

    def draw(self, canvas: np.ndarray, now: float) -> None:
        alive = [pop for pop in self._pops if pop[3] > now]
        self._pops = alive
        if not alive:
            return
        self._ensure()
        if self._bgr is None or self._alpha is None:
            return
        sprite_h, sprite_w = self._bgr.shape[:2]
        for x, y, side, _until in alive:
            scale = side / max(sprite_h, sprite_w, 1)
            draw_w = max(1, int(sprite_w * scale))
            draw_h = max(1, int(sprite_h * scale))
            sprite = cv2.resize(self._bgr, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(self._alpha, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
            ChaseInCar._blit(canvas, sprite, alpha, x - draw_w // 2, y - draw_h // 2)

    def _ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        image = cv2.imread(str(self._path), cv2.IMREAD_COLOR)
        if image is None:
            return
        self._bgr, self._alpha = _cut_white_background(image, spread=12)


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

        #9 #1 Spil igen skal kunne klikkes i begge spil.
        run_window_loop(self.window_name, self.cap, render, on_key=self.on_key, on_click=self.on_click)

    def on_key(self, key: int) -> None:
        return


class FlyingUsersARGame(RoundScoreGame, WebcamARGame):
    """Kast burgere efter Humdingers killinger i webcam-AR."""

    window_name = "gamConsole: Humdingers killinger"

    def __init__(self, camera: int, scores: Optional[HighScoreList] = None) -> None:
        super().__init__(camera)
        self.rng = np.random.default_rng()
        self.burgers: List[ThrownBurger] = []
        self.throw_detector = WristThrowDetector()
        self.city = AdventureCityBackdrop()
        self.player = ARPlayerView()
        #1 Næste killing efter den ene, der allerede står i forgrunden.
        self.next_kitten_id = 2
        self.score = 0
        self.lives = 5
        self.hit_flashes: List[Tuple[int, int, int, int]] = []
        self.game_over = False
        #1 Gurli Gris' score-tal. Boblerne derfra bliver ikke startet.
        self._points = None
        self._gain_clock: Optional[float] = None
        self.hit_flash = CarHitFlash()
        self.hit_sound = CarHitSound()
        self.kitten_splash = KittenBurgerSplash()
        #1 #9 Samme 3 minutter og highscore som Gurli, i Humdingers' egen fil.
        board = scores if scores is not None else HighScoreList(highscore_file("humdingers"))
        self.match = RoundScore(board)
        #21 Sangen kører kun, mens runden spilles.
        self.track = DrumTrack(SONG_PATH)

    def close(self) -> None:
        #21 Menuen og lukket vindue stopper sangen.
        self.track.stop()

    def _reset(self) -> None:
        self.burgers = []
        self.throw_detector.reset()
        self.city.reset_kittens()
        self.score = 0
        self.lives = 5
        self.hit_flashes = []
        self.game_over = False
        #1 Næste killing efter den ene, der allerede står i forgrunden.
        self.next_kitten_id = 2
        self._gain_clock = None
        self.hit_flash.clear()
        self.kitten_splash.clear()
        if self._points is not None:
            self._points.gains.clear()
            self._points.score = 0
        #1 #9 Nyt spil starter uret og resultatskærmen forfra.
        self.match.clear()

    @property
    def kittens(self) -> List[StreetKitten]:
        return self.city.kittens

    def play_again(self) -> None:
        #21 Spil igen starter sangen forfra.
        self.track.stop()
        self._reset()

    def _release_throw(self, pose: Optional[BodyPose], frame_w: int, frame_h: int) -> None:
        """#1 Burgeren starter i håndleddet og beholder den retning."""
        throw = self.throw_detector.detect(pose)
        if throw is None or len(self.burgers) >= 8:
            return
        self.burgers.append(ThrownBurger.from_throw(throw, frame_w, frame_h))
        self.hit_flashes.append((throw.x_px, throw.y_px, 18, 8))

    def on_frame(self, frame: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        frame = cv2.flip(frame, 1)
        moment = time.monotonic() if now is None else now
        #1 #9 3 minutter, eller livene er allerede brugt op.
        shown = self.match.begin(moment, self.score, self.play_again, lambda: frame)
        if shown is not None:
            self.game_over = True
            #21 Resultatskærmen er stille. Næste runde starter sangen forfra.
            self.track.stop()
            return shown
        if self.game_over:
            #21 Livene er brugt op. Resultatskærmen er stille.
            self.track.stop()
            return self.match.finish(frame, self.score, self.play_again)
        self.track.ensure_playing()

        h, w = frame.shape[:2]
        #1 Posen læses før kastet, så burgeren slippes fra håndleddet.
        self.player.refresh_pose(frame)
        self._release_throw(self.player.last_pose(), w, h)
        self.city.tick()
        #1 En modkørende bil i samme bane trækker point fra, én gang pr. bil.
        self._apply_car_hit(w, h)
        for burger in self.burgers:
            burger.tick()
        self.burgers = [b for b in self.burgers if not b.missed(w, h)]

        self._keep_one_kitten(w, h, moment)

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
            #1 Kenney confirmation_001, når en killing rammes.
            game_audio.points(10)
            self.kitten_splash.trigger(cx, cy, radius, moment)
            self.burgers = [b for b in self.burgers if b is not burger]
            self.city.kittens = [k for k in self.city.kittens if k is not hit]
            self.city.kittens.append(self._spawn_kitten(moment))

        scene = self.player.composite(frame, self.city.render(w, h))
        for burger in self.burgers:
            burger.draw(scene)
        self._draw_flashes(scene)
        self.kitten_splash.draw(scene, moment)
        self._tick_gains()
        self.hit_flash.draw(scene)
        for gain in self._score_view().gains:
            gain.draw(scene)
        self._draw_hud(scene)
        self.match.stamp(scene, moment)
        if self.lives <= 0:
            self.game_over = True
            #21 Livene er brugt op. Resultatskærmen er stille.
            self.track.stop()
            return self.match.finish(scene, self.score, self.play_again)
        return scene

    def _keep_one_kitten(self, w: int, h: int, now: float) -> None:
        """#1 Én killing i 5 sekunder. Tiden fjerner den uden at tage et liv."""
        kept: List[StreetKitten] = []
        for kitten in self.city.kittens:
            if kitten.expired(now):
                continue
            if kitten.wrapped(self.city.t, w, h):
                cx, cy, radius = kitten.hitbox(self.city.t, w, h)
                self.lives -= 1
                self.hit_flashes.append((cx, cy, radius, 10))
                if self.lives <= 0:
                    self.game_over = True
                continue
            kept.append(kitten)
        self.city.kittens = kept
        if not self.city.kittens and not self.game_over:
            self.city.kittens.append(self._spawn_kitten(now))

    def _spawn_kitten(self, now: float) -> StreetKitten:
        idx = self.next_kitten_id - 1
        enter_from_left = bool(self.rng.integers(0, 2))
        kitten = StreetKitten(
            phase=0.0,
            speed=float(self.rng.uniform(1.5, 2.3)),
            lane=idx % 2,
            kitten_id=self.next_kitten_id,
            fur=StreetKitten.FUR[idx % len(StreetKitten.FUR)],
            enter_from_left=enter_from_left,
            depth=float(self.rng.uniform(0.0, 1.0)),
            shown_at=now,
        )
        kitten.start_t = self.city.t
        self.next_kitten_id += 1
        return kitten

    def _score_view(self):
        """#1 Samme score-bjælke som Gurli Gris. Bobleregnen kører ikke med."""
        if self._points is None:
            from gamConsole.gurli_gris import BubbleRain

            self._points = BubbleRain()
        self._points.score = self.score
        return self._points

    def _tick_gains(self) -> None:
        view = self._score_view()
        now = time.monotonic()
        if self._gain_clock is None:
            self._gain_clock = now
        dt = max(0.0, min(0.1, now - self._gain_clock))
        self._gain_clock = now
        alive = []
        for gain in view.gains:
            gain.tick(dt)
            if gain.visible():
                alive.append(gain)
        view.gains = alive

    def _apply_car_hit(self, width: int, height: int) -> None:
        """#1 Overlap med den blå bil giver minuspoint, lyd og eksplosionen."""
        traffic = self.city.traffic
        if not traffic.active:
            return
        box = self.player.car_box(width, height)
        if not traffic.car.strike(box, width, height):
            return
        self.score += CAR_HIT_POINTS
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        from gamConsole.gurli_gris import PointGain

        self._score_view().gains.append(PointGain(float(cx), float(cy), CAR_HIT_POINTS))
        self.hit_flash.trigger(cx, cy)
        self.hit_sound.play()

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
            f"Liv {self.lives}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 230, 255),
            2,
            cv2.LINE_AA,
        )
        #1 Score står i toppen, tegnet med Gurli Gris' score.
        self._score_view().draw_score(frame)
        cv2.putText(
            frame,
            "Kast burgere",
            (12, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

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


def make_kitten_sprite(fur: Tuple[int, int, int]) -> np.ndarray:
    from gamConsole.adventure_city import draw_kitten_side

    img = np.full((80, 96, 3), (22, 22, 30), dtype=np.uint8)
    draw_kitten_side(img, 48, 72, fur, 0.0)
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
    game = FlyingUsersARGame(args.camera)
    try:
        game.run()
    finally:
        #21 Lukket vindue stopper sangen.
        game.close()


if __name__ == "__main__":
    main()
