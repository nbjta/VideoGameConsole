#9
"""Gurli Gris i gamConsole.

Landskabet og figuren er de billeder, der ligger i assets/gurli_gris.
Armvinkel og punktsilhuet genbruges fra Humdingers killinger.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from gamConsole.flying_users_ar import WebcamARGame
from gamConsole.head_tilt import HeadTilt
from gamConsole.leader_back_view import ChaseInCar, _arm_angles, _cut_white_background, _rotate_bottom
from gamConsole.drum_track import DrumTrack
from gamConsole.game_audio import game_audio
from gamConsole.opencv_ui import draw_text, measure_text
from gamConsole.round_score import HighScoreList, RoundScore, RoundScoreGame, highscore_file
from gamConsole.pose_body_overlay import BodyPose, PoseBodyOverlay

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND_PATH = ROOT / "assets" / "gurli_gris" / "background.png"
FIGURE_PATH = ROOT / "assets" / "gurli_gris" / "gurli.png"
BRANCH_PATH = ROOT / "assets" / "gurli_gris" / "branch.png"
#19 Motivational af DailyVibe, Pixabay 614432. Filen er målt med afinfo:
# 178,834 sekunder, 256 kbps, 44,1 kHz, stereo.
SONG_PATH = ROOT / "assets" / "gurli_gris" / "dailyvibe-motivational-614432.mp3"
#9 Græsset i baggrundsbilledet starter ved række 101 af 148.
_FEET_Y = 0.90
_FIGURE_H = 0.62
#9 De nævnte tal, og de nærmeste i samme størrelse.
POINT_VALUES: Sequence[int] = (4, 2, 1, -1, -2)
POINT_BLUE = (255, 40, 20)
#9 Grenen skal vifte på boblen i 0,5 sekund, før den brister.
WAVE_SECONDS = 0.5
_WAVE_DEGREES_PER_SECOND = 15.0


class GurliGrisBackdrop:
    """Landskabet fylder billedet. Himmel og bakke bliver stående."""

    def __init__(self, image_path: Path = BACKGROUND_PATH) -> None:
        self.image_path = image_path
        loaded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if loaded is None:
            raise FileNotFoundError(f"Gurli Gris-baggrund mangler: {image_path}")
        self._image = loaded
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}

    def render(self, width: int, height: int) -> np.ndarray:
        #9 Cover: hele vinduet er landskabet, uden sorte kanter.
        key = (width, height)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()
        src_h, src_w = self._image.shape[:2]
        scale = max(width / src_w, height / src_h)
        resized = cv2.resize(
            self._image,
            (max(width, int(round(src_w * scale))), max(height, int(round(src_h * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
        y0 = max(0, (resized.shape[0] - height) // 2)
        x0 = max(0, (resized.shape[1] - width) // 2)
        crop = resized[y0 : y0 + height, x0 : x0 + width]
        if crop.shape[0] != height or crop.shape[1] != width:
            crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
        self._cache[key] = crop
        return crop.copy()


class GurliArm:
    """En arm klippet ud af gurli.png. Skulderen er nederst, så den kan drejes som poterne."""

    def __init__(self, sprite: np.ndarray, alpha: np.ndarray, shoulder: Tuple[int, int]) -> None:
        self._sprite = sprite
        self._alpha = alpha
        self.shoulder = shoulder
        self._grip: Optional[Tuple[float, float]] = None
        self._tip: Optional[Tuple[float, float]] = None
        self._branch_half = 0.0

    def hold(self, branch: np.ndarray, branch_alpha: np.ndarray, *, mirror: bool) -> None:
        #9 Venstre arm får den spejlvendte gren, så bladene vender ud på begge sider.
        if mirror:
            branch = cv2.flip(branch, 1)
            branch_alpha = cv2.flip(branch_alpha, 1)
        self._sprite, self._alpha, self._grip, self._tip, self._branch_half = _hold_branch(
            self._sprite, self._alpha, branch, branch_alpha
        )

    def swing(self, origin_x: int, origin_y: int, scale: float, clockwise: float) -> Optional["BranchSwing"]:
        if self._grip is None or self._tip is None:
            return None
        arm_h = max(8, int(round(self._sprite.shape[0] * scale)))
        arm_w = max(6, int(round(arm_h * self._sprite.shape[1] / max(self._sprite.shape[0], 1))))
        anchor_x = origin_x + self.shoulder[0] * scale
        anchor_y = origin_y + self.shoulder[1] * scale
        grip = _sprite_to_canvas(self._grip, self._sprite.shape, arm_w, arm_h, clockwise, anchor_x, anchor_y)
        tip = _sprite_to_canvas(self._tip, self._sprite.shape, arm_w, arm_h, clockwise, anchor_x, anchor_y)
        thickness = self._branch_half * arm_w / max(self._sprite.shape[1], 1)
        return BranchSwing(grip, tip, thickness, clockwise)

    def draw(self, canvas: np.ndarray, origin_x: int, origin_y: int, scale: float, clockwise: float) -> None:
        #9 Samme drejning som PawArms i Humdingers: 0 er op, 180 er ned.
        arm_h = max(8, int(round(self._sprite.shape[0] * scale)))
        arm_w = max(6, int(round(arm_h * self._sprite.shape[1] / max(self._sprite.shape[0], 1))))
        sprite = cv2.resize(self._sprite, (arm_w, arm_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(self._alpha, (arm_w, arm_h), interpolation=cv2.INTER_LINEAR)
        turned, turned_a, pivot = _rotate_bottom(sprite, mask, -clockwise)
        ax = origin_x + int(round(self.shoulder[0] * scale))
        ay = origin_y + int(round(self.shoulder[1] * scale))
        ChaseInCar._blit(canvas, turned, turned_a, ax - pivot[0], ay - pivot[1])


class GurliGrisFigure:
    """Figuren står i landskabet. Armene følger silhuettens vinkel."""

    def __init__(self, image_path: Path = FIGURE_PATH) -> None:
        self.image_path = image_path
        loaded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if loaded is None:
            raise FileNotFoundError(f"Gurli Gris-figur mangler: {image_path}")
        cut_bgr, cut_alpha = _cut_white_background(loaded)
        self._body, self._body_alpha, self.left_arm, self.right_arm = _split_arms(cut_bgr, cut_alpha)
        branch, branch_alpha = _load_branch()
        self.left_arm.hold(branch, branch_alpha, mirror=True)
        self.right_arm.hold(branch, branch_alpha, mirror=False)

    def branch_swings(self, pose: Optional[BodyPose], width: int, height: int) -> List["BranchSwing"]:
        origin_x, origin_y, scale, left_deg, right_deg = self._layout(pose, width, height)
        swings: List[BranchSwing] = []
        for arm, angle in ((self.left_arm, left_deg), (self.right_arm, right_deg)):
            swing = arm.swing(origin_x, origin_y, scale, angle)
            if swing is not None:
                swings.append(swing)
        return swings

    def draw(self, canvas: np.ndarray, pose: Optional[BodyPose]) -> None:
        height, width = canvas.shape[:2]
        origin_x, origin_y, scale, left_deg, right_deg = self._layout(pose, width, height)
        body_w = max(1, int(round(self._body.shape[1] * scale)))
        body_h = max(1, int(round(self._body.shape[0] * scale)))
        sprite = cv2.resize(self._body, (body_w, body_h), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(self._body_alpha, (body_w, body_h), interpolation=cv2.INTER_LINEAR)
        ChaseInCar._blit(canvas, sprite, alpha, origin_x, origin_y)
        self.left_arm.draw(canvas, origin_x, origin_y, scale, left_deg)
        self.right_arm.draw(canvas, origin_x, origin_y, scale, right_deg)

    def _layout(
        self, pose: Optional[BodyPose], width: int, height: int
    ) -> Tuple[int, int, float, float, float]:
        feet_y = int(height * _FEET_Y)
        fig_h = min(int(height * _FIGURE_H), max(40, feet_y - 4))
        scale = fig_h / self._body.shape[0]
        body_w = max(1, int(round(self._body.shape[1] * scale)))
        #9 Vip til højre flytter Gurli til højre. Samme metode som Humdingers.
        origin_x = width // 2 - body_w // 2 + HeadTilt().offset_x(pose, width)
        origin_x = int(np.clip(origin_x, 0, max(0, width - body_w)))
        origin_y = feet_y - int(round(self._body.shape[0] * scale))
        left_deg, right_deg = _arm_angles(pose)
        return origin_x, origin_y, scale, left_deg, right_deg


class BranchSwing:
    """Grenen fra håndtag til blade, i billedets koordinater."""

    def __init__(self, grip: Tuple[float, float], tip: Tuple[float, float], thickness: float, angle: float) -> None:
        self.grip = grip
        self.tip = tip
        self.thickness = thickness
        self.angle = angle

    def touches(self, x: float, y: float, radius: float) -> bool:
        return _point_segment_distance(x, y, self.grip, self.tip) <= radius + self.thickness


class SoapBubble:
    """En sæbeboble med et blåt pointtal. Den falder nedad."""

    def __init__(self, x: float, y: float, value: int, radius: int, speed: float) -> None:
        self.x = x
        self.y = y
        self.value = value
        self.radius = radius
        self.speed = speed
        self.wave_time = 0.0

    def tick(self) -> None:
        self.y += self.speed

    def fallen_off(self, height: int) -> bool:
        return self.y - self.radius > height

    def hits(self, px: int, py: int) -> bool:
        return math.hypot(self.x - px, self.y - py) <= self.radius

    def label(self) -> str:
        if self.value > 0:
            return f"+{self.value}"
        return str(self.value)

    def draw(self, canvas: np.ndarray) -> None:
        height, width = canvas.shape[:2]
        cx, cy, radius = int(self.x), int(self.y), self.radius
        x1, y1 = max(0, cx - radius - 2), max(0, cy - radius - 2)
        x2, y2 = min(width, cx + radius + 3), min(height, cy + radius + 3)
        if x2 <= x1 or y2 <= y1:
            return
        roi = canvas[y1:y2, x1:x2]
        overlay = roi.copy()
        local = (cx - x1, cy - y1)
        cv2.circle(overlay, local, radius, (250, 230, 200), -1, cv2.LINE_AA)
        cv2.circle(overlay, local, radius, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(
            overlay,
            (local[0] - radius // 3, local[1] - radius // 3),
            max(2, radius // 6),
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, dst=roi)
        text = self.label()
        scale = 0.55
        text_w, text_h = measure_text(text, scale)
        #9 Tallet er blåt og sidder inde i boblen.
        draw_text(canvas, text, (cx - text_w // 2, cy - text_h // 2), scale=scale, color=POINT_BLUE)
        if self.wave_time > 0:
            #9 Buen viser, hvor længe grenen har viftet på boblen.
            fraction = min(1.0, self.wave_time / WAVE_SECONDS)
            cv2.ellipse(
                canvas,
                (cx, cy),
                (radius + 6, radius + 6),
                0,
                -90,
                -90 + int(360 * fraction),
                POINT_BLUE,
                3,
                cv2.LINE_AA,
            )


class PointGain:
    """Tallet stiger op fra boblen, når pointene er modtaget."""

    def __init__(self, x: float, y: float, value: int) -> None:
        self.x = x
        self.y = y
        self.origin_y = y
        self.value = value
        self.age = 0.0

    def tick(self, dt: float) -> None:
        self.age += dt
        self.y -= 46.0 * dt

    def visible(self) -> bool:
        return self.age < 1.05

    def label(self) -> str:
        if self.value > 0:
            return f"+{self.value}"
        return str(self.value)

    def draw(self, canvas: np.ndarray) -> None:
        burst = (int(self.x), int(self.origin_y))
        ring = int(18 + self.age * 70)
        cv2.circle(canvas, burst, ring, (255, 255, 255), 2, cv2.LINE_AA)
        text = self.label()
        scale = 0.95
        text_w, text_h = measure_text(text, scale)
        #9 Blåt tal, samme farve som inde i boblen, stiger opad.
        draw_text(canvas, text, (int(self.x) - text_w // 2, int(self.y) - text_h // 2), scale=scale, color=POINT_BLUE)


class BubbleRain:
    """Bobler falder ned. De står stille, mens viftetiden tæller, og falder igen, når grenen slipper. De brister efter 0,5 sekund."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.score = 0
        self.bubbles: List[SoapBubble] = []
        self.gains: List[PointGain] = []
        self._rng = rng if rng is not None else np.random.default_rng()
        self._spawn_in = 0
        self._last_now: Optional[float] = None
        self._prev_angles: List[float] = []

    #9 Nyt spil starter med score 0.
    def reset(self) -> None:
        self.score = 0
        self.bubbles.clear()
        self.gains.clear()
        self._spawn_in = 0
        self._last_now = None
        self._prev_angles = []

    def tick(self, width: int, height: int, swings: Sequence[BranchSwing], now: float) -> None:
        if self._last_now is None:
            self._last_now = now
        dt = max(0.0, min(0.1, now - self._last_now))
        self._last_now = now
        self._spawn_in -= 1
        if self._spawn_in <= 0 and len(self.bubbles) < 5:
            self.bubbles.append(self._make(width, height))
            self._spawn_in = 40
        alive: List[SoapBubble] = []
        for bubble in self.bubbles:
            if self._waving_on(bubble, swings, dt):
                bubble.wave_time += dt
            elif not _branch_touches(bubble, swings):
                bubble.wave_time = 0.0
            #9 Boblen står stille, mens viftetiden tæller. Slipper grenen, falder den igen.
            if bubble.wave_time <= 0.0:
                bubble.tick()
            if bubble.wave_time + 1e-9 >= WAVE_SECONDS:
                self.score += bubble.value
                self.gains.append(PointGain(bubble.x, bubble.y, bubble.value))
                #9 Plus-boble popper som glas. Minus-boble bruger fejl-lyden.
                if bubble.value < 0:
                    game_audio.points(bubble.value)
                else:
                    game_audio.bubble()
                continue
            if not bubble.fallen_off(height):
                alive.append(bubble)
        self.bubbles = alive
        self._prev_angles = [swing.angle for swing in swings]
        still_gains: List[PointGain] = []
        for gain in self.gains:
            gain.tick(dt)
            if gain.visible():
                still_gains.append(gain)
        self.gains = still_gains

    def _waving_on(self, bubble: SoapBubble, swings: Sequence[BranchSwing], dt: float) -> bool:
        if dt <= 0 or len(swings) != len(self._prev_angles):
            return False
        for swing, previous in zip(swings, self._prev_angles):
            if not swing.touches(bubble.x, bubble.y, bubble.radius):
                continue
            delta = _angle_delta(swing.angle, previous)
            if delta >= 0.4 and delta / dt >= _WAVE_DEGREES_PER_SECOND:
                return True
        return False

    def draw(self, canvas: np.ndarray) -> None:
        for bubble in self.bubbles:
            bubble.draw(canvas)
        for gain in self.gains:
            gain.draw(canvas)

    def draw_score(self, canvas: np.ndarray) -> None:
        text = f"Score {self.score}"
        scale = 0.75
        text_w, text_h = measure_text(text, scale)
        x = canvas.shape[1] // 2 - text_w // 2
        y = 10
        cv2.rectangle(canvas, (x - 10, y - 4), (x + text_w + 10, y + text_h + 6), (32, 36, 48), -1)
        draw_text(canvas, text, (x, y), scale=scale, color=(240, 240, 240))

    def _make(self, width: int, height: int) -> SoapBubble:
        radius = max(26, int(height * 0.07))
        margin = radius + 4
        x = float(self._rng.integers(margin, max(margin + 1, width - margin)))
        value = int(self._rng.choice(POINT_VALUES))
        return SoapBubble(x, float(-radius), value, radius, max(2.5, height * 0.012))


def _branch_touches(bubble: SoapBubble, swings: Sequence[BranchSwing]) -> bool:
    return any(swing.touches(bubble.x, bubble.y, bubble.radius) for swing in swings)


def _angle_delta(current: float, previous: float) -> float:
    delta = (current - previous) % 360.0
    if delta > 180.0:
        delta -= 360.0
    return abs(delta)


def _point_segment_distance(
    px: float, py: float, start: Tuple[float, float], end: Tuple[float, float]
) -> float:
    ax, ay = start
    bx, by = end
    abx, aby = bx - ax, by - ay
    length2 = abx * abx + aby * aby
    if length2 < 1.0:
        return math.hypot(px - ax, py - ay)
    along = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length2))
    return math.hypot(px - (ax + along * abx), py - (ay + along * aby))


def _sprite_to_canvas(
    point: Tuple[float, float],
    sprite_shape: Tuple[int, ...],
    arm_w: int,
    arm_h: int,
    clockwise: float,
    anchor_x: float,
    anchor_y: float,
) -> Tuple[float, float]:
    sprite_h, sprite_w = sprite_shape[:2]
    px = point[0] * arm_w / max(sprite_w, 1)
    py = point[1] * arm_h / max(sprite_h, 1)
    #9 Samme drejning som _rotate_bottom: 0 peger op, positiv vinkel med uret.
    theta = math.radians(-clockwise)
    cosine, sine = math.cos(theta), math.sin(theta)
    vx = px - arm_w / 2.0
    vy = py - arm_h
    return anchor_x + cosine * vx + sine * vy, anchor_y - sine * vx + cosine * vy


class GurliGrisPlayer:
    """Baggrund, Gurli og punktsilhuet. Samme lag som ARPlayerView, uden webcam-billede."""

    def __init__(self) -> None:
        self._pose = PoseBodyOverlay()
        self.figure = GurliGrisFigure()
        self.bubbles = BubbleRain()
        self._last_pose: Optional[BodyPose] = None

    def close(self) -> None:
        self._pose.close()

    def refresh_pose(self, webcam: np.ndarray) -> None:
        self._last_pose = self._pose.snapshot(webcam)

    def composite(self, webcam: np.ndarray, background: np.ndarray) -> np.ndarray:
        if background.shape[:2] != webcam.shape[:2]:
            background = cv2.resize(background, (webcam.shape[1], webcam.shape[0]))
        pose = self._last_pose
        if pose is None:
            pose = self._pose.snapshot(webcam)
            self._last_pose = pose
        scene = background.copy()
        height, width = scene.shape[:2]
        #9 Grenene skal ramme den boble, der tegnes i samme billede.
        self.bubbles.tick(width, height, self.figure.branch_swings(pose, width, height), time.monotonic())
        #9 Figuren står på græsset. Silhuetten må ligge hvor som helst, som i Humdingers.
        self.figure.draw(scene, pose)
        self.bubbles.draw(scene)
        #9 Punktsilhuetten er debug oven på figuren.
        self._pose.draw(scene, pose)
        self.bubbles.draw_score(scene)
        return scene


def _load_branch(path: Path = BRANCH_PATH) -> Tuple[np.ndarray, np.ndarray]:
    loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if loaded is None:
        raise FileNotFoundError(f"Gurli Gris-gren mangler: {path}")
    return _cut_white_background(loaded)


def _hold_branch(
    arm: np.ndarray,
    arm_alpha: np.ndarray,
    branch: np.ndarray,
    branch_alpha: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float]], Optional[Tuple[float, float]], float]:
    """Grenens tykke ende sidder i hånden. Skulderen bliver i bunden, så armen stadig drejer der."""
    arm_h, arm_w = arm.shape[:2]
    hand_ys, hand_xs = np.where(arm_alpha > 0.25)
    if len(hand_xs) == 0:
        return arm, arm_alpha, None, None, 0.0
    hand_y = int(hand_ys.min())
    hand_band = hand_ys <= hand_y + 2
    hand_x = int(np.median(hand_xs[hand_band]))
    target_h = max(12, int(round(arm_h * 1.15)))
    scale = target_h / max(branch.shape[0], 1)
    branch_w = max(4, int(round(branch.shape[1] * scale)))
    branch_h = max(4, int(round(branch.shape[0] * scale)))
    branch = cv2.resize(branch, (branch_w, branch_h), interpolation=cv2.INTER_AREA)
    branch_alpha = cv2.resize(branch_alpha, (branch_w, branch_h), interpolation=cv2.INTER_AREA)
    grip = max(2, branch_h // 6)
    x0 = min(0, hand_x - branch_w // 2)
    y0 = min(0, hand_y - branch_h + grip)
    x1 = max(arm_w, hand_x - branch_w // 2 + branch_w)
    y1 = arm_h
    content_w = x1 - x0
    pivot_x = arm_w // 2 - x0
    left_extent = pivot_x
    right_extent = content_w - pivot_x
    pad_left = max(0, right_extent - left_extent)
    pad_right = max(0, left_extent - right_extent)
    sheet_h = y1 - y0
    sheet_w = content_w + pad_left + pad_right
    sheet = np.zeros((sheet_h, sheet_w, 3), np.uint8)
    sheet_alpha = np.zeros((sheet_h, sheet_w), np.float32)
    arm_x = pad_left - x0
    arm_y = -y0
    sheet[arm_y : arm_y + arm_h, arm_x : arm_x + arm_w] = arm
    sheet_alpha[arm_y : arm_y + arm_h, arm_x : arm_x + arm_w] = arm_alpha
    branch_x = arm_x + hand_x - branch_w // 2
    branch_y = arm_y + hand_y - branch_h + grip
    _blit_alpha(sheet, sheet_alpha, branch, branch_alpha, branch_x, branch_y)
    grip = (branch_x + branch_w / 2.0, branch_y + branch_h)
    tip = (branch_x + branch_w / 2.0, float(branch_y))
    return sheet, sheet_alpha, grip, tip, branch_w / 2.0


def _blit_alpha(
    sheet: np.ndarray,
    sheet_alpha: np.ndarray,
    sprite: np.ndarray,
    sprite_alpha: np.ndarray,
    x: int,
    y: int,
) -> None:
    sh, sw = sprite.shape[:2]
    height, width = sheet.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + sw), min(height, y + sh)
    if x2 <= x1 or y2 <= y1:
        return
    sx, sy = x1 - x, y1 - y
    src = sprite[sy : sy + (y2 - y1), sx : sx + (x2 - x1)]
    mask = sprite_alpha[sy : sy + (y2 - y1), sx : sx + (x2 - x1)]
    roi = sheet[y1:y2, x1:x2]
    roi_a = sheet_alpha[y1:y2, x1:x2]
    keep = mask > 0.2
    roi[keep] = src[keep]
    roi_a[keep] = np.maximum(roi_a[keep], mask[keep])


def _split_arms(
    bgr: np.ndarray, alpha: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, GurliArm, GurliArm]:
    #9 Kjole og hud er målt på gurli.png. Hoved, hale og ben er andre komponenter end armene.
    opaque = alpha > 0.5
    blue = bgr[:, :, 0].astype(np.int16)
    green = bgr[:, :, 1].astype(np.int16)
    red = bgr[:, :, 2].astype(np.int16)
    dress = opaque & (red > 140) & (green < 140) & (blue < 150) & (red > green + 40)
    skin = opaque & ~dress & (red > 160) & (green > 80) & (blue > 100) & (red > green - 10)
    dress_ys, dress_xs = np.where(dress)
    if len(dress_xs) == 0:
        raise ValueError("Gurli Gris-billedet har ingen rød kjole, så armene kan ikke skilles fra.")
    dress_top = int(dress_ys.min())
    dress_bottom = int(dress_ys.max())
    center_x = (int(dress_xs.min()) + int(dress_xs.max())) // 2
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(skin.astype(np.uint8), 8)
    head = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    chosen: Dict[str, int] = {}
    ranked = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        cy = float(centroids[index][1])
        if area < 80 or not (dress_top <= cy <= dress_bottom):
            continue
        ranked.append((abs(cy - dress_top), index, float(centroids[index][0])))
    ranked.sort()
    for _dist, index, cx in ranked:
        side = "L" if cx < center_x else "R"
        chosen.setdefault(side, index)
    if "L" not in chosen or "R" not in chosen:
        raise ValueError("Kunne ikke finde begge arme i Gurli Gris-billedet.")
    kernel = np.ones((5, 5), np.uint8)
    masks = []
    for side in ("L", "R"):
        grown = cv2.dilate((labels == chosen[side]).astype(np.uint8), kernel, iterations=1).astype(bool)
        masks.append(grown & opaque & ~dress & ~head)
    left_mask, right_mask = masks
    overlap = left_mask & right_mask
    left_mask = left_mask & ~overlap
    right_mask = right_mask & ~overlap
    body_alpha = alpha.copy()
    body_alpha[left_mask | right_mask] = 0
    left = _arm_from_mask(bgr, alpha, left_mask, center_x)
    right = _arm_from_mask(bgr, alpha, right_mask, center_x)
    return bgr, body_alpha, left, right


def _arm_from_mask(bgr: np.ndarray, alpha: np.ndarray, mask: np.ndarray, center_x: int) -> GurliArm:
    ys, xs = np.where(mask)
    dist = np.abs(xs.astype(np.int32) - center_x)
    near = dist <= np.percentile(dist, 8)
    shoulder = (int(np.median(xs[near])), int(np.median(ys[near])))
    farthest = int(np.argmax((xs - shoulder[0]) ** 2 + (ys - shoulder[1]) ** 2))
    hand = (int(xs[farthest]), int(ys[farthest]))
    sprite, arm_alpha = _point_arm_up(bgr, alpha, mask, shoulder, hand)
    return GurliArm(sprite, arm_alpha, shoulder)


def _point_arm_up(
    bgr: np.ndarray,
    alpha: np.ndarray,
    mask: np.ndarray,
    shoulder: Tuple[int, int],
    hand: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    sprite = bgr[y1:y2, x1:x2].copy()
    arm_alpha = np.where(mask[y1:y2, x1:x2], alpha[y1:y2, x1:x2], 0).astype(np.float32)
    shoulder_xy = (shoulder[0] - x1, shoulder[1] - y1)
    hand_xy = (hand[0] - x1, hand[1] - y1)
    current = math.degrees(math.atan2(hand_xy[0] - shoulder_xy[0], -(hand_xy[1] - shoulder_xy[1])))
    crop_h, crop_w = sprite.shape[:2]
    radius = int(math.hypot(crop_w, crop_h)) + 4
    side = radius * 2
    sheet = np.zeros((side, side, 3), np.uint8)
    sheet_alpha = np.zeros((side, side), np.float32)
    ox = side // 2 - shoulder_xy[0]
    oy = side // 2 - shoulder_xy[1]
    sheet[oy : oy + crop_h, ox : ox + crop_w] = sprite
    sheet_alpha[oy : oy + crop_h, ox : ox + crop_w] = arm_alpha
    matrix = cv2.getRotationMatrix2D((side / 2.0, side / 2.0), current, 1.0)
    turned = cv2.warpAffine(sheet, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    turned_a = cv2.warpAffine(sheet_alpha, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
    ys, xs = np.where(turned_a > 0.25)
    pad = 2
    top = max(0, int(ys.min()) - pad)
    bottom = side // 2 + 1
    left = max(0, int(xs.min()) - pad)
    right = min(side - 1, int(xs.max()) + pad)
    return turned[top:bottom, left : right + 1], turned_a[top:bottom, left : right + 1]


class GurliGrisGame(RoundScoreGame, WebcamARGame):
    """Gurli står i landskabet. Kameraet bruges kun til silhuetten."""

    window_name = "gamConsole: Gurli Gris"

    def __init__(self, camera: int, scores: Optional[HighScoreList] = None) -> None:
        super().__init__(camera)
        self.backdrop = GurliGrisBackdrop()
        self.player = GurliGrisPlayer()
        #9 Samme 3 minutter og highscore som Humdingers, i Gurlis egen fil.
        board = scores if scores is not None else HighScoreList(highscore_file("gurli_gris"))
        self.match = RoundScore(board)
        #19 Sangen kører kun, mens runden spilles.
        self.track = DrumTrack(SONG_PATH)

    def close(self) -> None:
        #19 Menuen og lukket vindue stopper sangen.
        self.track.stop()
        self.player.close()

    def play_again(self) -> None:
        #19 Spil igen starter sangen forfra.
        self.track.stop()
        self.player.bubbles.reset()
        self.match.clear()

    def on_frame(self, frame: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        #9 Spejlvendt webcam, samme retning som Humdingers killinger.
        camera = cv2.flip(frame, 1)
        moment = time.monotonic() if now is None else now
        shown = self.match.begin(
            moment,
            self.player.bubbles.score,
            self.play_again,
            lambda: self.backdrop.render(camera.shape[1], camera.shape[0]),
        )
        if shown is not None:
            #19 Resultatskærmen er stille. Næste runde starter sangen forfra.
            self.track.stop()
            return shown
        self.track.ensure_playing()
        self.player.refresh_pose(camera)
        height, width = camera.shape[:2]
        background = self.backdrop.render(width, height)
        scene = self.player.composite(camera, background)
        return self.match.stamp(scene, moment)


def main() -> None:
    parser = argparse.ArgumentParser(description="gamConsole: Gurli Gris")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    game = GurliGrisGame(args.camera)
    try:
        game.run()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    finally:
        game.close()


if __name__ == "__main__":
    main()
