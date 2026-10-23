#1
"""Brugeren er det medsendte billede, siddende i sin bil. Punktsilhuetten er debug."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from gamConsole.adventure_city import _fill_round_rect
from gamConsole.pose_body_overlay import (
    BodyPose,
    L_ELBOW,
    L_SHOULDER,
    L_WRIST,
    R_ELBOW,
    R_SHOULDER,
    R_WRIST,
)

_ASSET = Path(__file__).resolve().parents[1] / "assets" / "player_chase.png"
_PAW_ASSET = Path(__file__).resolve().parents[1] / "assets" / "player_paw.png"
#1 Blå bil, så den ikke ligner de orange kegler den erstatter.
_CAR_BLUE = (220, 130, 40)
_ARM_DOWN = 180.0


def _leader_span(scale: float) -> float:
    return max(48.0, 70.0 * max(0.65, scale))


def _car_layout(width: int, height: int, cx: int, span: float) -> Tuple[int, int, int, int]:
    """#1 Målene på den blå bil, så et sammenstød rammer det rektangel, der tegnes."""
    car_w = max(72, int(span * 2.5))
    car_h = max(54, int(car_w * 0.74))
    side = car_w // 2 + max(4, car_w // 11)
    #1 Portrættet rager op over bilen. Hele hovedet skal blive i billedet.
    above = int(car_h * 2.15)
    below = max(6, car_h // 14)
    if above + below > height - 8 or side * 2 > width - 8:
        car_h = max(36, min(car_h, int((height - 12) / 1.36)))
        car_w = max(48, min(car_w, int(car_h / 0.74), width - 16))
        car_h = max(36, int(car_w * 0.74))
        side = car_w // 2 + max(4, car_w // 11)
    cx = int(np.clip(cx, side + 4, max(side + 4, width - side - 4)))
    #1 Hele figuren står i bunden.
    top = height - 1 - car_h
    return cx, top, car_w, car_h


def player_car_box(width: int, height: int, cx: int, scale: float) -> Tuple[int, int, int, int]:
    """#1 Den blå bils rektangel. Samme mål som tegningen."""
    cx, top, car_w, car_h = _car_layout(width, height, cx, _leader_span(scale))
    return cx - car_w // 2, top, cx + car_w // 2, top + car_h


class ChaseInCar:
    """#1 Billedet klippes ud og sættes i brugerens bil, altid inden for billedet."""

    def __init__(self, path: Path = _ASSET) -> None:
        self._path = path
        self._bgr: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._paws = PawArms()
        self._loaded = False

    def draw(
        self,
        canvas: np.ndarray,
        cx: int,
        feet_y: int,
        span: float,
        *,
        blend: float,
        pose: Optional[BodyPose] = None,
    ) -> None:
        self._ensure()
        height, width = canvas.shape[:2]
        del feet_y
        cx, top, car_w, car_h = _car_layout(width, height, cx, span)
        overlay = canvas.copy()
        #1 Blåt rektangel med runde hjørner i stedet for bil-ikonet.
        _fill_round_rect(
            overlay,
            cx - car_w // 2,
            top,
            cx + car_w // 2,
            top + car_h,
            _CAR_BLUE,
            max(16, car_w // 5),
        )
        seated = None
        if self._bgr is not None and self._alpha is not None:
            seated = self._seat_sprite(int(cx), top, car_w, car_h)
        if seated is not None:
            sprite, alpha, px, py, pw, ph = seated
            #1 Armene sidder på vesten, ikke på hovedet, og drejer om skulderleddet.
            shoulder_y = py + int(ph * 0.82)
            self._paws.draw_pair(
                overlay,
                (px + int(pw * 0.28), shoulder_y),
                (px + int(pw * 0.72), shoulder_y),
                max(28, int(car_h * 0.84)),
                *_arm_angles(pose),
            )
            self._blit(overlay, sprite, alpha, px, py)
        cv2.addWeighted(overlay, blend, canvas, 1.0 - blend, 0, dst=canvas)

    def _seat_sprite(
        self,
        cx: int,
        car_top: int,
        car_w: int,
        car_h: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int, int, int, int]]:
        assert self._bgr is not None and self._alpha is not None
        #1 Kun overkroppen. Snittet ligger i forruden, så han sidder i bilen.
        cut = max(8, int(self._bgr.shape[0] * 0.72))
        body = self._bgr[:cut]
        mask = self._alpha[:cut]
        sh, sw = body.shape[:2]
        #1 Portrættet skal fylde lige så meget som poterne.
        plush_h = max(24, int(car_h * 1.70))
        plush_w = max(16, int(plush_h * sw / max(sh, 1)))
        limit = int(car_w * 0.92)
        if plush_w > limit:
            plush_w = limit
            plush_h = max(16, int(plush_w * sh / max(sw, 1)))
        sprite = cv2.resize(body, (plush_w, plush_h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(mask, (plush_w, plush_h), interpolation=cv2.INTER_AREA)
        seat_y = car_top + int(car_h * 0.58)
        return sprite, alpha, cx - plush_w // 2, seat_y - plush_h, plush_w, plush_h

    def _ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        image = cv2.imread(str(self._path), cv2.IMREAD_COLOR)
        if image is None:
            return
        self._bgr, self._alpha = _cut_white_background(image)
        #1 Armene får samme pelsfarve som Chase.
        self._paws.tint_like(_fur_bgr(self._bgr, self._alpha))

    @staticmethod
    def _blit(
        canvas: np.ndarray,
        sprite: np.ndarray,
        alpha: np.ndarray,
        x: int,
        y: int,
    ) -> None:
        sh, sw = sprite.shape[:2]
        height, width = canvas.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + sw), min(height, y + sh)
        if x2 <= x1 or y2 <= y1:
            return
        sx, sy = x1 - x, y1 - y
        src = sprite[sy : sy + (y2 - y1), sx : sx + (x2 - x1)]
        mask = alpha[sy : sy + (y2 - y1), sx : sx + (x2 - x1)]
        roi = canvas[y1:y2, x1:x2]
        keep = mask > 0.15
        roi[keep] = src[keep]


def _fur_bgr(image: np.ndarray, alpha: np.ndarray) -> Tuple[int, int, int]:
    """Pelsfarven: varm brun, hvor rød er den stærkeste kanal."""
    pixels = image[alpha > 0.5].astype(np.int16)
    blue, green, red = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    fur = pixels[(red > green + 15) & (green > blue + 5) & (red > 80)]
    if len(fur) == 0:
        return 63, 93, 155
    mean = fur.mean(axis=0)
    return int(mean[0]), int(mean[1]), int(mean[2])


def _cut_white_background(image: np.ndarray, spread: int = 18) -> Tuple[np.ndarray, np.ndarray]:
    """Hvid baggrund væk. Hvide poter bliver, fordi de ikke hænger sammen med kanten."""
    height, width = image.shape[:2]
    filled = np.zeros((height + 2, width + 2), np.uint8)
    work = image.copy()
    # Fast interval mod den hvide kant. Ellers kravler fyldet ind i bamsen.
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    delta = (spread, spread, spread)
    for seed in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        cv2.floodFill(work, filled, seed, (0, 0, 0), delta, delta, flags=flags)
    alpha = (filled[1:-1, 1:-1] == 0).astype(np.float32)
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return image, np.ones(image.shape[:2], np.float32)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return image[y1:y2, x1:x2], alpha[y1:y2, x1:x2]


class PawArms:
    """#1 En pote på hver side. Vinklen følger silhuettens arm."""

    def __init__(self, path: Path = _PAW_ASSET) -> None:
        self._path = path
        self._bgr: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._loaded = False

    def draw_pair(
        self,
        canvas: np.ndarray,
        left: Tuple[int, int],
        right: Tuple[int, int],
        height: int,
        left_clockwise: float,
        right_clockwise: float,
    ) -> None:
        self._ensure()
        if self._bgr is None or self._alpha is None:
            return
        flipped = cv2.flip(self._bgr, 1)
        flipped_a = cv2.flip(self._alpha, 1)
        self._draw_one(canvas, flipped, flipped_a, left, height, left_clockwise)
        self._draw_one(canvas, self._bgr, self._alpha, right, height, right_clockwise)

    def _draw_one(
        self,
        canvas: np.ndarray,
        bgr: np.ndarray,
        alpha: np.ndarray,
        anchor: Tuple[int, int],
        height: int,
        clockwise: float,
    ) -> None:
        sh, sw = bgr.shape[:2]
        paw_h = max(16, int(height))
        paw_w = max(10, int(paw_h * sw / max(sh, 1)))
        sprite = cv2.resize(bgr, (paw_w, paw_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(alpha, (paw_w, paw_h), interpolation=cv2.INTER_AREA)
        turned, turned_a, pivot = _rotate_bottom(sprite, mask, -clockwise)
        ChaseInCar._blit(canvas, turned, turned_a, anchor[0] - pivot[0], anchor[1] - pivot[1])

    def _ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        image = cv2.imread(str(self._path), cv2.IMREAD_COLOR)
        if image is None:
            return
        #1 Den lyse arm er tæt på hvid. Et snævert interval lader den blive.
        self._bgr, self._alpha = _cut_white_background(image, spread=8)
        #1 Armens ende var et skarpt hjørne. Den rundes af.
        self._alpha = _round_arm_end(self._alpha)

    def tint_like(self, fur_bgr: Tuple[int, int, int]) -> None:
        """#1 Pote og underarm males i Chase's pelsfarve. Puder og omrids bliver."""
        self._ensure()
        if self._bgr is None or self._alpha is None:
            return
        hsv = cv2.cvtColor(self._bgr, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        fill = (self._alpha > 0.2) & (value > 75)
        shade = np.clip(value.astype(np.float32) / 130.0, 0.82, 1.12)
        fur = np.array(fur_bgr, np.float32)
        tinted = np.clip(fur * shade[:, :, None], 0, 255)
        painted = self._bgr.astype(np.float32)
        painted[fill] = tinted[fill]
        self._bgr = painted.astype(np.uint8)


def _round_arm_end(alpha: np.ndarray) -> np.ndarray:
    """#1 Underarmen ender i en halvcirkel, så hjørnerne ikke er spidse."""
    solid = alpha > 0.2
    ys, xs = np.where(solid)
    if len(ys) < 8:
        return alpha
    bottom = int(ys.max())
    sample_y = max(0, bottom - max(4, alpha.shape[0] // 12))
    shaft = np.where(solid[sample_y])[0]
    if len(shaft) < 4:
        return alpha
    left, right = int(shaft.min()), int(shaft.max())
    radius = max(4, (right - left) // 2)
    mid = (left + right) / 2.0
    cap_y = bottom - radius
    height, width = alpha.shape
    yy, xx = np.ogrid[:height, :width]
    outside = (yy > cap_y) & (((xx - mid) ** 2 + (yy - cap_y) ** 2) > radius ** 2)
    rounded = alpha.copy()
    rounded[outside] = 0.0
    return rounded


def _rotate_bottom(
    sprite: np.ndarray,
    alpha: np.ndarray,
    opencv_deg: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """#1 Drej om skulderleddet. Armens rod bliver liggende, poten svinger rundt."""
    sh, sw = sprite.shape[:2]
    radius = int(math.hypot(sw / 2.0, sh)) + 2
    side = radius * 2
    sheet = np.zeros((side, side, 3), np.uint8)
    mask = np.zeros((side, side), np.float32)
    ox = side // 2 - sw // 2
    oy = side // 2 - sh
    sheet[oy : oy + sh, ox : ox + sw] = sprite
    mask[oy : oy + sh, ox : ox + sw] = alpha
    center = (side / 2.0, side / 2.0)
    matrix = cv2.getRotationMatrix2D(center, opencv_deg, 1.0)
    turned = cv2.warpAffine(sheet, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    turned_a = cv2.warpAffine(mask, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
    return turned, turned_a, (side // 2, side // 2)


def _arm_angles(pose: Optional[BodyPose]) -> Tuple[float, float]:
    """Skærm-venstre arm styrer venstre pote. 0 er op, 180 er ned."""
    if pose is None:
        return _ARM_DOWN, _ARM_DOWN
    left_ids, right_ids = _screen_arms(pose)
    return _arm_clockwise(pose, *left_ids), _arm_clockwise(pose, *right_ids)


def _screen_arms(pose: BodyPose) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    left = (L_SHOULDER, L_ELBOW, L_WRIST)
    right = (R_SHOULDER, R_ELBOW, R_WRIST)
    ls, rs = pose.get(L_SHOULDER), pose.get(R_SHOULDER)
    if ls is not None and rs is not None and ls[0] > rs[0]:
        return right, left
    return left, right


def _arm_clockwise(pose: BodyPose, shoulder_i: int, elbow_i: int, wrist_i: int) -> float:
    shoulder = pose.get(shoulder_i)
    #1 Skulderleddet er overarmen: skulder til albue.
    end = pose.get(elbow_i) or pose.get(wrist_i)
    if shoulder is None or end is None:
        return _ARM_DOWN
    dx = float(end[0] - shoulder[0])
    dy = float(end[1] - shoulder[1])
    if dx * dx + dy * dy < 36.0:
        return _ARM_DOWN
    return math.degrees(math.atan2(dx, -dy))


_CHASE = ChaseInCar()


def draw_leader_back(
    canvas: np.ndarray,
    feet_x: int,
    feet_y: int,
    *,
    scale: float = 1.0,
    pose: Optional[BodyPose] = None,
) -> None:
    """#1 Altid på vejen. Pote-armene følger silhuetten."""
    _CHASE.draw(canvas, feet_x, feet_y, _leader_span(scale), blend=1.0, pose=pose)
