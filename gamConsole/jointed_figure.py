"""Fælles udklip og drejning for avatarer med hofte, knæ, håndled og hals.

Danseren, bokseren og målmanden bruger de samme led. En pixel hører til
det nærmeste ben, og delen drejer, så den peger samme vej som silhuetten.
0 er op, 180 er ned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from gamConsole.leader_back_view import ChaseInCar, _arm_clockwise
from gamConsole.pose_body_overlay import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    BodyPose,
)

# MediaPipe-hånden. Pose-filen er issue 1, så indeksene ligger her.
L_INDEX, R_INDEX = 19, 20
_HEAD_PAIRS = ((7, 8), (3, 6), (2, 5))
_HEAD_LIMIT = 35.0
Point = Tuple[float, float]

DRAW_ORDER = (
    "torso",
    "thigh_left",
    "thigh_right",
    "shin_left",
    "shin_right",
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "hand_left",
    "hand_right",
    "head",
)

_CHAINS = (
    ("upper_left", "lower_left"),
    ("lower_left", "hand_left"),
    ("upper_right", "lower_right"),
    ("lower_right", "hand_right"),
    ("thigh_left", "shin_left"),
    ("thigh_right", "shin_right"),
)


@dataclass
class Bone:
    """Et stykke af skelettet. reach er None for hoved og krop."""

    kind: str
    start: Point
    end: Point
    reach: Optional[float] = None
    along_min: float = -0.08
    along_max: float = 1.22


class BodyPart:
    """Et udklip. Pivot ligger i billedets koordinater, før beskæring."""

    def __init__(self, name: str, bgr: np.ndarray, alpha: np.ndarray, origin: Tuple[int, int], pivot: Point, tip: Point) -> None:
        self.name = name
        self.bgr = bgr
        self.alpha = alpha
        self.origin = origin
        self.pivot = pivot
        self.tip = tip
        self.rest = _direction(pivot, tip) if name != "torso" else 0.0
        self.joint_color, self.joint_radius = _joint_cap(self)

    @property
    def area(self) -> int:
        return int((self.alpha > 0.2).sum())


def cut_parts(
    image: np.ndarray,
    body: np.ndarray,
    bones: Sequence[Bone],
    pivots: Dict[str, Point],
    tips: Dict[str, Point],
    shares: Sequence[Tuple[Point, str, str, int]],
    head_y_max: float,
    head_reach: float,
) -> Dict[str, BodyPart]:
    """Del figuren op. Kroppen får de pixels, ingen lemmer tager."""
    ys, xs = np.where(body)
    if len(xs) == 0:
        raise ValueError("Avataren har ingen figur")
    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    best = np.full(len(points), 1e9, np.float32)
    owner = np.full(len(points), -1, np.int32)
    for index, bone in enumerate(bones):
        dist = _segment_distance(points, bone.start, bone.end)
        along = _along(points, bone.start, bone.end)
        if bone.kind == "head":
            ok = (points[:, 1] <= head_y_max) & (dist < head_reach)
        elif bone.kind == "torso":
            ok = np.ones(len(points), dtype=bool)
        else:
            reach = bone.reach if bone.reach is not None else 1e9
            ok = (dist <= reach) & (along >= bone.along_min) & (along <= bone.along_max)
        dist = np.where(ok, dist, 1e9)
        take = dist < best
        best[take] = dist[take]
        owner[take] = index
    torso_index = next(i for i, bone in enumerate(bones) if bone.kind == "torso")
    owner[owner < 0] = torso_index
    masks: Dict[str, np.ndarray] = {}
    for bone in bones:
        masks.setdefault(bone.kind, np.zeros(body.shape, np.uint8))
    for point, bone_index in zip(points, owner):
        masks[bones[int(bone_index)].kind][int(point[1]), int(point[0])] = 1
    for mask in masks.values():
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask[:] = (closed > 0) & body
    for joint, parent, child, radius in shares:
        _share_joint(masks, body, joint, parent, child, radius)
    parts: Dict[str, BodyPart] = {}
    for name, mask in masks.items():
        if name not in pivots:
            continue
        crop_ys, crop_xs = np.where(mask > 0)
        if len(crop_xs) == 0:
            continue
        x0, x1 = int(crop_xs.min()), int(crop_xs.max()) + 1
        y0, y1 = int(crop_ys.min()), int(crop_ys.max()) + 1
        parts[name] = BodyPart(
            name,
            image[y0:y1, x0:x1].copy(),
            mask[y0:y1, x0:x1].astype(np.float32),
            (x0, y0),
            pivots[name],
            tips.get(name, pivots[name]),
        )
    return parts


def posed_layout(
    parts: Dict[str, BodyPart],
    pose: Optional[BodyPose],
    origin_x: float,
    origin_y: float,
    scale: float,
) -> Tuple[Dict[str, float], Dict[str, Point]]:
    angles = player_angles(pose, parts)
    return angles, _anchors(parts, angles, origin_x, origin_y, scale)


def draw_parts(
    canvas: np.ndarray,
    parts: Dict[str, BodyPart],
    pose: Optional[BodyPose],
    origin_x: float,
    origin_y: float,
    scale: float,
    edge: int = 0,
) -> None:
    angles, anchors = posed_layout(parts, pose, origin_x, origin_y, scale)
    covered = np.zeros(canvas.shape[:2], np.bool_)
    for name in DRAW_ORDER:
        part = parts.get(name)
        if part is None:
            continue
        _paint(canvas, part, anchors[name], scale, angles.get(name, 0.0), covered, edge)
    for name in DRAW_ORDER:
        part = parts.get(name)
        if part is None:
            continue
        # Cirklen fylder kun hullet ved leddet, i delens egen farve.
        _fill_joint(canvas, covered, anchors[name], part.joint_color, part.joint_radius * scale)


def limb_tip(part: BodyPart, anchor: Point, scale: float, clockwise: float) -> Point:
    turned = turn(part.tip, part.pivot, clockwise)
    return (
        anchor[0] + (turned[0] - part.pivot[0]) * scale,
        anchor[1] + (turned[1] - part.pivot[1]) * scale,
    )


def player_angles(pose: Optional[BodyPose], parts: Dict[str, BodyPart]) -> Dict[str, float]:
    """Hofte, knæ, albue, håndled og hals. Et led uden punkter bliver i hvile."""
    if pose is None:
        return {name: 0.0 for name in parts}
    angles = {name: 0.0 for name in parts}
    left_arm, right_arm = _ordered(
        pose,
        (L_SHOULDER, L_ELBOW, L_WRIST, L_INDEX),
        (R_SHOULDER, R_ELBOW, R_WRIST, R_INDEX),
    )
    left_leg, right_leg = _ordered(pose, (L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE))
    _set_delta(angles, parts, "upper_left", pose, left_arm[0], left_arm[1])
    _set_delta(angles, parts, "lower_left", pose, left_arm[1], left_arm[2])
    _set_hand(angles, parts, "hand_left", "lower_left", pose, left_arm[2], left_arm[3])
    _set_delta(angles, parts, "upper_right", pose, right_arm[0], right_arm[1])
    _set_delta(angles, parts, "lower_right", pose, right_arm[1], right_arm[2])
    _set_hand(angles, parts, "hand_right", "lower_right", pose, right_arm[2], right_arm[3])
    _set_delta(angles, parts, "thigh_left", pose, left_leg[0], left_leg[1])
    _set_delta(angles, parts, "shin_left", pose, left_leg[1], left_leg[2])
    _set_delta(angles, parts, "thigh_right", pose, right_leg[0], right_leg[1])
    _set_delta(angles, parts, "shin_right", pose, right_leg[1], right_leg[2])
    if "head" in parts:
        angles["head"] = head_roll(pose)
    return angles


def head_roll(pose: Optional[BodyPose]) -> float:
    """Halsen. Samme øre-linje som hovedvippen, her som en drejning."""
    if pose is None:
        return 0.0
    for left_i, right_i in _HEAD_PAIRS:
        left = pose.get(left_i)
        right = pose.get(right_i)
        if left is None or right is None:
            continue
        if left[0] > right[0]:
            left, right = right, left
        dx = float(right[0] - left[0])
        dy = float(right[1] - left[1])
        if dx < 4.0:
            continue
        return max(-_HEAD_LIMIT, min(_HEAD_LIMIT, math.degrees(math.atan2(dy, dx))))
    return 0.0


def hand_angles(pose: Optional[BodyPose], fallback: Tuple[float, float]) -> Tuple[float, float]:
    """Håndleddet. Uden pegefinger følger hånden underarmen."""
    if pose is None:
        return fallback
    left_arm, right_arm = _ordered(
        pose,
        (L_SHOULDER, L_ELBOW, L_WRIST, L_INDEX),
        (R_SHOULDER, R_ELBOW, R_WRIST, R_INDEX),
    )
    return (
        _absolute_angle(pose, left_arm[2], left_arm[3], fallback[0]),
        _absolute_angle(pose, right_arm[2], right_arm[3], fallback[1]),
    )


def turn(point: Point, pivot: Point, clockwise: float) -> Point:
    """Drej med uret. 90 grader sender et punkt over leddet ud til højre."""
    ang = math.radians(clockwise)
    cos = math.cos(ang)
    sin = math.sin(ang)
    dx = point[0] - pivot[0]
    dy = point[1] - pivot[1]
    return (pivot[0] + dx * cos - dy * sin, pivot[1] + dx * sin + dy * cos)


def _direction(start: Point, end: Point) -> float:
    return math.degrees(math.atan2(end[0] - start[0], -(end[1] - start[1])))


def _wrap(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


def _absolute_angle(pose: BodyPose, start_i: int, end_i: int, fallback: float) -> float:
    start = pose.get(start_i)
    end = pose.get(end_i)
    if start is None or end is None:
        return fallback
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if dx * dx + dy * dy < 36.0:
        return fallback
    return _arm_clockwise(pose, start_i, end_i, end_i)


def _set_hand(
    angles: Dict[str, float],
    parts: Dict[str, BodyPart],
    name: str,
    forearm: str,
    pose: BodyPose,
    wrist_i: int,
    index_i: int,
) -> None:
    """Uden pegefinger peger hånden samme vej som underarmen."""
    part = parts.get(name)
    if part is None:
        return
    forearm_part = parts.get(forearm)
    start = pose.get(wrist_i)
    end = pose.get(index_i)
    missing = start is None or end is None
    if not missing:
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        missing = dx * dx + dy * dy < 36.0
    if missing:
        if forearm_part is None:
            return
        angles[name] = _wrap(forearm_part.rest + angles.get(forearm, 0.0) - part.rest)
        return
    absolute = _arm_clockwise(pose, wrist_i, index_i, index_i)
    angles[name] = _wrap(absolute - part.rest)


def _set_delta(angles: Dict[str, float], parts: Dict[str, BodyPart], name: str, pose: BodyPose, start_i: int, end_i: int) -> None:
    part = parts.get(name)
    if part is None:
        return
    absolute = _absolute_angle(pose, start_i, end_i, part.rest)
    if absolute == part.rest and (pose.get(start_i) is None or pose.get(end_i) is None):
        return
    angles[name] = _wrap(absolute - part.rest)


def _ordered(pose: BodyPose, left: Tuple[int, ...], right: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    a = pose.get(left[0])
    b = pose.get(right[0])
    if a is not None and b is not None and a[0] > b[0]:
        return right, left
    return left, right


def _anchors(parts: Dict[str, BodyPart], angles: Dict[str, float], origin_x: float, origin_y: float, scale: float) -> Dict[str, Point]:
    def place(point: Point) -> Point:
        return (origin_x + point[0] * scale, origin_y + point[1] * scale)

    moved = {name: part.pivot for name, part in parts.items()}
    for parent, child in _CHAINS:
        upper = parts.get(parent)
        lower = parts.get(child)
        if upper is None or lower is None:
            continue
        # Spidsen følger det led, forælderen allerede er flyttet til.
        dx = upper.tip[0] - upper.pivot[0]
        dy = upper.tip[1] - upper.pivot[1]
        moved[child] = turn(
            (moved[parent][0] + dx, moved[parent][1] + dy),
            moved[parent],
            angles.get(parent, 0.0),
        )
    return {name: place(point) for name, point in moved.items()}


def _share_joint(masks: Dict[str, np.ndarray], body: np.ndarray, joint: Point, parent: str, child: str, radius: int) -> None:
    if parent not in masks or child not in masks:
        return
    x, y = int(round(joint[0])), int(round(joint[1]))
    height, width = body.shape
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = ((xx - x) ** 2 + (yy - y) ** 2) <= radius * radius
    local = body[y0:y1, x0:x1] & disk
    masks[parent][y0:y1, x0:x1][local] = 1
    masks[child][y0:y1, x0:x1][local] = 1


def _segment_distance(points: np.ndarray, start: Point, end: Point) -> np.ndarray:
    ax, ay = start
    bx, by = end
    abx = bx - ax
    aby = by - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1.0:
        return np.hypot(points[:, 0] - ax, points[:, 1] - ay)
    t = ((points[:, 0] - ax) * abx + (points[:, 1] - ay) * aby) / ab2
    t = np.clip(t, 0.0, 1.0)
    qx = ax + t * abx
    qy = ay + t * aby
    return np.hypot(points[:, 0] - qx, points[:, 1] - qy)


def _along(points: np.ndarray, start: Point, end: Point) -> np.ndarray:
    ax, ay = start
    abx, aby = end[0] - ax, end[1] - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1.0:
        return np.zeros(len(points), np.float32)
    return ((points[:, 0] - ax) * abx + (points[:, 1] - ay) * aby) / ab2


def _joint_cap(part: BodyPart) -> Tuple[Tuple[int, int, int], float]:
    """Farve og radius fra den kropsdel, der sidder omkring leddet."""
    local_x = part.pivot[0] - part.origin[0]
    local_y = part.pivot[1] - part.origin[1]
    tip_x = part.tip[0] - part.origin[0]
    tip_y = part.tip[1] - part.origin[1]
    ys, xs = np.where(part.alpha > 0.25)
    if len(xs) == 0:
        return (0, 0, 0), 0.0
    dx = xs.astype(np.float32) - local_x
    dy = ys.astype(np.float32) - local_y
    bone_x = tip_x - local_x
    bone_y = tip_y - local_y
    length = math.hypot(bone_x, bone_y)
    if length < 1.0:
        band = np.hypot(dx, dy) <= 10.0
        perp = np.hypot(dx, dy)
    else:
        along = (dx * bone_x + dy * bone_y) / length
        perp = np.abs(dx * bone_y - dy * bone_x) / length
        band = (along >= -6.0) & (along <= 14.0)
    if int(band.sum()) < 6:
        band = np.hypot(dx, dy) <= 10.0
        perp = np.hypot(dx, dy)
    if int(band.sum()) < 1:
        return (0, 0, 0), 0.0
    color = np.median(part.bgr[ys[band], xs[band]], axis=0)
    radius = float(np.percentile(perp[band], 85))
    radius = max(4.0, min(radius, 22.0))
    return (int(color[0]), int(color[1]), int(color[2])), radius


def _fill_joint(
    canvas: np.ndarray,
    covered: np.ndarray,
    anchor: Point,
    color: Tuple[int, int, int],
    radius: float,
) -> None:
    """Cirkel i samme farve som leddet. Kun hakket mellem delene bliver fyldt."""
    if radius < 1.0:
        return
    cx, cy = int(round(anchor[0])), int(round(anchor[1]))
    joint = max(4, int(round(radius)))
    reach = int(round(joint * 1.6))
    pad = joint + 2
    height, width = canvas.shape[:2]
    x0, x1 = max(0, cx - reach - pad), min(width, cx + reach + pad + 1)
    y0, y1 = max(0, cy - reach - pad), min(height, cy + reach + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return
    solid = covered[y0:y1, x0:x1]
    if not solid.any():
        return
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (joint * 2 + 1, joint * 2 + 1))
    closed = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    local_x, local_y = cx - x0, cy - y0
    yy, xx = np.ogrid[0 : y1 - y0, 0 : x1 - x0]
    disk = (xx - local_x) ** 2 + (yy - local_y) ** 2 <= reach * reach
    fill = closed & disk & ~solid
    if not fill.any():
        return
    canvas[y0:y1, x0:x1][fill] = color
    solid[fill] = True


def _paint(
    canvas: np.ndarray,
    part: BodyPart,
    anchor: Point,
    scale: float,
    clockwise: float,
    covered: np.ndarray,
    edge: int = 0,
) -> None:
    src_h, src_w = part.bgr.shape[:2]
    width = max(1, int(round(src_w * scale)))
    height = max(1, int(round(src_h * scale)))
    sprite = cv2.resize(part.bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(part.alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    pivot = ((part.pivot[0] - part.origin[0]) * width / src_w, (part.pivot[1] - part.origin[1]) * height / src_h)
    if abs(clockwise) > 0.4:
        sprite, alpha, pivot = _rotate_pivot(sprite, alpha, pivot, -clockwise)
    if edge > 0:
        sprite, alpha = _shave_edge(sprite, alpha, edge)
    x = int(round(anchor[0] - pivot[0]))
    y = int(round(anchor[1] - pivot[1]))
    ChaseInCar._blit(canvas, sprite, alpha, x, y)
    _cover(covered, alpha, x, y)


#31 Målt i 960×540. Ærmet har fire hvide pixels, underarmen fem.
def _shave_edge(bgr: np.ndarray, alpha: np.ndarray, pixels: int) -> Tuple[np.ndarray, np.ndarray]:
    solid = (alpha > 0.15).astype(np.uint8)
    size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    kept = cv2.erode(solid, kernel, iterations=1)
    return bgr, kept.astype(np.float32)


def _cover(covered: np.ndarray, alpha: np.ndarray, x: int, y: int) -> None:
    sprite_h, sprite_w = alpha.shape[:2]
    height, width = covered.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + sprite_w), min(height, y + sprite_h)
    if x2 <= x1 or y2 <= y1:
        return
    sx, sy = x1 - x, y1 - y
    patch = alpha[sy : sy + (y2 - y1), sx : sx + (x2 - x1)] > 0.15
    covered[y1:y2, x1:x2] |= patch


def _rotate_pivot(
    sprite: np.ndarray,
    alpha: np.ndarray,
    pivot: Tuple[float, float],
    opencv_deg: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    height, width = sprite.shape[:2]
    corners = ((0, 0), (width, 0), (0, height), (width, height))
    radius = max(math.hypot(x - pivot[0], y - pivot[1]) for x, y in corners) + 2.0
    side = max(2, int(math.ceil(radius * 2.0)))
    sheet = np.zeros((side, side, 3), np.uint8)
    mask = np.zeros((side, side), np.float32)
    ox = int(round(side / 2.0 - pivot[0]))
    oy = int(round(side / 2.0 - pivot[1]))
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(side, ox + width), min(side, oy + height)
    if x1 <= x0 or y1 <= y0:
        return sprite, alpha, pivot
    sx, sy = x0 - ox, y0 - oy
    sheet[y0:y1, x0:x1] = sprite[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
    mask[y0:y1, x0:x1] = alpha[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
    center = (side / 2.0, side / 2.0)
    matrix = cv2.getRotationMatrix2D(center, opencv_deg, 1.0)
    turned = cv2.warpAffine(sheet, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    turned_a = cv2.warpAffine(mask, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
    return turned, turned_a, center
