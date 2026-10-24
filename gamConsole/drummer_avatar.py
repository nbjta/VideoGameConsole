#22
"""Trommeslageren fra assets/drum_hero/drummer.png.

Billedet er en stående figur. Delene er målt på den og drejet, så hvilen
sidder med knæene ude og armene oppe. Benene følger ikke spilleren, for
så ville han rejse sig. Hænderne er slagpunktet. Billedet har ingen stikker.
Cirklerne i leddene er målt som lemmets bredde, så der ikke kommer hul.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from gamConsole.jointed_figure import (
    Bone,
    BodyPart,
    Point,
    _rotate_pivot,
    cut_parts,
    draw_parts,
    limb_tip,
    posed_layout,
    turn,
)
from gamConsole.pose_body_overlay import (
    L_ANKLE,
    L_HIP,
    L_KNEE,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    BodyPose,
)

ROOT = Path(__file__).resolve().parents[1]
DRUMMER_PATH = ROOT / "assets" / "drum_hero" / "drummer.png"

#22 Målt på drummer.png, 494×619. x vokser mod højre, y vokser nedad.
_NECK = (262.0, 148.0)
_HEAD_TOP = (262.0, 66.0)
_SHOULDER_L = (208.0, 170.0)
_ELBOW_L = (192.0, 250.0)
_WRIST_L = (193.0, 328.0)
_HAND_L = (193.0, 356.0)
_SHOULDER_R = (316.0, 170.0)
_ELBOW_R = (334.0, 250.0)
_WRIST_R = (332.0, 328.0)
_HAND_R = (332.0, 356.0)
_HIP_L = (240.0, 330.0)
_KNEE_L = (237.0, 424.0)
_FOOT_L = (222.0, 550.0)
_HIP_R = (288.0, 330.0)
_KNEE_R = (293.0, 424.0)
_FOOT_R = (309.0, 550.0)
_PELVIS = (264.0, 336.0)

#22 Grader med uret fra den stående figur. Skinnebenene bliver hængende.
_TURN = {
    "upper_left": 130.0,
    "lower_left": 155.0,
    "hand_left": 165.0,
    "upper_right": -130.0,
    "lower_right": -155.0,
    "hand_right": -165.0,
    "thigh_left": 62.0,
    "thigh_right": -62.0,
}

_BONES = (
    Bone("head", _NECK, _HEAD_TOP),
    Bone("torso", _NECK, _PELVIS),
    Bone("torso", (230.0, 190.0), (246.0, 320.0)),
    Bone("torso", (294.0, 190.0), (286.0, 320.0)),
    Bone("upper_left", _SHOULDER_L, _ELBOW_L, 22.0, along_max=1.15),
    Bone("lower_left", _ELBOW_L, _WRIST_L, 18.0, along_max=1.08),
    Bone("hand_left", _WRIST_L, _HAND_L, 18.0, along_max=1.25),
    Bone("upper_right", _SHOULDER_R, _ELBOW_R, 22.0, along_max=1.15),
    Bone("lower_right", _ELBOW_R, _WRIST_R, 18.0, along_max=1.08),
    Bone("hand_right", _WRIST_R, _HAND_R, 18.0, along_max=1.25),
    Bone("thigh_left", _HIP_L, _KNEE_L, 26.0, along_max=1.2),
    Bone("shin_left", _KNEE_L, _FOOT_L, 24.0, along_max=1.15),
    Bone("thigh_right", _HIP_R, _KNEE_R, 26.0, along_max=1.2),
    Bone("shin_right", _KNEE_R, _FOOT_R, 24.0, along_max=1.15),
)

_PIVOT = {
    "head": _NECK,
    "torso": _NECK,
    "upper_left": _SHOULDER_L,
    "lower_left": _ELBOW_L,
    "hand_left": _WRIST_L,
    "upper_right": _SHOULDER_R,
    "lower_right": _ELBOW_R,
    "hand_right": _WRIST_R,
    "thigh_left": _HIP_L,
    "shin_left": _KNEE_L,
    "thigh_right": _HIP_R,
    "shin_right": _KNEE_R,
}

_TIP = {
    "head": _HEAD_TOP,
    "upper_left": _ELBOW_L,
    "lower_left": _WRIST_L,
    "hand_left": _HAND_L,
    "upper_right": _ELBOW_R,
    "lower_right": _WRIST_R,
    "hand_right": _HAND_R,
    "thigh_left": _KNEE_L,
    "shin_left": _FOOT_L,
    "thigh_right": _KNEE_R,
    "shin_right": _FOOT_R,
}

_SHARES = (
    (_NECK, "head", "torso", 16),
    (_SHOULDER_L, "torso", "upper_left", 16),
    (_SHOULDER_R, "torso", "upper_right", 16),
    (_ELBOW_L, "upper_left", "lower_left", 14),
    (_ELBOW_R, "upper_right", "lower_right", 14),
    (_WRIST_L, "lower_left", "hand_left", 12),
    (_WRIST_R, "lower_right", "hand_right", 12),
    (_HIP_L, "torso", "thigh_left", 20),
    (_HIP_R, "torso", "thigh_right", 20),
    (_KNEE_L, "thigh_left", "shin_left", 16),
    (_KNEE_R, "thigh_right", "shin_right", 16),
)

#22 Cirklen måles her, hvor lemmet er fri af kroppen. Ikke i selve leddet,
#22 for dér er jakken og bukserne stadig én flade.
_JOINTS = (
    "thigh_left",
    "thigh_right",
    "upper_left",
    "upper_right",
    "shin_left",
    "shin_right",
    "lower_left",
    "lower_right",
    "hand_left",
    "hand_right",
    "head",
)
_SAMPLE = {
    "upper_left": (192.0, 318.0),
    "lower_left": (192.0, 318.0),
    "hand_left": (193.0, 343.0),
    "upper_right": (333.0, 318.0),
    "lower_right": (333.0, 318.0),
    "hand_right": (332.0, 343.0),
    "thigh_left": (237.0, 430.0),
    "shin_left": (228.0, 490.0),
    "thigh_right": (293.0, 430.0),
    "shin_right": (301.0, 490.0),
    "head": (262.0, 132.0),
}
#22 Bøjningen åbner et kilehul. Cirklen er derfor bredere end selve lemmet dér.
_FACTOR = {
    "upper_left": 0.55,
    "upper_right": 0.55,
    "lower_left": 1.0,
    "lower_right": 1.0,
    "hand_left": 0.9,
    "hand_right": 0.9,
    "thigh_left": 0.8,
    "thigh_right": 0.8,
    "shin_left": 0.8,
    "shin_right": 0.8,
    "head": 0.55,
}

_LEGS = (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)
_SHARED: Optional[Tuple] = None


class JointCap:
    """En fyldt cirkel i leddets farve. radius er målt på billedet."""

    def __init__(self, part: str, color: Tuple[int, int, int], radius: float) -> None:
        self.part = part
        self.color = color
        self.radius = radius


class DrummerAvatar:
    """Hoved, krop, arme, hænder, lår og underben. Hænderne rammer trommerne."""

    def __init__(self) -> None:
        parts, caps, top, bottom, left, right, pixels = _shared_cut()
        self.parts = parts
        self.caps = caps
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.cx = _PELVIS[0]
        self.pixels = pixels

    def draw(self, canvas: np.ndarray, pose: Optional[BodyPose], box: Tuple[int, int, int, int]) -> None:
        seated = _seated(pose)
        origin_x, origin_y, scale = self.placement(box)
        draw_parts(canvas, self.parts, seated, origin_x, origin_y, scale)
        _draw_caps(canvas, self.parts, self.caps, seated, origin_x, origin_y, scale)

    def stick_tips(self, pose: Optional[BodyPose], box: Tuple[int, int, int, int]) -> Tuple[Point, Point]:
        """Hændernes spids i billedet. Billedet har ingen stikker."""
        origin_x, origin_y, scale = self.placement(box)
        angles, anchors = posed_layout(self.parts, _seated(pose), origin_x, origin_y, scale)
        return (
            limb_tip(self.parts["hand_left"], anchors["hand_left"], scale, angles.get("hand_left", 0.0)),
            limb_tip(self.parts["hand_right"], anchors["hand_right"], scale, angles.get("hand_right", 0.0)),
        )

    def placement(self, box: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
        x1, _y1, x2, y2 = box
        figure_h = max(1, y2 - _y1)
        scale = figure_h / max(1, self.bottom - self.top)
        center = (x1 + x2) / 2.0
        origin_x = center - self.cx * scale
        origin_y = y2 - self.bottom * scale
        return origin_x, origin_y, scale


def _shared_cut() -> Tuple:
    global _SHARED
    if _SHARED is None:
        image = cv2.imread(str(DRUMMER_PATH), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Kan ikke læse {DRUMMER_PATH}")
        body = _body_mask(image)
        parts = _cut_parts(image, body)
        parts = {name: _bake(part, _TURN.get(name, 0.0)) for name, part in parts.items()}
        caps = _caps(image, body)
        top, bottom, left, right = _bounds(parts, caps)
        pixels = {name: part.area for name, part in parts.items()}
        _SHARED = (parts, caps, top, bottom, left, right, pixels)
    return _SHARED


def _body_mask(image: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(image.astype(np.int16) - 255, axis=2)
    solid = dist > 18
    count, labels, stats, _cent = cv2.connectedComponentsWithStats(solid.astype(np.uint8), 8)
    if count < 2:
        raise ValueError("Trommeslageren har ingen figur")
    body = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return _fill_holes(body)


def _fill_holes(figure: np.ndarray) -> np.ndarray:
    height, width = figure.shape
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:-1, 1:-1] = figure.astype(np.uint8) * 255
    flood = padded.copy()
    cover = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(flood, cover, (0, 0), 128)
    holes = flood[1:-1, 1:-1] == 0
    return figure | holes


def _cut_parts(image: np.ndarray, body: np.ndarray) -> Dict[str, BodyPart]:
    parts = cut_parts(
        image,
        body,
        _BONES,
        _PIVOT,
        _TIP,
        _SHARES,
        head_y_max=_NECK[1] + 6.0,
        head_reach=70.0,
    )
    missing = [name for name in _PIVOT if name not in parts or parts[name].area < 30]
    if missing:
        raise ValueError(f"Trommeslageren mangler led: {', '.join(missing)}")
    return parts


def _bake(part: BodyPart, clockwise: float) -> BodyPart:
    if abs(clockwise) < 0.5:
        return part
    local = (part.pivot[0] - part.origin[0], part.pivot[1] - part.origin[1])
    sprite, alpha, new_local = _rotate_pivot(part.bgr, part.alpha, local, -clockwise)
    origin = (
        int(round(part.pivot[0] - new_local[0])),
        int(round(part.pivot[1] - new_local[1])),
    )
    return BodyPart(part.name, sprite, alpha, origin, part.pivot, turn(part.tip, part.pivot, clockwise))


def _caps(image: np.ndarray, body: np.ndarray) -> List[JointCap]:
    caps: List[JointCap] = []
    for name in _JOINTS:
        point = _SAMPLE[name]
        caps.append(JointCap(name, _color_at(image, body, point), _radius_at(body, point, _FACTOR[name])))
    return caps


def _color_at(image: np.ndarray, body: np.ndarray, point: Point) -> Tuple[int, int, int]:
    x, y = int(round(point[0])), int(round(point[1]))
    height, width = body.shape
    y0, y1 = max(0, y - 6), min(height, y + 7)
    x0, x1 = max(0, x - 6), min(width, x + 7)
    patch = body[y0:y1, x0:x1]
    pixels = image[y0:y1, x0:x1][patch]
    if len(pixels) == 0:
        return (40, 40, 40)
    color = np.median(pixels, axis=0)
    return int(color[0]), int(color[1]), int(color[2])


def _radius_at(body: np.ndarray, point: Point, factor: float) -> float:
    #22 Bredden er den sammenhængende række, punktet ligger i.
    x, y = int(round(point[0])), int(round(point[1]))
    height, width = body.shape
    y = min(max(y, 0), height - 1)
    xs = np.where(body[y])[0]
    if len(xs) == 0:
        return 12.0
    splits = np.where(np.diff(xs) > 2)[0]
    start = 0
    best = None
    best_dist = 1e9
    chunks = list(splits) + [len(xs) - 1]
    for stop in chunks:
        chunk = xs[start : stop + 1]
        start = stop + 1
        if len(chunk) < 2:
            continue
        center = float(chunk.mean())
        dist = abs(center - x)
        if dist < best_dist:
            best_dist = dist
            best = chunk
    if best is None:
        return 12.0
    return max(8.0, float(best.max() - best.min()) * factor)


def _bounds(parts: Dict[str, BodyPart], caps: List[JointCap]) -> Tuple[int, int, int, int]:
    _angles, anchors = posed_layout(parts, None, 0.0, 0.0, 1.0)
    xs: List[float] = []
    ys: List[float] = []
    for name, part in parts.items():
        anchor = anchors[name]
        tip = limb_tip(part, anchor, 1.0, 0.0)
        xs.extend((anchor[0], tip[0]))
        ys.extend((anchor[1], tip[1]))
        local_x = part.pivot[0] - part.origin[0]
        local_y = part.pivot[1] - part.origin[1]
        height, width = part.alpha.shape[:2]
        for corner_x, corner_y in ((0, 0), (width, 0), (0, height), (width, height)):
            xs.append(anchor[0] + (corner_x - local_x))
            ys.append(anchor[1] + (corner_y - local_y))
    for cap in caps:
        anchor = anchors[cap.part]
        xs.extend((anchor[0] - cap.radius, anchor[0] + cap.radius))
        ys.extend((anchor[1] - cap.radius, anchor[1] + cap.radius))
    return int(min(ys)) - 4, int(max(ys)) + 4, int(min(xs)) - 4, int(max(xs)) + 4


def _draw_caps(
    canvas: np.ndarray,
    parts: Dict[str, BodyPart],
    caps: List[JointCap],
    pose: Optional[BodyPose],
    origin_x: float,
    origin_y: float,
    scale: float,
) -> None:
    _angles, anchors = posed_layout(parts, pose, origin_x, origin_y, scale)
    height, width = canvas.shape[:2]
    for cap in caps:
        anchor = anchors.get(cap.part)
        if anchor is None:
            continue
        center = (int(round(anchor[0])), int(round(anchor[1])))
        radius = max(3, int(round(cap.radius * scale)))
        if center[0] < -radius or center[1] < -radius or center[0] > width + radius or center[1] > height + radius:
            continue
        cv2.circle(canvas, center, radius, cap.color, -1, cv2.LINE_AA)


def _seated(pose: Optional[BodyPose]) -> Optional[BodyPose]:
    #22 Spillerens ben ville rejse figuren. Arme og hoved må godt flytte sig.
    if pose is None:
        return None
    points = list(pose.points)
    for index in _LEGS:
        if index < len(points):
            points[index] = (-1, -1)
    return BodyPose(pose.width, pose.height, tuple(points))
