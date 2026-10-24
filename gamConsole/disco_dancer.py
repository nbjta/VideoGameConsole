"""Disco-danseren klippet op i hals, skulder, albue, håndled, hofte og knæ.

Leddene er målt på assets/just_dance/disco_dancer.png. Udklippet og
drejningen ligger i jointed_figure, samme sted som bokseren og målmanden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from gamConsole.head_tilt import HeadTilt
from gamConsole.jointed_figure import BodyPart, Bone, Point, cut_parts, draw_parts
from gamConsole.pose_body_overlay import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, BodyPose

ROOT = Path(__file__).resolve().parents[1]
DANCER_PATH = ROOT / "assets" / "just_dance" / "disco_dancer.png"

# Målt på disco_dancer.png (554×554). x vokser mod højre, y vokser nedad.
_NECK = (272.0, 146.0)
_HEAD_TOP = (262.0, 64.0)
_SHOULDER_L = (228.0, 180.0)
_ELBOW_L = (196.0, 234.0)
_WRIST_L = (176.0, 274.0)
_HAND_L = (164.0, 300.0)
_SHOULDER_R = (308.0, 156.0)
_ELBOW_R = (348.0, 110.0)
_WRIST_R = (363.0, 76.0)
_HAND_R = (382.0, 36.0)
_HIP_L = (252.0, 322.0)
_KNEE_L = (206.0, 402.0)
_FOOT_L = (162.0, 504.0)
_HIP_R = (308.0, 322.0)
_KNEE_R = (334.0, 392.0)
_FOOT_R = (332.0, 508.0)
_PELVIS = (280.0, 328.0)

_BONES = (
    Bone("head", _NECK, _HEAD_TOP),
    Bone("torso", _NECK, _PELVIS),
    Bone("torso", (214.0, 190.0), (248.0, 270.0)),
    Bone("torso", (300.0, 185.0), (328.0, 275.0)),
    Bone("upper_left", _SHOULDER_L, _ELBOW_L, 30.0),
    Bone("lower_left", _ELBOW_L, _WRIST_L, 22.0, along_max=1.08),
    Bone("hand_left", _WRIST_L, _HAND_L, 20.0, along_max=1.25),
    Bone("upper_right", _SHOULDER_R, _ELBOW_R, 30.0),
    Bone("lower_right", _ELBOW_R, _WRIST_R, 22.0, along_max=1.08),
    Bone("hand_right", _WRIST_R, _HAND_R, 20.0, along_max=1.25),
    Bone("thigh_left", _HIP_L, _KNEE_L, 36.0),
    Bone("shin_left", _KNEE_L, _FOOT_L, 48.0),
    Bone("thigh_right", _HIP_R, _KNEE_R, 36.0),
    Bone("shin_right", _KNEE_R, _FOOT_R, 48.0),
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
    "head": _HEAD_TOP,
}

_SHARES = (
    (_NECK, "head", "torso", 16),
    (_SHOULDER_L, "torso", "upper_left", 20),
    (_SHOULDER_R, "torso", "upper_right", 20),
    (_ELBOW_L, "upper_left", "lower_left", 14),
    (_ELBOW_R, "upper_right", "lower_right", 14),
    (_WRIST_L, "lower_left", "hand_left", 12),
    (_WRIST_R, "lower_right", "hand_right", 12),
    (_HIP_L, "torso", "thigh_left", 18),
    (_HIP_R, "torso", "thigh_right", 18),
    (_KNEE_L, "thigh_left", "shin_left", 16),
    (_KNEE_R, "thigh_right", "shin_right", 16),
)


class SilhouetteShift(HeadTilt):
    """Sidelæns placering følger skulder og hofte. Hovedvippen gør det ikke."""

    def offset_x(self, pose: Optional[BodyPose], width: int) -> int:
        if pose is None or width <= 0 or pose.width <= 0:
            return 0
        torso = _torso_x(pose)
        if torso is None:
            return 0
        return int(round((torso - pose.width / 2.0) / pose.width * width))


def _torso_x(pose: BodyPose) -> Optional[float]:
    shoulders = pose.mid(L_SHOULDER, R_SHOULDER)
    hips = pose.mid(L_HIP, R_HIP)
    if shoulders is None or hips is None:
        return None
    return (shoulders[0] + hips[0]) / 2.0


class DiscoDancer:
    """Figuren samles igen. Hals, håndled, hofte og knæ følger silhuetten."""

    def __init__(self, path: Path = DANCER_PATH) -> None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Disco-danser mangler: {path}")
        kept = _body_mask(image)
        pale = np.all(image > 220, axis=2)
        self.parts: Dict[str, BodyPart] = cut_parts(
            image,
            kept & ~pale,
            _BONES,
            _PIVOT,
            _TIP,
            _SHARES,
            head_y_max=_NECK[1] + 8.0,
            head_reach=62.0,
        )
        # Venstre ben i tegningen peger skråt ud. Det spejles om lodret, så det hænger som det højre.
        _mirror_left_leg(self.parts)
        ys, xs = np.where(kept)
        self.top = int(ys.min())
        self.bottom = int(ys.max()) + 1
        self.left = int(xs.min())
        self.right = int(xs.max()) + 1
        self.cx = (self.left + self.right) / 2.0
        self.shift = SilhouetteShift()

    def draw(self, canvas: np.ndarray, pose: Optional[BodyPose]) -> None:
        height, width = canvas.shape[:2]
        if height < 8 or width < 8:
            return
        feet_y = int(height * 0.90)
        fig_h = max(64, int(height * 0.78))
        scale = fig_h / max(1, self.bottom - self.top)
        center = width // 2 + self.shift.offset_x(pose, width)
        half = max(8, int((self.right - self.left) * scale * 0.35))
        center = int(np.clip(center, half, max(half, width - half)))
        origin_x = center - self.cx * scale
        origin_y = feet_y - self.bottom * scale
        draw_parts(canvas, self.parts, pose, origin_x, origin_y, scale)


def _mirror_left_leg(parts: Dict[str, BodyPart]) -> None:
    """Venstre ben er en spejling af det højre, så de hænger parallelt i hvile."""
    thigh = parts.get("thigh_left")
    other_thigh = parts.get("thigh_right")
    other_shin = parts.get("shin_right")
    if thigh is None or other_thigh is None or other_shin is None:
        return
    parts["thigh_left"] = _mirror_part(other_thigh, thigh.pivot, "thigh_left")
    parts["shin_left"] = _mirror_part(other_shin, parts["thigh_left"].tip, "shin_left")


def _mirror_part(source: BodyPart, pivot: Point, name: str) -> BodyPart:
    bgr = cv2.flip(source.bgr, 1)
    alpha = cv2.flip(source.alpha, 1)
    width = bgr.shape[1]
    local_x = source.pivot[0] - source.origin[0]
    local_y = source.pivot[1] - source.origin[1]
    origin = (pivot[0] - ((width - 1) - local_x), pivot[1] - local_y)
    dx = source.tip[0] - source.pivot[0]
    dy = source.tip[1] - source.pivot[1]
    tip = (pivot[0] - dx, pivot[1] + dy)
    return BodyPart(name, bgr, alpha, origin, pivot, tip)


def _body_mask(image: np.ndarray) -> np.ndarray:
    white = np.all(image > 248, axis=2)
    fg = (~white).astype(np.uint8)
    count, labels, stats, _cent = cv2.connectedComponentsWithStats(fg, 8)
    if count < 2:
        return ~white
    body_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == body_id
