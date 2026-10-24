#12
"""Bokserens avatar, klippet op i led.

Samme metode som disco-danseren og målmanden: hver pixel hører til det
nærmeste led, og leddet drejer, så det peger samme vej som spillerens
silhuet. Leddene er målt på assets/boksning/boxer.png.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from gamConsole.jointed_figure import (
    Bone,
    Point,
    cut_parts,
    draw_parts,
    limb_tip,
    posed_layout,
)
from gamConsole.pose_body_overlay import BodyPose

ROOT = Path(__file__).resolve().parents[1]
BOXER_PATH = ROOT / "assets" / "boksning" / "boxer.png"

#12 Målt på silhuetten i boxer.png (980×980). x vokser mod højre, y vokser nedad.
_NECK = (490.0, 216.0)
_HEAD_TOP = (490.0, 70.0)
_SHOULDER_L = (360.0, 300.0)
_ELBOW_L = (330.0, 392.0)
_WRIST_L = (288.0, 304.0)
_HAND_L = (267.0, 258.0)
_SHOULDER_R = (620.0, 300.0)
_ELBOW_R = (649.0, 392.0)
_WRIST_R = (692.0, 304.0)
_HAND_R = (712.0, 258.0)
_HIP_L = (446.0, 590.0)
_KNEE_L = (433.0, 745.0)
_FOOT_L = (435.0, 901.0)
_HIP_R = (539.0, 590.0)
_KNEE_R = (547.0, 745.0)
_FOOT_R = (549.0, 901.0)
_PELVIS = (490.0, 530.0)

_BONES = (
    Bone("head", _NECK, _HEAD_TOP),
    Bone("torso", _NECK, _PELVIS),
    Bone("torso", (440.0, 280.0), (430.0, 520.0)),
    Bone("torso", (540.0, 280.0), (550.0, 520.0)),
    Bone("upper_left", _SHOULDER_L, _ELBOW_L, 56.0, along_max=1.40),
    Bone("lower_left", _ELBOW_L, _WRIST_L, 28.0, along_max=1.08),
    Bone("hand_left", _WRIST_L, _HAND_L, 48.0, along_max=1.55),
    Bone("upper_right", _SHOULDER_R, _ELBOW_R, 56.0, along_max=1.40),
    Bone("lower_right", _ELBOW_R, _WRIST_R, 28.0, along_max=1.08),
    Bone("hand_right", _WRIST_R, _HAND_R, 48.0, along_max=1.55),
    Bone("thigh_left", _HIP_L, _KNEE_L, 52.0, along_max=1.40),
    Bone("shin_left", _KNEE_L, _FOOT_L, 72.0, along_max=1.40),
    Bone("thigh_right", _HIP_R, _KNEE_R, 52.0, along_max=1.40),
    Bone("shin_right", _KNEE_R, _FOOT_R, 72.0, along_max=1.40),
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

#12 Handskens kant ligger forbi manchetten, så hånden rækker længere end underarmen.
_SHARES = (
    (_NECK, "head", "torso", 28),
    (_SHOULDER_L, "torso", "upper_left", 52),
    (_SHOULDER_R, "torso", "upper_right", 52),
    (_ELBOW_L, "upper_left", "lower_left", 24),
    (_ELBOW_R, "upper_right", "lower_right", 24),
    (_WRIST_L, "lower_left", "hand_left", 22),
    (_WRIST_R, "lower_right", "hand_right", 22),
    (_HIP_L, "torso", "thigh_left", 32),
    (_HIP_R, "torso", "thigh_right", 32),
    (_KNEE_L, "thigh_left", "shin_left", 28),
    (_KNEE_R, "thigh_right", "shin_right", 28),
)


class BoxerAvatar:
    """Hoved, krop, arme, hænder, lår og underben. Hals, håndled, hofte og knæ følger silhuetten."""

    def __init__(self, image_path: Path = BOXER_PATH) -> None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Bokser-avatar mangler: {image_path}")
        self.parts = _cut_parts(image)
        body = _body_mask(image)
        ys, xs = np.where(body)
        self.top = int(ys.min())
        self.bottom = int(ys.max()) + 1
        self.left = int(xs.min())
        self.right = int(xs.max()) + 1
        self.cx = _PELVIS[0]
        #12 Knælinjen. Det der ligger under den, skæres af når skærmen er for lav.
        self.knee_y = (_KNEE_L[1] + _KNEE_R[1]) / 2.0
        self.pixels = {name: part.area for name, part in self.parts.items()}

    def draw(self, canvas: np.ndarray, pose: Optional[BodyPose], box: Tuple[int, int, int, int]) -> None:
        origin_x, origin_y, scale = self.placement(box)
        draw_parts(canvas, self.parts, pose, origin_x, origin_y, scale)

    def glove_tips(self, pose: Optional[BodyPose], box: Tuple[int, int, int, int]) -> Tuple[Point, Point]:
        """Handskernes spids i billedet, efter håndleddet er drejet."""
        origin_x, origin_y, scale = self.placement(box)
        angles, anchors = posed_layout(self.parts, pose, origin_x, origin_y, scale)
        return (
            limb_tip(self.parts["hand_left"], anchors["hand_left"], scale, angles.get("hand_left", 0.0)),
            limb_tip(self.parts["hand_right"], anchors["hand_right"], scale, angles.get("hand_right", 0.0)),
        )

    def placement(self, box: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
        x1, _y1, x2, y2 = box
        figure_h = max(1, y2 - _y1)
        scale = figure_h / max(1, self.bottom - self.top)
        center = (x1 + x2) / 2.0
        #12 Kroppen står i midten af kassen. Hovedvippen ligger allerede i kassens x.
        origin_x = center - self.cx * scale
        origin_y = y2 - self.bottom * scale
        return origin_x, origin_y, scale


def _body_mask(image: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(image.astype(np.int16) - 255, axis=2)
    solid = dist > 18
    count, labels, stats, _cent = cv2.connectedComponentsWithStats(solid.astype(np.uint8), 8)
    if count < 2:
        raise ValueError("Bokser-avataren har ingen figur")
    body = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return _fill_holes(body)


def _fill_holes(figure: np.ndarray) -> np.ndarray:
    #12 Huller inde i figuren lukkes. Mellemrummet mellem benene rører kanten og bliver stående.
    height, width = figure.shape
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:-1, 1:-1] = figure.astype(np.uint8) * 255
    flood = padded.copy()
    cover = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(flood, cover, (0, 0), 128)
    holes = flood[1:-1, 1:-1] == 0
    return figure | holes


def _cut_parts(image: np.ndarray):
    #12 Samme udklip som disco-danseren. Handsken er hånden, ikke underarmen.
    parts = cut_parts(
        image,
        _body_mask(image),
        _BONES,
        _PIVOT,
        _TIP,
        _SHARES,
        head_y_max=_NECK[1] + 8.0,
        head_reach=80.0,
    )
    missing = [name for name in _PIVOT if name not in parts or parts[name].area < 30]
    if missing:
        raise ValueError(f"Bokser-avataren mangler led: {', '.join(missing)}")
    return parts
