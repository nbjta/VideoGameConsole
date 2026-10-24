#11
"""Målmandens avatar, klippet op i led.

Samme metode som disco-danseren: hver pixel hører til det nærmeste led,
og leddet drejer, så det peger samme vej som spillerens silhuet.
Leddene er målt på assets/maalmand/keeper.png.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from gamConsole.jointed_figure import Bone, cut_parts, draw_parts
from gamConsole.pose_body_overlay import BodyPose

ROOT = Path(__file__).resolve().parents[1]
KEEPER_PATH = ROOT / "assets" / "maalmand" / "keeper.png"

#11 Målt på silhuettens huller i keeper.png. x vokser mod højre, y vokser nedad.
_NECK = (138.0, 184.0)
_HEAD_TOP = (154.0, 12.0)
_SHOULDER_L = (72.0, 224.0)
_ELBOW_L = (34.0, 340.0)
_WRIST_L = (40.0, 430.0)
_HAND_L = (40.0, 455.0)
_SHOULDER_R = (188.0, 248.0)
_ELBOW_R = (320.0, 268.0)
_WRIST_R = (408.0, 274.0)
_HAND_R = (432.0, 300.0)
_HIP_L = (110.0, 430.0)
_KNEE_L = (104.0, 530.0)
_FOOT_L = (101.0, 615.0)
_HIP_R = (198.0, 430.0)
_KNEE_R = (224.0, 530.0)
_FOOT_R = (226.0, 610.0)
_PELVIS = (150.0, 455.0)

_BONES = (
    Bone("head", _NECK, _HEAD_TOP),
    Bone("torso", _NECK, _PELVIS),
    Bone("torso", (95.0, 250.0), (108.0, 450.0)),
    Bone("torso", (175.0, 250.0), (165.0, 430.0)),
    Bone("upper_left", _SHOULDER_L, _ELBOW_L, 28.0),
    Bone("lower_left", _ELBOW_L, _WRIST_L, 22.0, along_max=1.08),
    Bone("hand_left", _WRIST_L, _HAND_L, 24.0, along_max=1.45),
    Bone("upper_right", _SHOULDER_R, _ELBOW_R, 28.0),
    Bone("lower_right", _ELBOW_R, _WRIST_R, 22.0, along_max=1.08),
    Bone("hand_right", _WRIST_R, _HAND_R, 24.0, along_max=1.45),
    Bone("thigh_left", _HIP_L, _KNEE_L, 30.0),
    Bone("shin_left", _KNEE_L, _FOOT_L, 28.0),
    Bone("thigh_right", _HIP_R, _KNEE_R, 30.0),
    Bone("shin_right", _KNEE_R, _FOOT_R, 28.0),
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

#11 Samme skiveradius som disco-danseren, så leddene ikke får et hul.
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


class KeeperAvatar:
    """Hoved, krop, arme, hænder, lår og underben. Hals, håndled, hofte og knæ følger silhuetten."""

    def __init__(self, image_path: Path = KEEPER_PATH) -> None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Målmand-avatar mangler: {image_path}")
        image = _without_banner(image)
        self.parts = _cut_parts(image)
        body = _body_mask(image)
        ys, xs = np.where(body)
        self.top = int(ys.min())
        self.bottom = int(ys.max()) + 1
        self.cx = _PELVIS[0]
        self.pixels = {name: part.area for name, part in self.parts.items()}

    def draw(self, canvas: np.ndarray, pose: Optional[BodyPose], box: Tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = box
        figure_h = max(1, y2 - y1)
        scale = figure_h / max(1, self.bottom - self.top)
        center = (x1 + x2) / 2.0
        #11 Kroppen står i midten af kassen. Hovedvippen ligger allerede i kassens x.
        origin_x = center - self.cx * scale
        origin_y = y2 - self.bottom * scale
        draw_parts(canvas, self.parts, pose, origin_x, origin_y, scale)


def _without_banner(image: np.ndarray) -> np.ndarray:
    #11 Den sorte Alamy-bjælke er ikke en del af figuren.
    row_mean = image.mean(axis=(1, 2))
    dark = np.where(row_mean < 40)[0]
    banner = dark[dark > image.shape[0] * 0.7]
    if len(banner) == 0:
        return image
    return image[: int(banner.min()) - 2]


def _body_mask(image: np.ndarray) -> np.ndarray:
    image = _without_banner(image)
    dist = np.linalg.norm(image.astype(np.int16) - 255, axis=2)
    solid = dist > 30
    count, labels, stats, _cent = cv2.connectedComponentsWithStats(solid.astype(np.uint8), 8)
    if count < 2:
        raise ValueError("Målmand-avataren har ingen figur")
    body = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return _fill_holes(body)


def _fill_holes(figure: np.ndarray) -> np.ndarray:
    #11 Hånden rører kanten, så skjoldets hul lukkes inden for en tom ramme.
    height, width = figure.shape
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:-1, 1:-1] = figure.astype(np.uint8) * 255
    flood = padded.copy()
    cover = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(flood, cover, (0, 0), 128)
    holes = flood[1:-1, 1:-1] == 0
    return figure | holes


def _cut_parts(image: np.ndarray):
    #11 Samme udklip som disco-danseren. Hånden drejer i håndleddet.
    parts = cut_parts(
        image,
        _body_mask(image),
        _BONES,
        _PIVOT,
        _TIP,
        _SHARES,
        head_y_max=_NECK[1] + 8.0,
        head_reach=70.0,
    )
    missing = [name for name in _PIVOT if name not in parts or parts[name].area < 30]
    if missing:
        raise ValueError(f"Målmand-avataren mangler led: {', '.join(missing)}")
    return parts
