#1
"""Krop som punkter (MediaPipe Pose) — samme stil som hånd-skelet i Husbygning."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

JOINT = (255, 255, 255)
BONE = (255, 255, 0)
Point = Tuple[int, int]

# MediaPipe pose index
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


@dataclass
class BodyPose:
    """Landmarks i pixel-koordinater (-1,-1) = usynlig."""

    width: int
    height: int
    points: Tuple[Point, ...]

    def get(self, index: int) -> Optional[Point]:
        if index >= len(self.points):
            return None
        p = self.points[index]
        if p[0] < 0:
            return None
        return p

    def mid(self, a: int, b: int) -> Optional[Tuple[float, float]]:
        pa, pb = self.get(a), self.get(b)
        if pa is None or pb is None:
            return None
        return (pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5

    def head_top_y(self) -> Optional[int]:
        ys: List[int] = []
        for i in (NOSE, 1, 2, 3, 4, 5, 6, 7, 8):
            p = self.get(i)
            if p is not None:
                ys.append(p[1])
        if not ys:
            sh = self.mid(L_SHOULDER, R_SHOULDER)
            if sh is None:
                return None
            return int(sh[1] - self.torso_height() * 0.45)
        return min(ys)

    def torso_height(self) -> float:
        sh = self.mid(L_SHOULDER, R_SHOULDER)
        hp = self.mid(L_HIP, R_HIP)
        if sh is None or hp is None:
            return 120.0
        return max(40.0, float(np.hypot(hp[0] - sh[0], hp[1] - sh[1])))


class PoseBodyOverlay:
    """Tegner kropsskelet som prikker og linjer oven på scenen — ikke webcam."""

    def __init__(self) -> None:
        self._landmarker = None
        self._connections: Sequence = ()
        self._init_error: Optional[str] = None
        self._try_init()

    def _try_init(self) -> None:
        try:
            from mediapipe import Image, ImageFormat
            from mediapipe.tasks.python.core import base_options as base_options_module
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
            from mediapipe.tasks.python.vision import PoseLandmarksConnections
            from mediapipe.tasks.python.vision.core import vision_task_running_mode

            model = _ensure_model_file()
            options = PoseLandmarkerOptions(
                base_options=base_options_module.BaseOptions(model_asset_path=str(model)),
                running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
            self._connections = PoseLandmarksConnections.POSE_LANDMARKS
            self._mp_image = Image
            self._mp_format = ImageFormat
        except Exception as exc:
            self._landmarker = None
            self._init_error = str(exc)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def snapshot(self, camera: np.ndarray) -> Optional[BodyPose]:
        if self._landmarker is None:
            return None
        result = self._detect(camera)
        if result is None or not result.pose_landmarks:
            return None
        h, w = camera.shape[:2]
        lm = result.pose_landmarks[0]
        pts: List[Point] = []
        for p in lm:
            if p.visibility is not None and p.visibility < 0.35:
                pts.append((-1, -1))
            else:
                pts.append((int(p.x * w), int(p.y * h)))
        return BodyPose(w, h, tuple(pts))

    def draw(self, canvas: np.ndarray, pose: Optional[BodyPose]) -> None:
        if pose is None:
            return
        pts = pose.points
        for conn in self._connections:
            a, b = pts[conn.start], pts[conn.end]
            if a[0] < 0 or b[0] < 0:
                continue
            cv2.line(canvas, a, b, BONE, 3, cv2.LINE_AA)
        for px, py in pts:
            if px < 0:
                continue
            cv2.circle(canvas, (px, py), 6, JOINT, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 4, BONE, -1, cv2.LINE_AA)

    def is_ducked_under(self, pose: Optional[BodyPose], clearance_y: int) -> bool:
        if pose is None:
            return True
        top = pose.head_top_y()
        if top is None:
            return True
        return top >= clearance_y - 4

    def _detect(self, frame: np.ndarray):
        detect_w = 480
        h, w = frame.shape[:2]
        scale = min(1.0, detect_w / max(w, 1))
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        mp_image = self._mp_image(image_format=self._mp_format.SRGB, data=rgb)
        return self._landmarker.detect(mp_image)


def _ensure_model_file() -> Path:
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1000:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if not MODEL_PATH.is_file():
        raise OSError(f"Kunne ikke hente pose_landmarker_lite.task til {MODEL_PATH}")
    return MODEL_PATH
