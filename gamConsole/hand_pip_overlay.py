#8
"""Lille hånd-preview (PiP) med skelet — som Handora (MediaPipe Tasks)."""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

PIP_MARGIN = 10
PIP_BOTTOM_INSET = 48
SKELETON = (255, 255, 0)
JOINT = (255, 255, 255)
Point = Tuple[float, float]


@dataclass(frozen=True)
class HandTrack:
    """21 normaliserede punkter og om hånden er klemt sammen."""

    points: Tuple[Point, ...]
    clenched: bool

    @property
    def palm_norm(self) -> Point:
        ids = (0, 5, 9, 13, 17)
        xs = [self.points[i][0] for i in ids]
        ys = [self.points[i][1] for i in ids]
        return sum(xs) / len(xs), sum(ys) / len(ys)


#8
def hand_is_clenched(points: Sequence[Point]) -> bool:
    """Knytnæve eller knib: fingrene er inde, eller tommel og pegefinger mødes."""
    if len(points) < 21:
        return False
    palm = math.hypot(points[9][0] - points[0][0], points[9][1] - points[0][1])
    if palm < 1e-4:
        return False
    curled = 0
    for tip in (8, 12, 16, 20):
        reach = math.hypot(points[tip][0] - points[0][0], points[tip][1] - points[0][1])
        if reach < palm * 1.35:
            curled += 1
    pinch = math.hypot(points[4][0] - points[8][0], points[4][1] - points[8][1]) < palm * 0.45
    return curled >= 3 or pinch


class HandPipOverlay:
    """Nederste højre hjørne: webcam + MediaPipe-hånd-skelet."""

    def __init__(self) -> None:
        self._landmarker = None
        self._connections: Sequence = ()
        self._init_error: Optional[str] = None
        self._last_result = None
        self._has_cache = False
        self._try_init()

    def _try_init(self) -> None:
        try:
            from mediapipe import Image, ImageFormat
            from mediapipe.tasks.python.core import base_options as base_options_module
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
            from mediapipe.tasks.python.vision import HandLandmarksConnections
            from mediapipe.tasks.python.vision.core import vision_task_running_mode

            model = _ensure_model_file()
            options = HandLandmarkerOptions(
                base_options=base_options_module.BaseOptions(model_asset_path=str(model)),
                running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
            self._connections = HandLandmarksConnections.HAND_CONNECTIONS
            self._mp_image = Image
            self._mp_format = ImageFormat
        except Exception as exc:
            self._landmarker = None
            self._init_error = str(exc)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def draw(self, frame: np.ndarray, *, camera: Optional[np.ndarray] = None) -> None:
        """Tegn PiP på frame. Brug camera=webcam hvis canvas ikke er kamerabilledet."""
        source = camera if camera is not None else frame
        h, w = frame.shape[:2]
        pip_w = max(168, int(w * 0.26))
        pip_h = max(126, int(h * 0.22))
        x2 = w - PIP_MARGIN
        y2 = h - PIP_BOTTOM_INSET - PIP_MARGIN
        x1 = x2 - pip_w
        y1 = y2 - pip_h
        if x1 < 0 or y1 < 0:
            return

        pip = cv2.resize(source, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
        label = "Vis hånden"
        if self._landmarker is not None:
            if self._has_cache:
                result = self._last_result
                self._has_cache = False
            else:
                result = self._detect(source)
            if result is not None and result.hand_landmarks:
                for idx, landmarks in enumerate(result.hand_landmarks):
                    if result.handedness and idx < len(result.handedness):
                        cats = result.handedness[idx]
                        if cats:
                            label = cats[0].category_name or cats[0].display_name or "Hand"
                    self._draw_skeleton(pip, landmarks, pip_w, pip_h)
            elif self._init_error is None:
                label = "Vis hånden"
        elif self._init_error:
            cv2.putText(
                pip,
                "Hand-tracking utilgængelig",
                (6, pip_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (160, 160, 180),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(pip, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        frame[y1:y2, x1:x2] = pip
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 190), 2, cv2.LINE_AA)

    def read(self, frame: np.ndarray) -> Optional[HandTrack]:
        """#8 Ét detektionskald. draw() bruger samme resultat bagefter."""
        if self._landmarker is None:
            self._last_result = None
            self._has_cache = False
            return None
        result = self._detect(frame)
        self._last_result = result
        self._has_cache = True
        if result is None or not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]
        points = tuple((float(p.x), float(p.y)) for p in lm)
        return HandTrack(points=points, clenched=hand_is_clenched(points))

    def palm_centroid_norm(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        track = self.read(frame)
        if track is None:
            return None
        return track.palm_norm

    def _detect(self, frame: np.ndarray):
        detect_w = 640
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

    def _draw_skeleton(self, pip: np.ndarray, landmarks, pip_w: int, pip_h: int) -> None:
        pts: List[Tuple[int, int]] = []
        for lm in landmarks:
            pts.append((int(lm.x * pip_w), int(lm.y * pip_h)))
        for conn in self._connections:
            cv2.line(pip, pts[conn.start], pts[conn.end], SKELETON, 3, cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(pip, (px, py), 5, JOINT, -1, cv2.LINE_AA)
            cv2.circle(pip, (px, py), 3, SKELETON, -1, cv2.LINE_AA)


def _ensure_model_file() -> Path:
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1000:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if not MODEL_PATH.is_file():
        raise OSError(f"Kunne ikke hente hand_landmarker.task til {MODEL_PATH}")
    return MODEL_PATH
