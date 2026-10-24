"""Klip spilleren ud og læg den som lag oven på baggrunden."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

HAAR_FACE = "haarcascade_frontalface_default.xml"


class PlayerCutout:
    """Tæt person-lag: ansigt styrer silhuet, GrabCut fjerner væg/luft omkring."""

    def __init__(self) -> None:
        path = getattr(cv2, "data", None)
        cascade_path = f"{path.haarcascades}{HAAR_FACE}" if path is not None else HAAR_FACE
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self._alpha: Optional[np.ndarray] = None
        self._face: Optional[Tuple[int, int, int, int]] = None
        self._frame_i = 0
        self._grab_every = 4
        self._bgd = np.zeros((1, 65), np.float64)
        self._fgd = np.zeros((1, 65), np.float64)

    def composite(self, webcam: np.ndarray, background: np.ndarray) -> np.ndarray:
        if background.shape[:2] != webcam.shape[:2]:
            background = cv2.resize(background, (webcam.shape[1], webcam.shape[0]))
        alpha = self.alpha(webcam)[:, :, None]
        layer = webcam.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
        return layer.astype(np.uint8)

    def head_top_y(self, frame: np.ndarray) -> Optional[int]:
        """Øverste kant af ansigt (y). Større y = hovedet længere nede = bukket."""
        face = self._detect_face(frame)
        if face is None:
            if self._face is not None:
                return self._face[1]
            return None
        self._face = face
        return int(face[1])

    def is_ducked_under(self, frame: np.ndarray, clearance_y: int) -> bool:
        top = self.head_top_y(frame)
        if top is None:
            return True
        return top >= clearance_y - 4

    def alpha(self, frame: np.ndarray) -> np.ndarray:
        self._frame_i += 1
        face = self._detect_face(frame)
        if face is not None:
            self._face = face
        prior = self._silhouette_prior(frame, self._face)
        cut = prior
        if self._frame_i % self._grab_every == 0:
            grab = self._grab(frame, prior)
            if grab is not None:
                cut = grab
        cut = self._tighten(cut)
        if self._alpha is None or self._alpha.shape != cut.shape:
            self._alpha = cut
        else:
            self._alpha = 0.55 * self._alpha + 0.45 * cut
        edge = cv2.GaussianBlur(self._alpha, (5, 5), 0)
        hard = np.clip((edge - 0.35) / 0.40, 0.0, 1.0)
        return hard.astype(np.float32)

    def _detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self._cascade.empty():
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.15, 4, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda b: int(b[2]) * int(b[3]))
        return int(x), int(y), int(fw), int(fh)

    def _silhouette_prior(
        self, frame: np.ndarray, face: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        if face is None:
            cx, cy = w // 2, int(h * 0.42)
            cv2.ellipse(mask, (cx, cy), (int(w * 0.16), int(h * 0.22)), 0, 0, 360, 1.0, -1)
            cv2.ellipse(
                mask,
                (cx, int(h * 0.72)),
                (int(w * 0.18), int(h * 0.28)),
                0,
                0,
                360,
                1.0,
                -1,
            )
            return mask

        x, y, fw, fh = face
        cx = x + fw // 2
        head_cy = y + fh // 2
        cv2.ellipse(
            mask,
            (cx, head_cy),
            (max(8, int(fw * 0.55)), max(8, int(fh * 0.62))),
            0,
            0,
            360,
            1.0,
            -1,
        )
        chin = y + fh
        cv2.ellipse(
            mask,
            (cx, chin + int(fh * 0.85)),
            (max(10, int(fw * 1.05)), max(12, int(fh * 1.35))),
            0,
            0,
            360,
            1.0,
            -1,
        )
        torso_w = max(12, int(fw * 1.7))
        cv2.rectangle(
            mask,
            (max(0, cx - torso_w // 2), chin + int(fh * 0.35)),
            (min(w, cx + torso_w // 2), h),
            1.0,
            -1,
        )
        skin = self._skin_near_body(frame, face, mask)
        return np.maximum(mask, skin)

    def _skin_near_body(
        self,
        frame: np.ndarray,
        face: Tuple[int, int, int, int],
        body: np.ndarray,
    ) -> np.ndarray:
        x, y, fw, fh = face
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        generic = cv2.inRange(ycrcb, (40, 133, 77), (255, 173, 127))
        y0, y1 = y + fh // 5, y + (4 * fh) // 5
        x0, x1 = x + fw // 5, x + (4 * fw) // 5
        inner = ycrcb[max(0, y0) : y1, max(0, x0) : x1]
        if inner.size > 0:
            mean = inner.reshape(-1, 3).mean(axis=0)
            lo = np.clip(mean - np.array([50.0, 20.0, 20.0]), 0, 255).astype(np.uint8)
            hi = np.clip(mean + np.array([50.0, 20.0, 20.0]), 0, 255).astype(np.uint8)
            sampled = cv2.inRange(ycrcb, lo, hi)
            generic = cv2.bitwise_or(generic, sampled)
        skin = (generic > 0).astype(np.float32)
        near = cv2.dilate(body, np.ones((max(3, fh // 3), max(3, fw // 2)), np.uint8))
        return skin * near

    def _grab(self, frame: np.ndarray, prior: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        scale = min(1.0, 360.0 / max(w, 1))
        small_w, small_h = max(32, int(w * scale)), max(32, int(h * scale))
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        prior_s = cv2.resize(prior, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gc = np.full((small_h, small_w), cv2.GC_BGD, dtype=np.uint8)
        gc[prior_s > 0.12] = cv2.GC_PR_FGD
        gc[prior_s > 0.75] = cv2.GC_FGD
        gc[:6, :] = cv2.GC_BGD
        gc[:, :6] = cv2.GC_BGD
        gc[:, -6:] = cv2.GC_BGD
        try:
            cv2.grabCut(small, gc, None, self._bgd, self._fgd, 3, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return None
        binary = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
        return cv2.resize(binary, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _tighten(mask: np.ndarray) -> np.ndarray:
        bin8 = (mask > 0.5).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        bin8 = cv2.morphologyEx(bin8, cv2.MORPH_OPEN, k, iterations=1)
        bin8 = cv2.morphologyEx(bin8, cv2.MORPH_CLOSE, k, iterations=1)
        bin8 = cv2.erode(bin8, k, iterations=1)
        return bin8.astype(np.float32)
