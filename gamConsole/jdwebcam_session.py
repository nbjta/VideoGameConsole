"""JDWebcam-info i samme vindue (fuld controller kører stadig via subprocess)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class JDWebcamSession:
    title = "JDWebcam"

    def __init__(self, camera: int) -> None:
        self.cap = cv2.VideoCapture(camera)
        self._started = False

    def stop(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

    def on_key(self, key: int) -> None:
        return

    def tick(self) -> Optional[np.ndarray]:
        if not self.cap.isOpened():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Kamera utilgængeligt", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            return frame
        ok, frame = self.cap.read()
        if not ok:
            return None
        cv2.putText(
            frame,
            "JDWebcam: forbinder til Just Dance 2017 PC i baggrunden",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Brug Hjem for at returnere til menuen",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame
