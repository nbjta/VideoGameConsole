"""Just Dance i samme vindue (ét frame ad gangen)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor" / "just-dance"


class JustDanceSession:
    title = "Just Dance"

    def __init__(self, song: str, camera: int) -> None:
        self.song = song
        self.camera = camera
        self._game = None
        self._counter = 0
        self._ready = False
        self._error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not (REPO / "just_dance_main.py").is_file():
            self._error = "Kør ./scripts/setup_just_dance.sh"
            return
        os.chdir(REPO)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        try:
            from just_dance_main import JustDanceGame

            self._game = JustDanceGame(
                "model/model.tflite",
                f"songs/{self.song}.mp4",
                self.camera,
            )
            self._game.controller.play_sound(f"songs_audio/{self.song}.mp3")
            self.title = f"Just Dance — {self.song}"
            self._ready = True
        except Exception as exc:  # noqa: BLE001 — vis fejl i menu
            self._error = str(exc)

    def finalize_score(self) -> Optional[int]:
        if self._game is None:
            return None
        try:
            self._game.calculate_final_score()
            self._game.store_leaderboard("leaderboard.csv")
            return int(self._game.score)
        except Exception:
            return None

    def stop(self) -> None:
        if self._game is None:
            return
        try:
            self._game.controller.release_capture()
        except Exception:
            pass

    def on_key(self, key: int) -> None:
        return

    def tick(self) -> Optional[np.ndarray]:
        if self._error:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, self._error, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 120), 2)
            return frame
        if not self._ready or self._game is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Loader Just Dance / TensorFlow…", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return frame

        ctrl = self._game.controller
        if not ctrl.cap1.isOpened():
            return None
        start = time.time()
        ok1, frame1 = ctrl.cap1.read()
        ok2, frame2 = ctrl.cap2.read()
        if not ok1 or frame1 is None:
            return None
        if not ok2 or frame2 is None:
            frame2 = np.zeros_like(frame1)
        frame2 = cv2.flip(frame2, 1)

        if self._counter == 0:
            kpv = ctrl.process_frame(frame1)
            kpc = ctrl.process_frame(frame2)
            ctrl.model.store_angles(ctrl.angles_video, frame2, kpv)
            ctrl.model.store_angles(ctrl.angles_camera, frame2, kpc)
        self._counter = (self._counter + 1) % 100

        h1 = frame1.shape[0]
        h2, w2 = frame2.shape[:2]
        if h1 != h2:
            w2 = int(w2 * (h1 / h2))
            frame2 = cv2.resize(frame2, (w2, h1))
        combined = np.concatenate((frame1, frame2), axis=1)

        elapsed = time.time() - start
        rate = ctrl.frame1_rate or 30.0
        delay = max(0.0, (1.0 / rate) - elapsed)
        time.sleep(delay)
        return combined
