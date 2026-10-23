"""Webcam-spil i Python (fallback når CvGames ikke er bygget)."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gamConsole.hand_pip_overlay import HandPipOverlay
from gamConsole.house_build import HouseBuildScene
from gamConsole.opencv_ui import run_window_loop


def _open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Kunne ikke åbne kamera {index}", file=sys.stderr)
        sys.exit(1)
    return cap


def _motion_mask(gray: np.ndarray, prev: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if prev is None:
        return np.zeros_like(gray), gray
    diff = cv2.absdiff(gray, prev)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask, gray


def _motion_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    m = cv2.moments(mask)
    if m["m00"] < 5000:
        return None
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def run_debug(camera: int) -> None:
    cap = _open_camera(camera)
    state: dict = {"prev": None}

    def render():
        ok, frame = cap.read()
        if not ok:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        mask, state["prev"] = _motion_mask(gray, state["prev"])
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([frame, mask_bgr])
        cv2.putText(combined, "Bevægelse = højre billede", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return combined

    run_window_loop("gamConsole: debug", cap, render)


def run_bub(camera: int) -> None:
    cap = _open_camera(camera)
    state: dict = {"prev": None, "bubbles": []}
    rng = np.random.default_rng(42)

    def render():
        ok, frame = cap.read()
        if not ok:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        mask, state["prev"] = _motion_mask(gray, state["prev"])
        c = _motion_centroid(mask)
        if c is not None:
            color = tuple(int(x) for x in rng.integers(80, 255, size=3))
            state["bubbles"].append((float(c[0]), float(c[1]), 18.0, color))
        alive = []
        for x, y, r, col in state["bubbles"]:
            r += 0.6
            if r < 120:
                alive.append((x, y, r, col))
                cv2.circle(frame, (int(x), int(y)), int(r), col, 2)
        state["bubbles"] = alive
        cv2.putText(frame, "Bevæg dig for bobler", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    run_window_loop("gamConsole: bubble shot", cap, render)


#8
def run_3d(camera: int) -> None:
    cap = _open_camera(camera)
    scene = HouseBuildScene()
    hand_pip = HandPipOverlay()

    def render():
        ok, frame = cap.read()
        if not ok:
            return None
        #8 Spejlvendt webcam — samme styring som i gamConsole-menuen.
        camera = cv2.flip(frame, 1)
        #8 Bjælken flytter sig kun mens hånden er klemt.
        track = hand_pip.read(camera)
        if track is not None:
            h, w = camera.shape[:2]
            px, py = track.palm_norm
            scene.steer(
                camera.shape,
                (int(px * w), int(py * h)),
                holding=track.clenched,
                landmarks=track.points,
            )
        else:
            scene.steer(camera.shape, None)
        scene.draw(frame)
        hand_pip.draw(frame, camera=camera)
        return frame

    try:
        run_window_loop("gamConsole: husbygning", cap, render)
    finally:
        hand_pip.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="gamConsole builtin webcam game")
    parser.add_argument("mode", choices=("3d", "bub", "debug"))
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    runners = {"3d": run_3d, "bub": run_bub, "debug": run_debug}
    runners[args.mode](args.camera)


if __name__ == "__main__":
    main()
