"""Webcam-spil i Python (fallback når CvGames ikke er bygget)."""

from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gamConsole.opencv_ui import run_window_loop

DATA = Path(__file__).resolve().parents[1] / "vendor" / "webcam-games" / "Data"


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


def _load_vj_images() -> List[np.ndarray]:
    paths: List[str] = []
    if DATA.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(glob.glob(str(DATA / ext)))
    images: List[np.ndarray] = []
    for p in sorted(paths)[:12]:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)
    if not images:
        for hue in range(0, 360, 45):
            block = np.zeros((480, 640, 3), dtype=np.uint8)
            block[:] = cv2.cvtColor(np.uint8([[[hue // 2, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
            images.append(block)
    return images


def run_vj(camera: int) -> None:
    cap = _open_camera(camera)
    state: dict = {"prev": None, "idx": 0, "blend": 0.0, "images": _load_vj_images()}

    def render():
        ok, frame = cap.read()
        if not ok:
            return None
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        mask, state["prev"] = _motion_mask(gray, state["prev"])
        motion = float(np.count_nonzero(mask)) / mask.size
        if motion > 0.02:
            state["blend"] = min(1.0, state["blend"] + 0.04)
        else:
            state["blend"] = max(0.0, state["blend"] - 0.01)
        if state["blend"] >= 0.98:
            state["idx"] = (state["idx"] + 1) % len(state["images"])
            state["blend"] = 0.0
        overlay = cv2.resize(state["images"][state["idx"]], (w, h))
        out = cv2.addWeighted(frame, 1.0 - state["blend"], overlay, state["blend"], 0)
        cv2.putText(
            out,
            f"VJ lag {state['idx'] + 1}/{len(state['images'])}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return out

    run_window_loop("gamConsole: vj", cap, render)


def _draw_cube(frame: np.ndarray, angle_x: float, angle_y: float) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    s = min(w, h) // 6
    pts = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=float,
    )
    rx, ry = angle_x, angle_y
    rot_x = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    rot_y = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    projected = []
    for p in pts:
        v = rot_y @ rot_x @ (p * s)
        projected.append((int(cx + v[0]), int(cy + v[1])))
    for a, b in edges:
        cv2.line(frame, projected[a], projected[b], (80, 180, 255), 2)


def run_3d(camera: int) -> None:
    cap = _open_camera(camera)
    state: dict = {"prev": None, "ax": 0.4, "ay": 0.4}

    def render():
        ok, frame = cap.read()
        if not ok:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        mask, state["prev"] = _motion_mask(gray, state["prev"])
        c = _motion_centroid(mask)
        if c is not None:
            h, w = frame.shape[:2]
            state["ay"] = (c[0] / w - 0.5) * 3
            state["ax"] = (c[1] / h - 0.5) * 3
        _draw_cube(frame, state["ax"], state["ay"])
        cv2.putText(frame, "Styr kube med bevægelse", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    run_window_loop("gamConsole: 3d", cap, render)


def main() -> None:
    parser = argparse.ArgumentParser(description="gamConsole builtin webcam game")
    parser.add_argument("mode", choices=("3d", "bub", "vj", "debug"))
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    runners = {"3d": run_3d, "bub": run_bub, "vj": run_vj, "debug": run_debug}
    runners[args.mode](args.camera)


if __name__ == "__main__":
    main()
