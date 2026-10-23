#1
"""AR-spiller: leder bagfra + krop som punkter — uden webcam-billede."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2

from gamConsole.adventure_city import _road_layout
from gamConsole.leader_back_view import draw_leader_back, draw_leader_synced
from gamConsole.pose_body_overlay import BodyPose, PoseBodyOverlay


class ARPlayerView:
    """Scenen viser baggrund, leder-avatar og dit kropsskelet."""

    def __init__(self) -> None:
        self._pose = PoseBodyOverlay()
        self._last_pose: Optional[BodyPose] = None
        self._smooth_blend = 0.0

    def close(self) -> None:
        self._pose.close()

    def refresh_pose(self, webcam: np.ndarray) -> None:
        """Kaldes før gren-logik så bukning matcher samme frame."""
        self._last_pose = self._pose.snapshot(webcam)

    def composite(self, webcam: np.ndarray, background: np.ndarray) -> np.ndarray:
        if background.shape[:2] != webcam.shape[:2]:
            background = cv2.resize(background, (webcam.shape[1], webcam.shape[0]))
        pose = self._last_pose
        if pose is None:
            pose = self._pose.snapshot(webcam)
            self._last_pose = pose
        h, w = webcam.shape[:2]
        scene = background.copy()
        if pose is not None:
            self._smooth_blend = min(1.0, self._smooth_blend + 0.22)
            draw_leader_synced(scene, pose, blend=self._smooth_blend)
        else:
            self._smooth_blend = max(0.0, self._smooth_blend - 0.15)
            _horizon, road_top, vx, _vy, _hh, _hn = _road_layout(h, w)
            scale = h / 480.0
            draw_leader_back(scene, vx, road_top - int(6 * scale), scale=scale * 1.05)
        self._pose.draw(scene, pose)
        return scene

    def is_ducked_under(self, frame: np.ndarray, clearance_y: int) -> bool:
        del frame
        return self._pose.is_ducked_under(self._last_pose, clearance_y)
