#1
"""AR-spiller: leder bagfra + krop som punkter — uden webcam-billede."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2

from gamConsole.adventure_city import _road_layout
from gamConsole.head_tilt import HeadTilt
from gamConsole.leader_back_view import draw_leader_back, player_car_box
from gamConsole.pose_body_overlay import BodyPose, PoseBodyOverlay


class ARPlayerView:
    """Scenen viser baggrund, leder-avatar og dit kropsskelet."""

    def __init__(self) -> None:
        self._pose = PoseBodyOverlay()
        self._last_pose: Optional[BodyPose] = None

    def close(self) -> None:
        self._pose.close()

    def refresh_pose(self, webcam: np.ndarray) -> None:
        """#1 Læser posen, før bilen og armene tegnes på samme billede."""
        self._last_pose = self._pose.snapshot(webcam)

    def last_pose(self) -> Optional[BodyPose]:
        """#1 Posen fra dette billede. Kastet læser den, før burgeren slippes."""
        return self._last_pose

    def composite(self, webcam: np.ndarray, background: np.ndarray) -> np.ndarray:
        if background.shape[:2] != webcam.shape[:2]:
            background = cv2.resize(background, (webcam.shape[1], webcam.shape[0]))
        pose = self._last_pose
        if pose is None:
            pose = self._pose.snapshot(webcam)
            self._last_pose = pose
        h, w = webcam.shape[:2]
        scene = background.copy()
        #1 Står nederst på vejen. Vip til højre flytter lederen til højre.
        draw_leader_back(scene, self.feet_x(w, h), h - 1, scale=self.leader_scale(h), pose=pose)
        #1 Punktsilhuetten bliver som debug oven på scenen.
        self._pose.draw(scene, pose)
        return scene

    def leader_scale(self, height: int) -> float:
        """#1 Samme størrelse som figuren tegnes i."""
        return (height / 480.0) * 1.05

    def feet_x(self, width: int, height: int) -> int:
        """#1 Vip til højre flytter lederen til højre."""
        _horizon, _road_top, vx, _vy, _hh, _hn = _road_layout(height, width)
        return vx + HeadTilt().offset_x(self._last_pose, width)

    def car_box(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """#1 Den blå bil, som en modkørende bil kan ramme."""
        return player_car_box(width, height, self.feet_x(width, height), self.leader_scale(height))

    def is_ducked_under(self, frame: np.ndarray, clearance_y: int) -> bool:
        del frame
        return self._pose.is_ducked_under(self._last_pose, clearance_y)
