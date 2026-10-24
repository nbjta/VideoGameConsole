"""Disco-scene med danseren klippet op i led. Kameraet bruges kun til posituren."""

from __future__ import annotations

from typing import Optional

import numpy as np

from gamConsole.disco_backdrop import DiscoBackdrop
from gamConsole.disco_dancer import DiscoDancer
from gamConsole.pose_body_overlay import BodyPose, PoseBodyOverlay


class DanceAvatar:
    """Disco-figuren. Hals, skulder, albue og knæ følger spillerens silhuet."""

    def __init__(self) -> None:
        self._pose = PoseBodyOverlay()
        self.dancer = DiscoDancer()

    def close(self) -> None:
        self._pose.close()

    def snapshot(self, webcam: np.ndarray) -> Optional[BodyPose]:
        return self._pose.snapshot(webcam)

    def draw(self, canvas: np.ndarray, pose: Optional[BodyPose]) -> None:
        self.dancer.draw(canvas, pose)


class JustDanceStage:
    def __init__(self) -> None:
        self.backdrop = DiscoBackdrop()
        self.avatar = DanceAvatar()

    def close(self) -> None:
        self.avatar.close()

    def composite(self, webcam: np.ndarray) -> np.ndarray:
        height, width = webcam.shape[:2]
        self.backdrop.tick()
        scene = self.backdrop.render(width, height)
        self.avatar.draw(scene, self.avatar.snapshot(webcam))
        return scene
