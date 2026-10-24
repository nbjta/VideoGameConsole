"""Point, når spillerens trin matcher dansen på sangens tidspunkt.

Sammenligningen bruger de vinkler, vendor-modellen allerede gemmer for
videobilledet og kameraet i samme øjeblik. Grænsen er 20 grader, samme
tal som calculate_final_score i vendor-spillet. Et led tæller kun med,
når forskellen er under den grænse. Scoren, highscoren og det stigende
tal er de samme klasser som Gurli Gris og Humdingers.
"""

from __future__ import annotations

import time
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from gamConsole.game_audio import game_audio
from gamConsole.round_score import HighScoreList, RoundScore

JOINTS = (
    "left_arm",
    "right_arm",
    "left_elbow",
    "right_elbow",
    "left_thigh",
    "right_thigh",
    "left_leg",
    "right_leg",
)
# Samme grænse som vendor just_dance_main.JustDanceGame.calculate_final_score.
MOVE_THRESHOLD = 20
POINTS_PER_JOINT = 10


def points_for_latest(
    video_angles: Mapping[str, Sequence[float]],
    camera_angles: Mapping[str, Sequence[float]],
    threshold: float = MOVE_THRESHOLD,
) -> int:
    """Point for det seneste målte øjeblik. Ældre målinger tæller ikke med igen."""
    earned = 0
    for joint in JOINTS:
        video = video_angles.get(joint) or ()
        camera = camera_angles.get(joint) or ()
        if not video or not camera:
            continue
        try:
            difference = abs(float(video[-1]) - float(camera[-1]))
        except (TypeError, ValueError):
            continue
        if difference < threshold:
            earned += POINTS_PER_JOINT
    return earned


class MoveScore:
    """Løbende point, det stigende tal og highscore-skærmen for én dans."""

    def __init__(self, scores: HighScoreList) -> None:
        self.match = RoundScore(scores)
        self.score = 0
        self.saved = False
        self._points = None
        self._gain_clock: Optional[float] = None

    def award(
        self,
        video_angles: Mapping[str, Sequence[float]],
        camera_angles: Mapping[str, Sequence[float]],
    ) -> int:
        gained = points_for_latest(video_angles, camera_angles)
        if gained <= 0:
            return 0
        self.score += gained
        from gamConsole.gurli_gris import PointGain

        self._view().gains.append(PointGain(0.0, 0.0, gained))
        game_audio.points(gained)
        return gained

    def place_latest(self, x: float, y: float) -> None:
        gains = self._view().gains
        if not gains:
            return
        gains[-1].x = x
        gains[-1].y = y
        gains[-1].origin_y = y

    def tick(self, now: Optional[float] = None) -> None:
        view = self._view()
        moment = time.monotonic() if now is None else now
        if self._gain_clock is None:
            self._gain_clock = moment
        dt = max(0.0, min(0.1, moment - self._gain_clock))
        self._gain_clock = moment
        alive = []
        for gain in view.gains:
            gain.tick(dt)
            if gain.visible():
                alive.append(gain)
        view.gains = alive

    def draw(self, canvas: np.ndarray) -> None:
        view = self._view()
        view.score = self.score
        for gain in view.gains:
            gain.draw(canvas)
        view.draw_score(canvas)

    def finish(self, scene: np.ndarray, play: Callable[[], None]) -> np.ndarray:
        self.saved = True
        return self.match.finish(scene, self.score, play)

    def result_frame(self) -> Optional[np.ndarray]:
        result = self.match.result
        frozen = self.match._frozen
        if result is None or frozen is None:
            return None
        return result.draw(frozen.copy())

    def reset_attempt(self) -> None:
        self.score = 0
        self.saved = False
        self._gain_clock = None
        if self._points is not None:
            self._points.score = 0
            self._points.gains.clear()
        if self.match.result is not None:
            self.match.clear()

    def _view(self):
        if self._points is None:
            from gamConsole.gurli_gris import BubbleRain

            self._points = BubbleRain()
        self._points.score = self.score
        return self._points
