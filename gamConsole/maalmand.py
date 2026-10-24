#11
"""Målmand i gamConsole.

Baggrunden er et fodboldmål. Avataren er klippet i led og følger silhuetten,
samme vinkler som Gurli og Humdingers. En bold i kroppen giver 1 point.
De 3 minutter og highscoren er de samme klasser som de to spil.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gamConsole.flying_users_ar import WebcamARGame
from gamConsole.game_audio import game_audio
from gamConsole.head_tilt import HeadTilt
from gamConsole.keeper_avatar import KeeperAvatar
from gamConsole.pose_body_overlay import BodyPose, PoseBodyOverlay
from gamConsole.round_score import HighScoreList, RoundScore, RoundScoreGame, highscore_file

#11 Målets plads i billedet. Overliggeren sidder højt, mållinjen nede på græsset.
LEFT_POST = 0.14
RIGHT_POST = 0.86
CROSSBAR_Y = 0.18
GOAL_LINE_Y = 0.82
FLIGHT_SECONDS = 1.8
SPAWN_GAP = 1.4
MAX_BALLS = 3
CATCH_POINTS = 1
_SUNK_SECONDS = 0.35

SKY = (250, 190, 140)
GRASS = (45, 155, 60)
POST = (245, 245, 245)
NET = (225, 225, 230)


class GoalFrame:
    """Stolper, overligger og mållinje i pixel, så bold og målmand rammer det samme mål."""

    def __init__(self, width: int, height: int) -> None:
        thick = max(8, width // 36)
        if thick % 2:
            thick += 1
        self.thick = thick
        half = thick // 2
        self.left_c = int(round(width * LEFT_POST))
        self.right_c = int(round(width * RIGHT_POST))
        self.top = int(round(height * CROSSBAR_Y))
        self.line = int(round(height * GOAL_LINE_Y))
        self.inner_left = self.left_c + half
        self.inner_right = self.right_c - half
        self.outer_left = self.left_c - half
        self.outer_right = self.right_c + half


class GoalBackdrop:
    """Himmel, græs og et hvidt mål med net. Kameraet tegnes ikke."""

    def __init__(self) -> None:
        self._cache: dict = {}

    def render(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()
        frame = _paint_goal(width, height)
        self._cache[key] = frame
        return frame.copy()


class Goalkeeper:
    """Avataren i målet. Vip flytter den, og leddene følger silhuetten."""

    def __init__(self, avatar: Optional[KeeperAvatar] = None) -> None:
        self.avatar = avatar if avatar is not None else KeeperAvatar()

    def size(self, width: int, height: int) -> Tuple[int, int]:
        goal = GoalFrame(width, height)
        mouth_h = max(40, goal.line - goal.top)
        body_h = min(max(70, int(height * 0.50)), max(40, mouth_h - 4))
        #11 Bredden dækker kroppen med armene nede, målt på avataren.
        body_w = max(48, int(body_h * 0.34))
        return body_w, body_h

    def origin_x(self, pose: Optional[BodyPose], width: int, height: int) -> int:
        body_w, _body_h = self.size(width, height)
        goal = GoalFrame(width, height)
        #11 Samme hovedvip som Gurli. Stolperne holder målmanden inde i målet.
        origin_x = width // 2 - body_w // 2 + HeadTilt().offset_x(pose, width)
        low = goal.inner_left
        high = max(low, goal.inner_right - body_w)
        return int(np.clip(origin_x, low, high))

    def box(self, pose: Optional[BodyPose], width: int, height: int) -> Tuple[int, int, int, int]:
        """Kroppen med udstrakte arme. En bold her er grebet."""
        body_w, body_h = self.size(width, height)
        goal = GoalFrame(width, height)
        x = self.origin_x(pose, width, height)
        y = goal.line - body_h
        return x, y, x + body_w, y + body_h

    def draw(self, canvas: np.ndarray, pose: Optional[BodyPose]) -> None:
        height, width = canvas.shape[:2]
        self.avatar.draw(canvas, pose, self.box(pose, width, height))


class Football:
    """En bold på vej fra forgrunden ind i målet."""

    def __init__(self, start: Tuple[float, float], target: Tuple[float, float], radius: int, born: float) -> None:
        self.start = start
        self.target = target
        self.radius = radius
        self.born = born
        self.sunk_at: Optional[float] = None

    def progress(self, now: float) -> float:
        return max(0.0, min(1.0, (now - self.born) / FLIGHT_SECONDS))

    def arrived(self, now: float) -> bool:
        return now - self.born >= FLIGHT_SECONDS

    def position(self, now: float) -> Tuple[float, float]:
        t = self.progress(now)
        x = self.start[0] + (self.target[0] - self.start[0]) * t
        y = self.start[1] + (self.target[1] - self.start[1]) * t
        return x, y

    def radius_now(self, now: float) -> float:
        #11 Bolden er størst tæt på og mindre, når den når målmanden.
        return self.radius * (1.2 - 0.35 * self.progress(now))

    def draw(self, canvas: np.ndarray, now: float) -> None:
        x, y = self.position(now)
        radius = max(4, int(round(self.radius_now(now))))
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(canvas, (cx, cy), radius, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), radius, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), max(2, radius // 3), (30, 30, 30), -1, cv2.LINE_AA)
        for turn in range(5):
            angle = math.radians(turn * 72 - 90)
            px = int(round(cx + math.cos(angle) * radius * 0.62))
            py = int(round(cy + math.sin(angle) * radius * 0.62))
            cv2.circle(canvas, (px, py), max(2, radius // 5), (30, 30, 30), -1, cv2.LINE_AA)


class ShotClock:
    """Boldene. De gribes, når de rammer målmandens kasse. Ved siden af giver de 0."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.score = 0
        self.balls: List[Football] = []
        self.gains: List[object] = []
        self._rng = rng if rng is not None else np.random.default_rng()
        self._next: Optional[float] = None
        self._last_now: Optional[float] = None
        self._score_view = None

    def reset(self) -> None:
        self.score = 0
        self.balls.clear()
        self.gains.clear()
        self._next = None
        self._last_now = None
        if self._score_view is not None:
            self._score_view.score = 0

    def tick(
        self,
        width: int,
        height: int,
        box: Tuple[int, int, int, int],
        now: float,
        *,
        spawn: bool = True,
    ) -> None:
        if self._last_now is None:
            self._last_now = now
        dt = max(0.0, min(0.1, now - self._last_now))
        self._last_now = now
        if spawn and len(self.balls) < MAX_BALLS and (self._next is None or now >= self._next):
            chest_y = (box[1] + box[3]) / 2.0
            self.balls.append(self._make(width, height, now, chest_y))
            self._next = now + SPAWN_GAP
        kept: List[Football] = []
        for ball in self.balls:
            if ball.sunk_at is None and ball.arrived(now):
                x, y = ball.position(now)
                if _overlaps(box, x, y, ball.radius_now(now)):
                    self.score += CATCH_POINTS
                    self.gains.append(_point_gain(x, y, CATCH_POINTS))
                    game_audio.points(CATCH_POINTS)
                    continue
                ball.sunk_at = now
            if ball.sunk_at is not None and now - ball.sunk_at > _SUNK_SECONDS:
                continue
            kept.append(ball)
        self.balls = kept
        still = []
        for gain in self.gains:
            gain.tick(dt)  # type: ignore[attr-defined]
            if gain.visible():  # type: ignore[attr-defined]
                still.append(gain)
        self.gains = still

    def draw(self, canvas: np.ndarray, now: float) -> None:
        for ball in self.balls:
            ball.draw(canvas, now)
        for gain in self.gains:
            gain.draw(canvas)  # type: ignore[attr-defined]

    def draw_score(self, canvas: np.ndarray) -> None:
        #11 Samme score-tal som Gurli Gris. Boblerne derfra bliver ikke tegnet.
        view = self._score_bar()
        view.score = self.score
        view.draw_score(canvas)

    def _score_bar(self):
        if self._score_view is None:
            from gamConsole.gurli_gris import BubbleRain

            self._score_view = BubbleRain()
        return self._score_view

    def _make(self, width: int, height: int, now: float, chest_y: float) -> Football:
        goal = GoalFrame(width, height)
        radius = max(10, int(height * 0.045))
        margin = radius + 2
        low = goal.inner_left + margin
        high = goal.inner_right - margin
        if high <= low:
            target_x = (goal.inner_left + goal.inner_right) / 2.0
        else:
            target_x = float(self._rng.integers(low, high))
        start = (width / 2.0, float(height - radius - 2))
        return Football(start, (target_x, chest_y), radius, now)


class GoalkeeperPlayer:
    """Mål, målmand og punktsilhuet. Kameraet bruges kun til hovedvippen."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self._pose = PoseBodyOverlay()
        self.keeper = Goalkeeper()
        self.shots = ShotClock(rng)
        self._last_pose: Optional[BodyPose] = None

    def close(self) -> None:
        self._pose.close()

    def refresh_pose(self, webcam: np.ndarray) -> None:
        self._last_pose = self._pose.snapshot(webcam)

    def composite(self, webcam: np.ndarray, background: np.ndarray, now: float) -> np.ndarray:
        if background.shape[:2] != webcam.shape[:2]:
            background = cv2.resize(background, (webcam.shape[1], webcam.shape[0]))
        pose = self._last_pose
        scene = background.copy()
        height, width = scene.shape[:2]
        box = self.keeper.box(pose, width, height)
        self.shots.tick(width, height, box, now)
        self.keeper.draw(scene, pose)
        self.shots.draw(scene, now)
        #11 Punktsilhuetten er debug ovenpå, som i Gurli og Humdingers.
        self._pose.draw(scene, pose)
        self.shots.draw_score(scene)
        return scene


class GoalkeeperGame(RoundScoreGame, WebcamARGame):
    """Målmanden står i målet. Kameraet bruges kun til silhuetten."""

    window_name = "gamConsole: Målmand"

    def __init__(self, camera: int, scores: Optional[HighScoreList] = None, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(camera)
        self.backdrop = GoalBackdrop()
        self.player = GoalkeeperPlayer(rng)
        #11 Samme 3 minutter og highscore som Gurli, i målets egen fil.
        board = scores if scores is not None else HighScoreList(highscore_file("maalmand"))
        self.match = RoundScore(board)

    def close(self) -> None:
        self.player.close()

    def play_again(self) -> None:
        self.player.shots.reset()
        self.match.clear()

    def on_frame(self, frame: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        #11 Spejlvendt webcam, samme retning som Gurli og Humdingers.
        camera = cv2.flip(frame, 1)
        moment = time.monotonic() if now is None else now
        shown = self.match.begin(
            moment,
            self.player.shots.score,
            self.play_again,
            lambda: self.backdrop.render(camera.shape[1], camera.shape[0]),
        )
        if shown is not None:
            return shown
        self.player.refresh_pose(camera)
        height, width = camera.shape[:2]
        scene = self.player.composite(camera, self.backdrop.render(width, height), moment)
        return self.match.stamp(scene, moment)


def _paint_goal(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), np.uint8)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    frame[:, :, 0] = np.clip(SKY[0] - 30 * ys, 0, 255)
    frame[:, :, 1] = np.clip(SKY[1] - 20 * ys, 0, 255)
    frame[:, :, 2] = np.clip(SKY[2] - 10 * ys, 0, 255)
    goal = GoalFrame(width, height)
    frame[goal.line :] = GRASS
    gap = max(12, width // 24)
    x = goal.inner_left
    while x < goal.inner_right:
        cv2.line(frame, (x, goal.top), (x, goal.line), NET, 1, cv2.LINE_AA)
        x += gap
    y = goal.top
    while y < goal.line:
        cv2.line(frame, (goal.inner_left, y), (goal.inner_right, y), NET, 1, cv2.LINE_AA)
        y += gap
    line_t = max(2, height // 140)
    cv2.line(frame, (0, goal.line), (width, goal.line), POST, line_t, cv2.LINE_AA)
    half = goal.thick // 2
    cv2.rectangle(
        frame,
        (goal.outer_left, goal.top - half),
        (goal.outer_right, goal.top + half),
        POST,
        -1,
    )
    for center in (goal.left_c, goal.right_c):
        cv2.rectangle(
            frame,
            (center - half, goal.top - half),
            (center + half, goal.line + half),
            POST,
            -1,
        )
    return frame


def _overlaps(box: Tuple[int, int, int, int], x: float, y: float, radius: float) -> bool:
    x1, y1, x2, y2 = box
    nearest_x = min(max(x, x1), x2)
    nearest_y = min(max(y, y1), y2)
    return math.hypot(x - nearest_x, y - nearest_y) <= radius


def _point_gain(x: float, y: float, value: int):
    from gamConsole.gurli_gris import PointGain

    return PointGain(x, y, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="gamConsole: Målmand")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    game = GoalkeeperGame(args.camera)
    try:
        game.run()
    finally:
        game.close()


if __name__ == "__main__":
    main()
