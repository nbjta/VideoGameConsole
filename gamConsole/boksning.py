#12
"""Boksning i gamConsole.

En music-target med seks neonfelter står rundt om avataren. Det felt der
lyser, skal rammes med handsken mens det er tændt. Avataren er klippet i
led og følger silhuetten. Score og highscore er de samme klasser som Gurli
og Humdingers. Et spil varer 3 minutter.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from gamConsole.boxer_avatar import BoxerAvatar
from gamConsole.boxing_track import BoxingTrack
from gamConsole.flying_users_ar import WebcamARGame
from gamConsole.game_audio import game_audio
from gamConsole.pose_body_overlay import BodyPose, PoseBodyOverlay
from gamConsole.round_score import HighScoreList, RoundScore, RoundScoreGame, highscore_file

#12 Felterne ligger længere ude i slaget, stadig uden for garden.
HIGH_OUT = 0.082
HIGH_UP = 0.146
SIDE_OUT = 0.168
SIDE_DOWN = 0.047
LOW_DOWN = 0.274
PAD_RADIUS = 0.046
GLOVE_PAD = 0.045
LIGHT_SECONDS = 1.4
NEXT_GAP = 0.4
PUNCH_POINTS = 1
PAD_COUNT = 6

#12 BGR. Samme rækkefølge som felterne: oppe venstre, oppe højre, midt højre, nede højre, nede venstre, midt venstre.
NEON_COLORS = (
    (255, 40, 200),
    (255, 230, 40),
    (50, 230, 70),
    (40, 140, 255),
    (255, 90, 40),
    (200, 40, 255),
)

ROOT = Path(__file__).resolve().parents[1]
RING_PATH = ROOT / "assets" / "boksning" / "ring.png"
HOUSING = (28, 28, 32)
PAD_OFF = (58, 58, 64)


class RingFrame:
    """Stolper, reb og lærred i pixel, så bokseren bliver inden for ringen."""

    def __init__(self, width: int, height: int) -> None:
        self.left = int(round(width * 0.10))
        self.right = int(round(width * 0.90))
        self.rope_top = int(round(height * 0.32))
        self.ropes = (
            int(round(height * 0.34)),
            int(round(height * 0.46)),
            int(round(height * 0.58)),
        )
        self.floor_top = int(round(height * 0.62))
        self.floor_bottom = int(round(height * 0.92))


class RingBackdrop:
    """Ringbilledet fylder vinduet. Kameraet tegnes ikke."""

    def __init__(self, image_path: Path = RING_PATH) -> None:
        loaded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if loaded is None:
            raise FileNotFoundError(f"Boksering-baggrund mangler: {image_path}")
        self._image = loaded
        self._cache: dict = {}

    def render(self, width: int, height: int) -> np.ndarray:
        #12 Cover, samme måde som Gurli: hele vinduet er ringen, uden sorte kanter.
        key = (width, height)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()
        src_h, src_w = self._image.shape[:2]
        scale = max(width / src_w, height / src_h)
        resized = cv2.resize(
            self._image,
            (max(width, int(round(src_w * scale))), max(height, int(round(src_h * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
        y0 = max(0, (resized.shape[0] - height) // 2)
        x0 = max(0, (resized.shape[1] - width) // 2)
        crop = resized[y0 : y0 + height, x0 : x0 + width]
        if crop.shape[0] != height or crop.shape[1] != width:
            crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
        self._cache[key] = crop
        return crop.copy()


class Boxer:
    """Avataren i ringen. Den står stille, og leddene følger silhuetten."""

    def __init__(self, avatar: Optional[BoxerAvatar] = None) -> None:
        self.avatar = avatar if avatar is not None else BoxerAvatar()

    def box(self, pose: Optional[BodyPose], width: int, height: int) -> Tuple[int, int, int, int]:
        #12 Bokseren bliver stående. Hovedvippen flytter ikke figuren.
        del pose
        #12 Hovedet starter under scoren. Knæet når bunden, og underbenene skæres af.
        margin = max(4, int(round(height * 0.10)))
        full = max(1, self.avatar.bottom - self.avatar.top)
        knee = max(1, int(round(self.avatar.knee_y - self.avatar.top)))
        visible = max(80, height - margin)
        body_h = max(80, int(round(visible * full / knee)))
        body_w = max(40, int(body_h * 0.28))
        y1 = margin
        x1 = width // 2 - body_w // 2
        return x1, y1, x1 + body_w, y1 + body_h

    def gloves(self, pose: Optional[BodyPose], width: int, height: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self.avatar.glove_tips(pose, self.box(pose, width, height))

    def draw(self, canvas: np.ndarray, pose: Optional[BodyPose]) -> None:
        height, width = canvas.shape[:2]
        self.avatar.draw(canvas, pose, self.box(pose, width, height))


class TargetPad:
    """Et af de seks felter. Farven er neon, og kun det tændte felt tæller."""

    def __init__(self, index: int, x: float, y: float, radius: int, color: Tuple[int, int, int]) -> None:
        self.index = index
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color


class MusicTarget:
    """Seks felter rundt om bokseren. Et felt lyser ad gangen."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.score = 0
        self.pads: List[TargetPad] = []
        self.hub = (0.0, 0.0)
        self.gains: List[object] = []
        self._rng = rng if rng is not None else np.random.default_rng()
        self._lit: Optional[int] = None
        self._lit_at = 0.0
        self._last = -1
        self._next: Optional[float] = None
        self._last_now: Optional[float] = None
        self._key: Optional[Tuple[int, int]] = None
        self._score_view = None

    @property
    def lit(self) -> Optional[int]:
        return self._lit

    def reset(self) -> None:
        self.score = 0
        self.gains.clear()
        self._lit = None
        self._lit_at = 0.0
        self._last = -1
        self._next = None
        self._last_now = None
        if self._score_view is not None:
            self._score_view.score = 0

    def place(self, boxer: Boxer, width: int, height: int) -> None:
        key = (width, height)
        if self._key == key and len(self.pads) == PAD_COUNT:
            return
        self.pads, self.hub = _pads_around(boxer, width, height)
        self._key = key

    def light(self, index: int, now: float) -> None:
        self._lit = index
        self._lit_at = now
        self._last = index

    def tick(
        self,
        boxer: Boxer,
        pose: Optional[BodyPose],
        width: int,
        height: int,
        now: float,
        *,
        auto: bool = True,
    ) -> None:
        self.place(boxer, width, height)
        if self._last_now is None:
            self._last_now = now
        dt = max(0.0, min(0.1, now - self._last_now))
        self._last_now = now
        if self._lit is not None:
            pad = self.pads[self._lit]
            alive = now - self._lit_at <= LIGHT_SECONDS
            gloves = boxer.gloves(pose, width, height)
            pad_reach = _glove_pad(boxer, pose, width, height)
            if alive and _touches(gloves, pad_reach, pad):
                self.score += PUNCH_POINTS
                self.gains.append(_point_gain(pad.x, pad.y, PUNCH_POINTS))
                game_audio.points(PUNCH_POINTS)
                self._lit = None
                self._next = now + NEXT_GAP
            elif not alive:
                self._lit = None
                self._next = now + NEXT_GAP
        if auto and self._lit is None and (self._next is None or now >= self._next):
            self.light(self._pick(), now)
        still = []
        for gain in self.gains:
            gain.tick(dt)  # type: ignore[attr-defined]
            if gain.visible():  # type: ignore[attr-defined]
                still.append(gain)
        self.gains = still

    def draw_back(self, canvas: np.ndarray, now: float) -> None:
        if len(self.pads) != PAD_COUNT:
            return
        hub = (int(round(self.hub[0])), int(round(self.hub[1])))
        outer = 0.0
        for pad in self.pads:
            outer = max(outer, math.hypot(pad.x - self.hub[0], pad.y - self.hub[1]))
        radius = int(round(outer + self.pads[0].radius + 10))
        cv2.circle(canvas, hub, radius, HOUSING, -1, cv2.LINE_AA)
        cv2.circle(canvas, hub, radius, (90, 90, 98), 4, cv2.LINE_AA)
        for pad in self.pads:
            lit = pad.index == self._lit
            color = pad.color if lit else _dim(pad.color, 0.35)
            thick = 10 if lit else 4
            cv2.line(canvas, hub, (int(round(pad.x)), int(round(pad.y))), color, thick, cv2.LINE_AA)
        cv2.circle(canvas, hub, max(8, self.pads[0].radius // 2), (16, 16, 18), -1, cv2.LINE_AA)

    def draw_pads(self, canvas: np.ndarray, now: float) -> None:
        for pad in self.pads:
            _draw_pad(canvas, pad, pad.index == self._lit, now, self._lit_at)
        for gain in self.gains:
            gain.draw(canvas)  # type: ignore[attr-defined]

    def draw_score(self, canvas: np.ndarray) -> None:
        #12 Samme score-tal som Gurli Gris. Boblerne derfra bliver ikke tegnet.
        view = self._score_bar()
        view.score = self.score
        view.draw_score(canvas)

    def _pick(self) -> int:
        choices = [index for index in range(PAD_COUNT) if index != self._last]
        return int(self._rng.choice(choices))

    def _score_bar(self):
        if self._score_view is None:
            from gamConsole.gurli_gris import BubbleRain

            self._score_view = BubbleRain()
        return self._score_view


class BoxingPlayer:
    """Ring, bokser og music-target. Kameraet bruges kun til silhuetten."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self._pose = PoseBodyOverlay()
        self.boxer = Boxer()
        self.target = MusicTarget(rng)
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
        self.target.place(self.boxer, width, height)
        self.target.tick(self.boxer, pose, width, height, now)
        self.target.draw_back(scene, now)
        self.boxer.draw(scene, pose)
        self.target.draw_pads(scene, now)
        #12 Punktsilhuetten er debug ovenpå, som i Gurli og Humdingers.
        self._pose.draw(scene, pose)
        self.target.draw_score(scene)
        return scene


class BoxingGame(RoundScoreGame, WebcamARGame):
    """Bokseren står i music-targeten. Kameraet bruges kun til silhuetten."""

    window_name = "gamConsole: Boksning"

    def __init__(self, camera: int, scores: Optional[HighScoreList] = None, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(camera)
        self.backdrop = RingBackdrop()
        self.player = BoxingPlayer(rng)
        #20 Action Driving Rock kører kun, mens runden spilles.
        self.track = BoxingTrack()
        #12 Samme 3 minutter og highscore som Gurli, i boksningens egen fil.
        board = scores if scores is not None else HighScoreList(highscore_file("boksning"))
        self.match = RoundScore(board)

    def close(self) -> None:
        #20 Menuen og lukket vindue stopper sangen.
        self.track.stop()
        self.player.close()

    def play_again(self) -> None:
        #20 Spil igen starter filen forfra på næste billede.
        self.track.stop()
        self.player.target.reset()
        self.match.clear()

    def on_frame(self, frame: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        #12 Spejlvendt webcam, samme retning som Gurli og Humdingers.
        camera = cv2.flip(frame, 1)
        moment = time.monotonic() if now is None else now
        shown = self.match.begin(
            moment,
            self.player.target.score,
            self.play_again,
            lambda: self.backdrop.render(camera.shape[1], camera.shape[0]),
        )
        if shown is not None:
            #20 Resultatskærmen er stille. Næste runde starter sangen forfra.
            self.track.stop()
            return shown
        self.track.ensure_playing()
        self.player.refresh_pose(camera)
        height, width = camera.shape[:2]
        scene = self.player.composite(camera, self.backdrop.render(width, height), moment)
        return self.match.stamp(scene, moment)


def _pads_around(boxer: Boxer, width: int, height: int) -> Tuple[List[TargetPad], Tuple[float, float]]:
    #12 Felterne måles fra garden, så de bliver stående når armen allerede er løftet.
    box = boxer.box(None, width, height)
    left, right = boxer.avatar.glove_tips(None, box)
    span = float(box[3] - box[1])
    radius = max(12, int(round(height * PAD_RADIUS)))
    spots = (
        (left[0] - span * HIGH_OUT, left[1] - span * HIGH_UP),
        (right[0] + span * HIGH_OUT, right[1] - span * HIGH_UP),
        (right[0] + span * SIDE_OUT, right[1] + span * SIDE_DOWN),
        (right[0], right[1] + span * LOW_DOWN),
        (left[0], left[1] + span * LOW_DOWN),
        (left[0] - span * SIDE_OUT, left[1] + span * SIDE_DOWN),
    )
    pads = [TargetPad(index, spot[0], spot[1], radius, NEON_COLORS[index]) for index, spot in enumerate(spots)]
    hub = (
        (left[0] + right[0]) / 2.0,
        (left[1] + right[1]) / 2.0 + span * 0.06,
    )
    return pads, hub


def _glove_pad(boxer: Boxer, pose: Optional[BodyPose], width: int, height: int) -> float:
    box = boxer.box(pose, width, height)
    return max(8.0, (box[3] - box[1]) * GLOVE_PAD)


def _touches(gloves: Sequence[Tuple[float, float]], pad: float, target: TargetPad) -> bool:
    limit = target.radius + pad
    return any(math.hypot(glove[0] - target.x, glove[1] - target.y) <= limit for glove in gloves)


def _dim(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(int(channel * factor) for channel in color)  # type: ignore[return-value]


def _draw_pad(canvas: np.ndarray, pad: TargetPad, lit: bool, now: float, lit_at: float) -> None:
    center = (int(round(pad.x)), int(round(pad.y)))
    radius = pad.radius
    if lit:
        pulse = 0.5 + 0.5 * math.sin((now - lit_at) * 9.0)
        glow = int(round(radius * (1.45 + 0.2 * pulse)))
        cv2.circle(canvas, center, glow, _dim(pad.color, 0.45), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, radius, pad.color, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, radius, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, center, max(3, radius // 3), (255, 255, 255), -1, cv2.LINE_AA)
        return
    cv2.circle(canvas, center, radius, PAD_OFF, -1, cv2.LINE_AA)
    cv2.circle(canvas, center, radius, _dim(pad.color, 0.85), 3, cv2.LINE_AA)
    cv2.circle(canvas, center, max(3, radius // 4), (110, 110, 118), -1, cv2.LINE_AA)


def _point_gain(x: float, y: float, value: int):
    from gamConsole.gurli_gris import PointGain

    return PointGain(x, y, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="gamConsole: Boksning")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    game = BoxingGame(args.camera)
    try:
        game.run()
    finally:
        game.close()


if __name__ == "__main__":
    main()
