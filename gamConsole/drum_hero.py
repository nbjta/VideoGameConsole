#13
"""Drum Hero i gamConsole.

Øverst er hero-banen: noderne kommer ned mod en streg. I bunden står
trommesættet, og armene slår på trommen mens noden er i træffet.
Hvile-stikket ligger over trommerne. Avataren er klippet i led og følger
silhuetten. Score og highscore er de samme klasser som Gurli og Humdingers.
Et spil varer 3 minutter.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from gamConsole.drum_hero_setup import MEDIUM, Difficulty, DrumSetup, DrumSong
from gamConsole.drum_track import DrumTrack
from gamConsole.drum_voices import DrumVoices
from gamConsole.jointed_figure import _shave_edge
from gamConsole.leader_back_view import ChaseInCar, _cut_white_background
from gamConsole.drummer_avatar import DrummerAvatar
from gamConsole.flying_users_ar import WebcamARGame
from gamConsole.pose_body_overlay import BodyPose, PoseBodyOverlay
from gamConsole.round_score import HighScoreList, RoundScore, RoundScoreGame, highscore_file

#28 Målt på overkroppens kasse i 640×360. Hvile rammer ikke.
# Slag til siden rammer bækkenet. Slag ned rammer bundtrommen i underkanten.
HIHAT_X = -0.1665
HIHAT_Y = 0.3081
SNARE_X = 0.0761
SNARE_Y = 0.4152

HIHAT, SNARE, TOM, CRASH = 0, 1, 2, 3
DRUM_COUNT = 4
#13 Venstre stik: hi-hat og lilletromme. Højre stik: tam og crash.
LANE_HAND = (0, 0, 1, 1)
#16 Mellem er det tempo, banen havde, før sværhedsgraden kom til.
BEAT = MEDIUM.beat
LEAD = 1.6
HIT_EARLY = MEDIUM.hit_early
HIT_LATE = MEDIUM.hit_late
NOTE_POINTS = 1
NOTE_TOP = 0.08
#14 Banen slutter her. Trommesættet og armene ligger under stregen.
HIGHWAY_END = 0.46
LANE_FAR = 0.34
DRUM_RADIUS = 0.046
#26 Nederste trommer er større end bækkenerne.
LOW_DRUM_RADIUS = 0.106
#28 Rækkevidden er en del af vinduet. Slag ned ender under kanten og skal stadig ramme.
STICK_REACH = 0.10
#32 Lidt fladere end billedets egen højde.
LOW_DRUM_FLAT = 0.85
#32 Mindre end 1.35. Ringen ligger tæt på billedet.
LOW_DRUM_LIGHT = 1.12
#32 Additivt neon. 1.0 ville være den gamle dækkende cirkel.
NEON_STRENGTH = 0.28
NEON_HOT = 0.48
LOW_DRUM_PATH = Path(__file__).resolve().parents[1] / "assets" / "drum_hero" / "low_drum.png"
#33 Samme bredde som den gamle ellipse. Højden følger billedet.
CYMBAL_WIDTH = 2.8
CYMBAL_PATH = Path(__file__).resolve().parents[1] / "assets" / "drum_hero" / "cymbal.png"
PATTERN = (SNARE, TOM, HIHAT, CRASH)

#13 BGR. Samme rækkefølge som banerne.
DRUM_COLORS = (
    (0, 220, 255),
    (36, 36, 230),
    (40, 200, 48),
    (235, 130, 28),
)
DRUM_KINDS = ("cymbal", "drum", "drum", "cymbal")


class StageFrame:
    """Kanten og gulvet, så trommeslageren bliver på scenen."""

    def __init__(self, width: int, height: int) -> None:
        self.left = int(round(width * 0.04))
        self.right = int(round(width * 0.96))
        #14 Gulvet er trommesættets område, under hero-banen.
        self.floor_top = int(round(height * HIGHWAY_END))
        self.floor_bottom = int(round(height * 0.985))


class StageBackdrop:
    """Scenen fylder vinduet. Kameraet tegnes ikke."""

    def __init__(self) -> None:
        self._cache: dict = {}

    def render(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()
        frame = _paint_stage(width, height)
        self._cache[key] = frame
        return frame.copy()


class Drummer:
    """Avataren på scenen. Den står stille, og leddene følger silhuetten."""

    def __init__(self, avatar: Optional[DrummerAvatar] = None) -> None:
        self.avatar = avatar if avatar is not None else DrummerAvatar()

    def box(self, pose: Optional[BodyPose], width: int, height: int) -> Tuple[int, int, int, int]:
        stage = StageFrame(width, height)
        avatar = self.avatar
        #28 Overkroppen fylder fra hero-banen til kanten. Benene ligger under billedet.
        span_y = avatar.waist - avatar.top
        span_x = avatar.right - avatar.left
        scale = min(
            (height - stage.floor_top) / span_y,
            (stage.right - stage.left) / span_x,
        )
        y2 = height + (avatar.bottom - avatar.waist) * scale
        y1 = y2 - (avatar.bottom - avatar.top) * scale
        body_w = span_x * scale
        low = stage.left
        high = max(low, stage.right - body_w)
        x1 = float(np.clip(width / 2 - body_w / 2, low, high))
        #13 Trommeslageren bliver stående. Hovedvippen flytter ikke figuren.
        del pose
        return int(round(x1)), int(round(y1)), int(round(x1 + body_w)), int(round(y2))

    def sticks(self, pose: Optional[BodyPose], width: int, height: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self.avatar.stick_tips(pose, self.box(pose, width, height))

    def draw(self, canvas: np.ndarray, pose: Optional[BodyPose]) -> None:
        height, width = canvas.shape[:2]
        self.avatar.draw(canvas, pose, self.box(pose, width, height))


class DrumPad:
    """En tromme eller et bækken. Kun det rigtige stik i træffet tæller."""

    def __init__(self, lane: int, x: float, y: float, radius: int, color: Tuple[int, int, int], kind: str) -> None:
        self.lane = lane
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.kind = kind


class DrumNote:
    """En node i en bane. hit_at er det tidspunkt, den skal rammes."""

    def __init__(self, lane: int, hit_at: float) -> None:
        self.lane = lane
        self.hit_at = hit_at


class DrumChart:
    """Det faste mønster. Noderne fødes LEAD sekunder før træffet."""

    def __init__(self, difficulty: Optional[Difficulty] = None) -> None:
        #16 Uden valg bruges mellem, som er det gamle mønster.
        self.difficulty = MEDIUM if difficulty is None else difficulty
        self._origin: Optional[float] = None
        self._spawned = 0

    def reset(self) -> None:
        self._origin = None
        self._spawned = 0

    def _spawn(self, index: int) -> Tuple[float, Tuple[int, ...]]:
        return index * self.difficulty.beat, self.difficulty.lanes(index)

    def take(self, now: float) -> List[DrumNote]:
        if self._origin is None:
            self._origin = now
        born: List[DrumNote] = []
        _, late = self.difficulty.windows()
        while self._spawned < 20000:
            offset, lanes = self._spawn(self._spawned)
            hit_at = self._origin + LEAD + offset
            if hit_at - LEAD > now:
                break
            if hit_at + late >= now:
                for lane in lanes:
                    born.append(DrumNote(lane, hit_at))
            self._spawned += 1
        return born


class HitChart(DrumChart):
    """Slagene fra én sang, i sekunder, på den tromme afkodningen gættede."""

    def __init__(self, difficulty: Difficulty, song: DrumSong) -> None:
        super().__init__(difficulty)
        #24 Noderne gentages, når filen loops, med samme sekundtal som lyden.
        self._duration = song.seconds
        self._hits = difficulty.select(song.hits, song.seconds)
        if not self._hits:
            raise ValueError("sangen har ingen slag at spille")

    def _spawn(self, index: int) -> Tuple[float, Tuple[int, ...]]:
        cycle, slot = divmod(index, len(self._hits))
        hit = self._hits[slot]
        return cycle * self._duration + hit.seconds, (hit.lane,)


class DrumKit:
    """Fire trommer og noderne der falder ned på dem."""

    def __init__(self, difficulty: Optional[Difficulty] = None) -> None:
        self.score = 0
        self.drums: List[DrumPad] = []
        self.notes: List[DrumNote] = []
        self.gains: List[object] = []
        self.difficulty = MEDIUM if difficulty is None else difficulty
        self.chart = DrumChart(self.difficulty)
        #26 Hvert træf spiller trommens egen lyd.
        self.voices = DrumVoices()
        self._width = 1
        self._height = 1
        self._last_now: Optional[float] = None
        self._score_view = None

    def use(self, difficulty: Difficulty, song: Optional[DrumSong] = None) -> None:
        #16 Nyt valg tømmer noderne, så den gamle bane ikke bliver hængende.
        self.difficulty = difficulty
        #24 Hver sang bruger sin egen liste. Uden sang bruges det faste mønster.
        if song is None:
            self.chart = DrumChart(difficulty)
        else:
            self.chart = HitChart(difficulty, song)
        self.reset()

    def reset(self) -> None:
        self.score = 0
        self.notes.clear()
        self.gains.clear()
        self.chart.reset()
        self._last_now = None
        if self._score_view is not None:
            self._score_view.score = 0

    def add_note(self, lane: int, hit_at: float) -> None:
        self.notes.append(DrumNote(lane, hit_at))

    def place(self, drummer: Drummer, pose: Optional[BodyPose], width: int, height: int) -> None:
        #13 Trommerne følger kassen, så et vip flytter sættet sammen med stikkerne.
        box = drummer.box(pose, width, height)
        self.drums = _drums_around(drummer, box, height)
        self._width = width
        self._height = height

    def tick(
        self,
        drummer: Drummer,
        pose: Optional[BodyPose],
        width: int,
        height: int,
        now: float,
        *,
        auto: bool = True,
    ) -> None:
        self.place(drummer, pose, width, height)
        if self._last_now is None:
            self._last_now = now
        dt = max(0.0, min(0.1, now - self._last_now))
        self._last_now = now
        if auto:
            self.notes.extend(self.chart.take(now))
        sticks = drummer.avatar.stick_tips(pose, drummer.box(pose, width, height))
        pad = _stick_reach(drummer, pose, width, height)
        still: List[DrumNote] = []
        for note in self.notes:
            if now > note.hit_at + self._limits()[1]:
                continue
            drum = self.drums[note.lane]
            if self._waiting(note, now) and _reaches(sticks[LANE_HAND[note.lane]], drum, pad):
                self.score += NOTE_POINTS
                self.gains.append(_point_gain(drum.x, drum.y, NOTE_POINTS))
                #26 Ikke bib. Lyden følger den tromme, stikket ramte.
                self.voices.hit(note.lane)
                continue
            still.append(note)
        self.notes = still
        alive = []
        for gain in self.gains:
            gain.tick(dt)  # type: ignore[attr-defined]
            if gain.visible():  # type: ignore[attr-defined]
                alive.append(gain)
        self.gains = alive

    def note_xy(self, note: DrumNote, now: float) -> Tuple[float, float]:
        #14 Noden kører på hero-banen og lander på stregen, ikke på trommen.
        drum = self.drums[note.lane]
        travel = _travel(note, now)
        top = self._height * NOTE_TOP
        hit_y = self._height * HIGHWAY_END
        return _lane_x(drum.x, travel, self._width), top + (hit_y - top) * travel

    def paint(self, canvas: np.ndarray, drummer: Drummer, pose: Optional[BodyPose], now: float, hud: bool = True) -> None:
        #14 Banen tegnes øverst. Armene tegnes over sættet, trommerne forrest.
        self._draw_highway(canvas, now)
        _draw_kit_hardware(canvas, self.drums)
        drummer.draw(canvas, pose)
        self._draw_drums(canvas, now)
        for gain in self.gains:
            gain.draw(canvas)  # type: ignore[attr-defined]
        if hud:
            self.draw_score(canvas)

    def draw_score(self, canvas: np.ndarray) -> None:
        view = self._score_bar()
        view.score = self.score
        view.draw_score(canvas)

    def _draw_highway(self, canvas: np.ndarray, now: float) -> None:
        if len(self.drums) != DRUM_COUNT:
            return
        height, width = canvas.shape[:2]
        top = height * NOTE_TOP
        hit_y = height * HIGHWAY_END
        #27 Banens bredde følger den gamle nodestørrelse, ikke de store trommer.
        mark = _highway_radius(height)
        pad = mark * 2.1
        near_l = min(drum.x for drum in self.drums) - pad
        near_r = max(drum.x for drum in self.drums) + pad
        track = np.array(
            [
                [int(round(_lane_x(near_l, 0.0, width))), int(round(top))],
                [int(round(_lane_x(near_r, 0.0, width))), int(round(top))],
                [int(round(near_r)), int(round(hit_y))],
                [int(round(near_l)), int(round(hit_y))],
            ],
            np.int32,
        )
        cv2.rectangle(canvas, (0, 0), (width, int(round(hit_y)) + 1), (12, 10, 16), -1)
        cv2.fillConvexPoly(canvas, track, (32, 28, 38), cv2.LINE_AA)
        cv2.polylines(canvas, [track], True, (90, 86, 104), 2, cv2.LINE_AA)
        for step in (0.33, 0.66):
            y = int(round(top + (hit_y - top) * step))
            cv2.line(
                canvas,
                (int(round(_lane_x(near_l, step, width))), y),
                (int(round(_lane_x(near_r, step, width))), y),
                (52, 48, 62),
                1,
                cv2.LINE_AA,
            )
        hot = {note.lane for note in self.notes if self._waiting(note, now)}
        for drum in self.drums:
            x0 = int(round(_lane_x(drum.x, 0.0, width)))
            x1 = int(round(drum.x))
            y0 = int(round(top))
            y1 = int(round(hit_y))
            cv2.line(canvas, (x0, y0), (x1, y1), _dim(drum.color, 0.75), 3, cv2.LINE_AA)
            ring = max(7, int(round(mark * 0.78)))
            cv2.circle(canvas, (x1, y1), ring, drum.color if drum.lane in hot else _dim(drum.color, 0.85), 3, cv2.LINE_AA)
        cv2.line(canvas, (int(round(near_l)), int(round(hit_y))), (int(round(near_r)), int(round(hit_y))), (235, 235, 240), 3, cv2.LINE_AA)
        for note in self.notes:
            self._draw_gem(canvas, note, now)

    def _draw_gem(self, canvas: np.ndarray, note: DrumNote, now: float) -> None:
        drum = self.drums[note.lane]
        travel = _travel(note, now)
        x, y = self.note_xy(note, now)
        radius = _gem_radius(canvas.shape[0], travel)
        center = (int(round(x)), int(round(y)))
        cv2.circle(canvas, center, radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, radius, drum.color, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, max(2, radius // 3), (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_drums(self, canvas: np.ndarray, now: float) -> None:
        hot = {note.lane for note in self.notes if self._waiting(note, now)}
        for drum in self.drums:
            _draw_drum(canvas, drum, drum.lane in hot)

    def _limits(self) -> Tuple[float, float]:
        return self.difficulty.windows()

    def _waiting(self, note: DrumNote, now: float) -> bool:
        early, late = self._limits()
        return _in_window(note, now, early, late)

    def _score_bar(self):
        if self._score_view is None:
            from gamConsole.gurli_gris import BubbleRain

            self._score_view = BubbleRain()
        return self._score_view


class DrumHeroPlayer:
    """Scene, trommeslager og trommesæt. Kameraet bruges kun til silhuetten."""

    def __init__(self) -> None:
        self._pose = PoseBodyOverlay()
        self.drummer = Drummer()
        self.kit = DrumKit()
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
        self.kit.tick(self.drummer, pose, width, height, now)
        self.kit.paint(scene, self.drummer, pose, now, hud=False)
        #13 Punktsilhuetten er debug ovenpå, som i Gurli og Humdingers.
        self._pose.draw(scene, pose)
        self.kit.draw_score(scene)
        return scene


class DrumHeroGame(RoundScoreGame, WebcamARGame):
    """Trommeslageren står på scenen. Kameraet bruges kun til silhuetten."""

    window_name = "gamConsole: Drum Hero"

    def __init__(self, camera: int, scores: Optional[HighScoreList] = None) -> None:
        super().__init__(camera)
        self.backdrop = StageBackdrop()
        self.player = DrumHeroPlayer()
        #16 Listen står, til der trykkes Spil.
        self.setup = DrumSetup()
        #24 Sangen venter på alle numre, så træffet rammer filens sekund.
        self._music_from: Optional[float] = None
        #15 Upbeat Instrumental kører kun, mens runden spilles.
        self.track = DrumTrack(self.setup.song.path)
        #13 Samme 3 minutter og highscore som Gurli, i Drum Heros egen fil.
        board = scores if scores is not None else HighScoreList(highscore_file("drum-hero"))
        self.match = RoundScore(board)

    def close(self) -> None:
        #15 Menuen og lukket vindue stopper sangen.
        self.track.stop()
        self.player.close()

    def play_again(self) -> None:
        #16 Spil igen går tilbage til listen, så nummer og sværhedsgrad kan skiftes.
        self.track.stop()
        self._music_from = None
        self.player.kit.reset()
        self.match.clear()
        self.setup.open = True

    def begin_round(self) -> None:
        #16 Spil bruger det nummer og den sværhedsgrad, listen har valgt.
        #24 Ventetiden sættes på første frame, så noden og filens sekund starter sammen.
        self._music_from = None
        self.track.use(self.setup.song.path)
        self.player.kit.use(self.setup.difficulty, self.setup.song)
        self.setup.open = False

    def _start_track(self, moment: float) -> None:
        #24 Lyden starter LEAD sekunder inde i runden. Første sekund i filen
        # høres, når en node fra sekund 0 ville nå stregen.
        if self._music_from is None:
            self._music_from = moment + LEAD
        if moment >= self._music_from:
            self.track.ensure_playing()

    def on_click(self, x: int, y: int) -> None:
        if self.setup.open:
            if self.setup.press(x, y) == "play":
                self.begin_round()
            return
        self.match.on_click(x, y)

    def on_frame(self, frame: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        #13 Spejlvendt webcam, samme retning som Gurli og Humdingers.
        camera = cv2.flip(frame, 1)
        moment = time.monotonic() if now is None else now
        if self.setup.open:
            #16 Uret og sangen venter, mens listen er fremme.
            self.track.stop()
            return self.setup.draw(camera.shape[1], camera.shape[0])
        shown = self.match.begin(
            moment,
            self.player.kit.score,
            self.play_again,
            lambda: self.backdrop.render(camera.shape[1], camera.shape[0]),
        )
        if shown is not None:
            #15 Resultatskærmen er stille. Næste runde starter sangen forfra.
            self.track.stop()
            self._music_from = None
            return shown
        self._start_track(moment)
        self.player.refresh_pose(camera)
        height, width = camera.shape[:2]
        scene = self.player.composite(camera, self.backdrop.render(width, height), moment)
        return self.match.stamp(scene, moment)


def _paint_stage(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), np.uint8)
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)
    frame[:, :, 0] = (18 + ramp * 14)[:, None]
    frame[:, :, 1] = (8 + ramp * 10)[:, None]
    frame[:, :, 2] = (26 + ramp * 8)[:, None]
    lights = frame.copy()
    cv2.ellipse(
        lights,
        (int(width * 0.30), 0),
        (max(8, int(width * 0.16)), max(8, int(height * 0.55))),
        18,
        0,
        360,
        (70, 36, 28),
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        lights,
        (int(width * 0.72), 0),
        (max(8, int(width * 0.16)), max(8, int(height * 0.55))),
        -18,
        0,
        360,
        (36, 28, 64),
        -1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(lights, 0.30, frame, 0.70, 0, frame)
    stage = StageFrame(width, height)
    cv2.rectangle(frame, (0, stage.floor_top), (width, height), (42, 74, 128), -1)
    cv2.line(frame, (0, stage.floor_top), (width, stage.floor_top), (150, 186, 220), 3, cv2.LINE_AA)
    for index in range(1, 5):
        y = stage.floor_top + int((height - stage.floor_top) * index / 5)
        cv2.line(frame, (0, y), (width, y), (28, 52, 96), 1, cv2.LINE_AA)
    speaker_y = stage.floor_top + max(6, int(height * 0.04))
    speaker_h = max(16, int(height * 0.16))
    _speaker(frame, int(width * 0.03), speaker_y, max(18, int(width * 0.07)), speaker_h)
    _speaker(frame, int(width * 0.90), speaker_y, max(18, int(width * 0.07)), speaker_h)
    return frame


def _speaker(frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (22, 22, 26), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 78), 2, cv2.LINE_AA)
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(frame, (cx, cy), max(4, min(w, h) // 4), (48, 48, 54), -1, cv2.LINE_AA)


def _travel(note: DrumNote, now: float) -> float:
    remain = note.hit_at - now
    return max(0.0, min(1.0, 1.0 - remain / LEAD))


def _lane_x(x: float, travel: float, width: int) -> float:
    #14 Banen snævrer ind opad, og noden rammer trommens x på stregen.
    center = width * 0.5
    far = center + (x - center) * LANE_FAR
    return far + (x - far) * travel


def _draw_kit_hardware(canvas: np.ndarray, drums: List[DrumPad]) -> None:
    if len(drums) != DRUM_COUNT:
        return
    hihat, snare, tom, crash = drums
    _stand(canvas, hihat)
    _stand(canvas, crash)
    bass_x = (snare.x + tom.x) / 2.0
    bass_y = max(snare.y, tom.y) + snare.radius * 0.15
    radius = int(round(snare.radius * 1.35))
    center = (int(round(bass_x)), int(round(bass_y)))
    cv2.circle(canvas, center, radius, (24, 24, 28), -1, cv2.LINE_AA)
    cv2.circle(canvas, center, radius, (150, 150, 158), 3, cv2.LINE_AA)
    cv2.circle(canvas, center, max(4, radius // 3), (70, 70, 76), 2, cv2.LINE_AA)


def _stand(canvas: np.ndarray, drum: DrumPad) -> None:
    foot_y = int(round(drum.y + drum.radius * 2.2))
    x = int(round(drum.x))
    cv2.line(canvas, (x, int(round(drum.y))), (x, foot_y), (90, 90, 98), 3, cv2.LINE_AA)
    cv2.line(canvas, (x - drum.radius, foot_y), (x + drum.radius, foot_y), (90, 90, 98), 3, cv2.LINE_AA)


def _drums_around(drummer: Drummer, box: Tuple[int, int, int, int], height: int) -> List[DrumPad]:
    left, right = drummer.avatar.stick_tips(None, box)
    span = float(box[3] - box[1])
    #25 Lilletromme og tam har deres egen, større radius.
    radii = (
        max(12, int(round(height * DRUM_RADIUS))),
        max(18, int(round(height * LOW_DRUM_RADIUS))),
        max(18, int(round(height * LOW_DRUM_RADIUS))),
        max(12, int(round(height * DRUM_RADIUS))),
    )
    spots = (
        (left[0] + span * HIHAT_X, left[1] + span * HIHAT_Y),
        (left[0] + span * SNARE_X, left[1] + span * SNARE_Y),
        (right[0] - span * SNARE_X, right[1] + span * SNARE_Y),
        (right[0] - span * HIHAT_X, right[1] + span * HIHAT_Y),
    )
    return [
        DrumPad(lane, spot[0], spot[1], radii[lane], DRUM_COLORS[lane], DRUM_KINDS[lane])
        for lane, spot in enumerate(spots)
    ]


def _highway_radius(height: int) -> int:
    #27 Samme størrelse som bækkenerne havde, før bundtrommerne blev større.
    return max(12, int(round(height * DRUM_RADIUS)))


def _gem_radius(height: int, travel: float) -> int:
    return max(5, int(round(_highway_radius(height) * (0.34 + 0.48 * travel))))


def _stick_reach(drummer: Drummer, pose: Optional[BodyPose], width: int, height: int) -> float:
    del drummer, pose, width
    return max(6.0, height * STICK_REACH)


def _in_window(note: DrumNote, now: float, early: float = HIT_EARLY, late: float = HIT_LATE) -> bool:
    return note.hit_at - early <= now <= note.hit_at + late


def _reaches(stick: Tuple[float, float], drum: DrumPad, pad: float) -> bool:
    return math.hypot(stick[0] - drum.x, stick[1] - drum.y) <= drum.radius + pad


def _dim(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(int(channel * factor) for channel in color)  # type: ignore[return-value]


def _neon(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    #32 Den stærkeste kanal trækkes op, så ringen lyser.
    peak = max(color)
    if peak < 1:
        return color
    return tuple(min(255, int(channel * 255 / peak)) for channel in color)  # type: ignore[return-value]


def _neon_disk(
    canvas: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    strength: float,
) -> None:
    if radius < 2:
        return
    x, y = center
    height, width = canvas.shape[:2]
    pad = max(2, radius // 4)
    x0, x1 = max(0, x - radius - pad), min(width, x + radius + pad + 1)
    y0, y1 = max(0, y - radius - pad), min(height, y + radius + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return
    layer = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    cv2.circle(layer, (x - x0, y - y0), radius, _neon(color), -1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), max(1.0, radius * 0.12))
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(roi + layer * strength, 0, 255).astype(np.uint8)


class LowDrumPicture:
    """Billedet af lilletromme og tam. Hvid baggrund er væk."""

    def __init__(self, path: Path) -> None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Kan ikke læse {path}")
        self.bgr, self.alpha = _cut_white_background(image)
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def draw(self, canvas: np.ndarray, x: float, y: float, width: int) -> None:
        sprite = self._sprite(max(1, width))
        bgr, alpha = sprite
        left = int(round(x - bgr.shape[1] / 2))
        top = int(round(y - bgr.shape[0] / 2))
        ChaseInCar._blit(canvas, bgr, alpha, left, top)

    def _sprite(self, width: int) -> Tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(width)
        if cached is not None:
            return cached
        height = max(1, int(round(width * self.bgr.shape[0] / self.bgr.shape[1] * LOW_DRUM_FLAT)))
        bgr = cv2.resize(self.bgr, (width, height), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(self.alpha, (width, height), interpolation=cv2.INTER_AREA)
        #31 Samme fem pixels som på trommeslageren.
        sprite = _shave_edge(bgr, alpha, 5)
        self._cache[width] = sprite
        return sprite


_LOW_DRUM: Optional[LowDrumPicture] = None


def _low_drum() -> LowDrumPicture:
    global _LOW_DRUM
    if _LOW_DRUM is None:
        _LOW_DRUM = LowDrumPicture(LOW_DRUM_PATH)
    return _LOW_DRUM


def _cymbal_glow(alpha: np.ndarray) -> np.ndarray:
    #33 Skiven fylder sin form, så 1.12 kun gav et par pixels. Randen udvides, til den ses.
    height, width = alpha.shape
    solid = (alpha > 0.15).astype(np.uint8)
    reach = max(2, int(round(0.22 * max(height, width) / 2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach * 2 + 1, reach * 2 + 1))
    grown = cv2.dilate(solid, kernel).astype(np.float32)
    sigma = max(1.0, reach * 0.45)
    pad = max(2, int(round(sigma * 3)))
    padded = cv2.copyMakeBorder(grown, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    blurred = cv2.GaussianBlur(padded, (0, 0), sigma)
    peak = float(blurred.max())
    if peak > 0:
        blurred /= peak
    return blurred


def _add_neon_mask(
    canvas: np.ndarray,
    mask: np.ndarray,
    x: float,
    y: float,
    color: Tuple[int, int, int],
    strength: float,
) -> None:
    glow_h, glow_w = mask.shape
    left = int(round(x - glow_w / 2))
    top = int(round(y - glow_h / 2))
    height, width = canvas.shape[:2]
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + glow_w), min(height, top + glow_h)
    if x1 <= x0 or y1 <= y0:
        return
    piece = mask[y0 - top:y0 - top + (y1 - y0), x0 - left:x0 - left + (x1 - x0)]
    tint = np.array(_neon(color), np.float32) * strength
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(roi + piece[:, :, None] * tint, 0, 255).astype(np.uint8)


#33 Billedet er bækkenet. Neonmasken er billedets form, lidt større.
class CymbalPicture:
    """Det gule bækken med knoppen. Neonet er banens farve."""

    def __init__(self, path: Path) -> None:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"Kan ikke læse {path}")
        self.bgr = image[:, :, :3]
        self.alpha = image[:, :, 3].astype(np.float32) / 255.0
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def draw(
        self,
        canvas: np.ndarray,
        x: float,
        y: float,
        width: int,
        color: Tuple[int, int, int],
        strength: float,
    ) -> None:
        bgr, alpha, glow = self._sprite(max(1, width))
        _add_neon_mask(canvas, glow, x, y, color, strength)
        left = int(round(x - bgr.shape[1] / 2))
        top = int(round(y - bgr.shape[0] / 2))
        ChaseInCar._blit(canvas, bgr, alpha, left, top)

    def _sprite(self, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get(width)
        if cached is not None:
            return cached
        height = max(1, int(round(width * self.bgr.shape[0] / self.bgr.shape[1])))
        bgr = cv2.resize(self.bgr, (width, height), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(self.alpha, (width, height), interpolation=cv2.INTER_AREA)
        sprite = (bgr, alpha, _cymbal_glow(alpha))
        self._cache[width] = sprite
        return sprite


_CYMBAL: Optional[CymbalPicture] = None


def _cymbal() -> CymbalPicture:
    global _CYMBAL
    if _CYMBAL is None:
        _CYMBAL = CymbalPicture(CYMBAL_PATH)
    return _CYMBAL


def _draw_drum(canvas: np.ndarray, drum: DrumPad, hot: bool) -> None:
    center = (int(round(drum.x)), int(round(drum.y)))
    if drum.kind == "cymbal":
        #33 Billedet ligger over neonet. Styrken er den samme som på trommerne.
        _cymbal().draw(
            canvas,
            drum.x,
            drum.y,
            max(1, int(round(drum.radius * CYMBAL_WIDTH))),
            drum.color,
            NEON_HOT if hot else NEON_STRENGTH,
        )
        return
    #32 Neonring tæt på trommen. Scenen ses igennem den.
    light = max(4, int(round(drum.radius * LOW_DRUM_LIGHT)))
    _neon_disk(canvas, center, light, drum.color, NEON_HOT if hot else NEON_STRENGTH)
    _low_drum().draw(canvas, drum.x, drum.y, drum.radius * 2)


def _point_gain(x: float, y: float, value: int):
    from gamConsole.gurli_gris import PointGain

    return PointGain(x, y, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="gamConsole: Drum Hero")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    game = DrumHeroGame(args.camera)
    try:
        game.run()
    finally:
        game.close()


if __name__ == "__main__":
    main()
