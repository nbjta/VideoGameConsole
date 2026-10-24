"""Just Dance i samme vindue (ét frame ad gangen)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from gamConsole.just_dance_audio import SongPlayback
from gamConsole.just_dance_score import MoveScore
from gamConsole.just_dance_songs import DANCE_SONGS, resolve_song, song_title
from gamConsole.just_dance_stage import JustDanceStage
from gamConsole.just_dance_tempo import DanceTempo
from gamConsole.just_dance_tracks import TrackList
from gamConsole.round_score import HighScoreList, highscore_file

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor" / "just-dance"


class JustDanceSession:
    title = "Just Dance"

    def __init__(self, camera: int, scores: Optional[HighScoreList] = None) -> None:
        self.song = ""
        self.camera = camera
        self._game = None
        self._ready = False
        self._picking = True
        self._pending: Optional[str] = None
        self._error: Optional[str] = None
        self._stage = JustDanceStage()
        self._playback = SongPlayback()
        self._tracks = TrackList()
        self._tempo: Optional[DanceTempo] = None
        self._frame1: Optional[np.ndarray] = None
        self._last_scene: Optional[np.ndarray] = None
        board = scores if scores is not None else HighScoreList(highscore_file("just_dance"))
        self.moves = MoveScore(board)
        self.title = "Just Dance"

    def _load(self) -> None:
        self.moves.reset_attempt()
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
            self._playback.play(REPO / "songs_audio" / f"{self.song}.mp3")
            self._tempo = DanceTempo(self._game.controller.frame1_rate)
            self.title = f"Just Dance — {song_title(self.song)}"
            self._ready = True
        except Exception as exc:  # noqa: BLE001 — vis fejl i menu
            self._error = str(exc)

    def finalize_score(self) -> Optional[int]:
        if self._game is None and self.moves.score == 0 and not self.moves.saved:
            return None
        if not self.moves.saved and self.moves.score > 0:
            self.moves.match.scores.add("", self.moves.score)
            self.moves.saved = True
        if not self.moves.saved:
            return None
        return int(self.moves.score)

    def stop(self) -> None:
        self._playback.stop()
        self._stage.close()
        if self._game is None:
            return
        try:
            self._game.controller.release_capture()
        except Exception:
            pass

    def claims_key(self, key: int) -> bool:
        return self.moves.match.accepts_name_key(key)

    def on_click(self, x: int, y: int) -> None:
        if self.moves.match.result is not None:
            self.moves.match.on_click(x, y)
            return
        song_id = self._tracks.hit(x, y)
        if song_id is None:
            return
        if song_id == self.song and not self._picking:
            return
        self._pending = song_id

    def on_key(self, key: int) -> None:
        if self.moves.match.result is not None:
            self.moves.match.on_key(key)
            return
        if ord("1") <= key <= ord("5"):
            index = key - ord("1")
            if index < len(DANCE_SONGS):
                self.on_click_song(DANCE_SONGS[index].id)

    def on_click_song(self, song_id: str) -> None:
        if song_id == self.song and not self._picking:
            return
        self._pending = song_id

    def tick(self) -> Optional[np.ndarray]:
        result = self.moves.result_frame()
        if result is not None:
            return result
        if self._pending:
            song_id = self._pending
            self._pending = None
            self._begin(song_id)
        if self._error:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, self._error, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 120), 2)
            return frame
        if self._picking or not self._ready or self._game is None:
            return self._tracks.render(current=self.song or None)

        ctrl = self._game.controller
        tempo = self._tempo
        if tempo is None or not ctrl.cap1.isOpened():
            return None
        frame1 = self._pull_video_frame(ctrl.cap1, tempo)
        if frame1 is None:
            self._playback.stop()
            scene = self._last_scene
            if scene is None:
                scene = np.zeros((480, 640, 3), dtype=np.uint8)
            return self.moves.finish(scene, self._play_again)
        ok2, frame2 = ctrl.cap2.read()
        camera_ok = ok2 and frame2 is not None
        if not camera_ok:
            frame2 = np.zeros_like(frame1)
        else:
            frame2 = cv2.flip(frame2, 1)

        gained = 0
        # Point kun for det billede, sangen er nået til, og kun når kameraet ser spilleren.
        if camera_ok and tempo.should_score():
            kpv = ctrl.process_frame(frame1)
            kpc = ctrl.process_frame(frame2)
            ctrl.model.store_angles(ctrl.angles_video, frame2, kpv)
            ctrl.model.store_angles(ctrl.angles_camera, frame2, kpc)
            gained = self.moves.award(ctrl.angles_video, ctrl.angles_camera)

        if camera_ok:
            frame2 = self._stage.composite(frame2)

        h1 = frame1.shape[0]
        h2, w2 = frame2.shape[:2]
        if h1 != h2:
            w2 = int(w2 * (h1 / h2))
            frame2 = cv2.resize(frame2, (w2, h1))
        combined = np.concatenate((frame1, frame2), axis=1)
        if gained:
            self.moves.place_latest(frame1.shape[1] + frame2.shape[1] / 2.0, combined.shape[0] * 0.38)
        self.moves.tick(time.monotonic())
        self.moves.draw(combined)
        self._tracks.draw_bar(combined, self.song)
        self._last_scene = combined
        return combined

    def _begin(self, song_id: str) -> None:
        self.song = resolve_song(song_id)
        self.title = f"Just Dance — {song_title(self.song)}"
        self._picking = False
        self._error = None
        if self._game is None:
            self._load()
            return
        self._switch_video()

    def _play_again(self) -> None:
        self._playback.stop()
        self.moves.reset_attempt()
        self._picking = True
        self._frame1 = None
        self._tempo = None
        self._last_scene = None

    def _switch_video(self) -> None:
        self.moves.reset_attempt()
        ctrl = self._game.controller
        self._playback.stop()
        if ctrl.cap1 is not None:
            ctrl.cap1.release()
        video = REPO / "songs" / f"{self.song}.mp4"
        ctrl.cap1 = cv2.VideoCapture(str(video))
        ctrl.frame1_rate = ctrl.cap1.get(cv2.CAP_PROP_FPS) or 30.0
        for angles in (ctrl.angles_video, ctrl.angles_camera):
            for key in angles:
                angles[key].clear()
        self._frame1 = None
        self._tempo = DanceTempo(ctrl.frame1_rate)
        if not ctrl.cap1.isOpened():
            self._error = f"Kan ikke åbne {song_title(self.song)}"
            self._picking = True
            return
        self._playback.play(REPO / "songs_audio" / f"{self.song}.mp3")
        self._ready = True
        self._picking = False

    def _pull_video_frame(self, capture: cv2.VideoCapture, tempo: DanceTempo) -> Optional[np.ndarray]:
        to_read = tempo.frames_to_read()
        if to_read == 0 and self._frame1 is not None:
            return self._frame1
        frame = None
        for _ in range(max(to_read, 1)):
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            tempo.note_read()
        self._frame1 = frame
        return frame
