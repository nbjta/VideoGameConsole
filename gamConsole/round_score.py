#9 #1
"""3 minutter, highscore og Spil igen.

Gurli Gris og Humdingers killinger bruger de samme klasser.
Hvert spil har sin egen fil, så listerne ikke blandes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from gamConsole.game_audio import game_audio
from gamConsole.opencv_ui import draw_text, draw_text_centered, measure_text

#9 #1 Et spil varer 3 minutter. Listen viser de 10 bedste. Kun top 3 må skrive navn.
ROUND_SECONDS = 180.0
HIGHSCORE_SHOWN = 10
NAME_RANK = 3
NAME_MAX = 12
_NAME_EXTRA = {197, 198, 216, 229, 230, 248}


def highscore_file(game_id: str) -> Path:
    #9 #1 Filen ligger i hjemmemappen, ikke i spillets mappe.
    return Path.home() / ".gamconsole" / f"{game_id}_highscores.json"


def clock_text(remaining: float) -> str:
    shown = max(0, min(int(ROUND_SECONDS), int(math.ceil(remaining - 1e-9))))
    return f"{shown // 60}:{shown % 60:02d}"


def _is_name_key(key: int) -> bool:
    if key in (8, 127):
        return True
    if key == 32 or (ord("0") <= key <= ord("9")):
        return True
    if ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z"):
        return True
    return key in _NAME_EXTRA


class ScoreEntry:
    """Et gemt resultat. Tomt navn vises som en streg."""

    def __init__(self, name: str, score: int) -> None:
        self.name = name
        self.score = score

    def label(self) -> str:
        return self.name if self.name else "—"


class HighScoreList:
    """De bedste resultater, højeste score først."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: List[ScoreEntry] = []
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            self.entries = []
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.entries = []
            return
        loaded: List[ScoreEntry] = []
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                score = item.get("score")
                name = item.get("name")
                if isinstance(score, int) and not isinstance(score, bool) and isinstance(name, str):
                    loaded.append(ScoreEntry(name[:NAME_MAX], score))
        loaded.sort(key=lambda entry: -entry.score)
        self.entries = loaded[:HIGHSCORE_SHOWN]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"name": entry.name, "score": entry.score} for entry in self.entries]
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def rank(self, score: int) -> int:
        return 1 + sum(1 for entry in self.entries if entry.score > score)

    def add(self, name: str, score: int) -> Optional[ScoreEntry]:
        #9 #1 Pladsen afgøres før resultatet lægges ind, så spilleren ikke tæller sig selv.
        if self.rank(score) > HIGHSCORE_SHOWN:
            return None
        entry = ScoreEntry(name.strip()[:NAME_MAX], score)
        self.entries.append(entry)
        self.entries.sort(key=lambda item: -item.score)
        self.entries = self.entries[:HIGHSCORE_SHOWN]
        self.save()
        return entry

    def remember(self, entry: ScoreEntry, name: str) -> None:
        entry.name = name.strip()[:NAME_MAX]
        self.save()


class TimedRound:
    """3 minutter fra første billede. Uret starter ikke, mens spillet indlæses."""

    def __init__(self) -> None:
        self.started_at: Optional[float] = None

    def start(self, now: float) -> None:
        if self.started_at is None:
            self.started_at = now

    def remaining(self, now: float) -> float:
        if self.started_at is None:
            return ROUND_SECONDS
        return max(0.0, ROUND_SECONDS - (now - self.started_at))

    def over(self, now: float) -> bool:
        return self.started_at is not None and now - self.started_at >= ROUND_SECONDS


class PlayButton:
    """Grøn knap nederst. Den ligger over LUK-bjælken, når spillet kører alene."""

    def __init__(self) -> None:
        self.rect: Optional[Tuple[int, int, int, int]] = None

    def layout(self, width: int, height: int) -> Tuple[int, int, int, int]:
        button_h = 52
        button_w = min(260, max(120, width - 24))
        y2 = height - 58
        y1 = y2 - button_h
        x1 = max(0, (width - button_w) // 2)
        self.rect = (x1, y1, x1 + button_w, y2)
        return self.rect

    def contains(self, x: int, y: int) -> bool:
        if self.rect is None:
            return False
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def draw(self, canvas: np.ndarray) -> None:
        height, width = canvas.shape[:2]
        x1, y1, x2, y2 = self.layout(width, height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (72, 210, 88), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (230, 255, 230), 2)
        text_h = measure_text("Spil igen", 0.72)[1]
        draw_text_centered(
            canvas,
            "Spil igen",
            (x1 + x2) // 2,
            y1 + (y2 - y1 - text_h) // 2,
            scale=0.72,
            color=(255, 255, 255),
        )


class ResultScreen:
    """Din score over highscore-listen. Top 3 kan skrive et navn. Spil igen starter forfra."""

    def __init__(self, scores: HighScoreList, score: int, play: Callable[[], None]) -> None:
        self.scores = scores
        self.score = score
        self.can_name = scores.rank(score) <= NAME_RANK
        self.entry = scores.add("", score)
        self._typed = ""
        self._play = play
        self.button = PlayButton()
        #9 #1 Kort jingle, når de 3 minutter eller livene er brugt op.
        game_audio.round_over()

    def score_line(self) -> str:
        return str(self.score)

    def highscore_lines(self) -> List[str]:
        lines: List[str] = []
        for index, entry in enumerate(self.scores.entries, start=1):
            lines.append(f"{index} {entry.label()} {entry.score}")
        return lines

    def accepts(self, key: int) -> bool:
        return self.can_name and _is_name_key(key)

    def on_key(self, key: int) -> None:
        if not self.accepts(key) or self.entry is None:
            return
        if key in (8, 127):
            self._typed = self._typed[:-1]
        elif len(self._typed) < NAME_MAX:
            self._typed += chr(key)
        self.scores.remember(self.entry, self._typed)

    def on_click(self, x: int, y: int) -> None:
        if self.button.contains(x, y):
            game_audio.click()
            self._play()

    def draw(self, canvas: np.ndarray) -> np.ndarray:
        height, width = canvas.shape[:2]
        veil = np.full_like(canvas, (32, 28, 24))
        cv2.addWeighted(veil, 0.62, canvas, 0.38, 0, dst=canvas)
        y = 28
        draw_text_centered(canvas, "Din score", width // 2, y, scale=0.7, color=(230, 230, 230))
        y += 36
        draw_text_centered(canvas, self.score_line(), width // 2, y, scale=1.15, color=(255, 255, 255))
        y += 58
        draw_text_centered(canvas, "Highscore", width // 2, y, scale=0.62, color=(210, 210, 220))
        y += 34
        button_top = self.button.layout(width, height)[1]
        name_reserve = 78 if self.can_name else 0
        for index, entry in enumerate(self.scores.entries, start=1):
            if y + 24 >= button_top - name_reserve:
                break
            color = (255, 230, 140) if entry is self.entry else (230, 230, 230)
            draw_text_centered(canvas, f"{index}  {entry.label()}  {entry.score}", width // 2, y, scale=0.55, color=color)
            y += 28
        if self.can_name:
            prompt_y = max(y + 8, button_top - 78)
            draw_text_centered(canvas, "Skriv dit navn", width // 2, prompt_y, scale=0.5, color=(200, 220, 255))
            shown = self._typed if self._typed else ""
            draw_text_centered(canvas, shown + "_", width // 2, prompt_y + 28, scale=0.7, color=(255, 255, 255))
        self.button.draw(canvas)
        return canvas


def draw_clock(canvas: np.ndarray, remaining: float) -> None:
    text = clock_text(remaining)
    scale = 0.75
    text_w, text_h = measure_text(text, scale)
    x = canvas.shape[1] - text_w - 16
    y = 10
    cv2.rectangle(canvas, (x - 10, y - 4), (x + text_w + 10, y + text_h + 6), (32, 36, 48), -1)
    draw_text(canvas, text, (x, y), scale=scale, color=(240, 240, 240))


class RoundScore:
    """Uret, den frosne sidste scene og resultatskærmen for ét spil."""

    def __init__(self, scores: HighScoreList) -> None:
        self.scores = scores
        self.round = TimedRound()
        self.result: Optional[ResultScreen] = None
        self._frozen: Optional[np.ndarray] = None

    def accepts_name_key(self, key: int) -> bool:
        return self.result is not None and self.result.accepts(key)

    def on_key(self, key: int) -> None:
        if self.result is not None:
            self.result.on_key(key)

    def on_click(self, x: int, y: int) -> None:
        if self.result is not None:
            self.result.on_click(x, y)

    def begin(
        self,
        now: float,
        score: int,
        play: Callable[[], None],
        fallback: Callable[[], np.ndarray],
    ) -> Optional[np.ndarray]:
        #9 #1 Allerede færdigt, eller de 3 minutter er gået, før dette billede spilles.
        if self.result is not None and self._frozen is not None:
            return self.result.draw(self._frozen.copy())
        if self.round.started_at is None:
            self.round.start(now)
            return None
        if not self.round.over(now):
            return None
        base = self._frozen if self._frozen is not None else fallback()
        return self.finish(base, score, play)

    def finish(self, scene: np.ndarray, score: int, play: Callable[[], None]) -> np.ndarray:
        self._frozen = scene.copy()
        self.result = ResultScreen(self.scores, score, play)
        return self.result.draw(self._frozen.copy())

    def stamp(self, scene: np.ndarray, now: float) -> np.ndarray:
        draw_clock(scene, self.round.remaining(now))
        self._frozen = scene.copy()
        return scene

    def clear(self) -> None:
        self.round = TimedRound()
        self.result = None
        self._frozen = None


class RoundScoreGame:
    """Taster og klik til resultatskærmen. Spillet skal have self.match."""

    match: RoundScore

    def on_key(self, key: int) -> None:
        self.match.on_key(key)

    def on_click(self, x: int, y: int) -> None:
        self.match.on_click(x, y)

    def accepts_name_key(self, key: int) -> bool:
        return self.match.accepts_name_key(key)


class RoundScoreSession:
    """Menuen sender klik og navnebogstaver videre. Sessionen skal have self.game."""

    def on_click(self, x: int, y: int) -> None:
        self.game.on_click(x, y)  # type: ignore[attr-defined]

    def claims_key(self, key: int) -> bool:
        #9 #1 q, h og m skal kunne stå i navnet.
        return self.game.accepts_name_key(key)  # type: ignore[attr-defined]
