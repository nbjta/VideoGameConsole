"""Dansens videobillede følger sangens tid, også når et billede er sent færdigt."""

from __future__ import annotations

import time


class DanceTempo:
    def __init__(self, fps: float, now: float | None = None) -> None:
        self.fps = fps if fps and fps > 1.0 else 30.0
        self.t0 = time.perf_counter() if now is None else now
        self.shown = -1
        self._score_bucket = -1

    def due(self, now: float | None = None) -> int:
        current = time.perf_counter() if now is None else now
        return max(0, int((current - self.t0) * self.fps + 1e-6))

    def frames_to_read(self, now: float | None = None) -> int:
        """Antal read() der skal til for at vise det billede, sangen er nået til. 0 betyder behold."""
        due = self.due(now)
        if due <= self.shown:
            return 0
        return due - self.shown

    def note_read(self) -> None:
        self.shown += 1

    def should_score(self) -> bool:
        if self.shown < 0:
            return False
        bucket = self.shown // 100
        if bucket == self._score_bucket:
            return False
        self._score_bucket = bucket
        return True
