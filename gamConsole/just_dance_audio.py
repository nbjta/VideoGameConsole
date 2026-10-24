"""Just Dance-sang. playsound giver ikke afspilleren tilbage, så den kan ikke stoppes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol


class SongSound(Protocol):
    def play(self) -> None: ...

    def stop(self) -> None: ...


class _NSSong:
    def __init__(self, sound: object) -> None:
        self._sound = sound

    def play(self) -> None:
        self._sound.play()

    def stop(self) -> None:
        self._sound.stop()


class SongPlayback:
    """Holder NSSound, så sangen kan stoppes når spillet forlades."""

    def __init__(self) -> None:
        self._sound: Optional[SongSound] = None

    def play(self, path: Path) -> None:
        sound = self._open(Path(path))
        if sound is None:
            raise FileNotFoundError(path)
        self.stop()
        self._sound = sound
        sound.play()

    def stop(self) -> None:
        sound = self._sound
        if sound is None:
            return
        try:
            sound.stop()
        finally:
            self._sound = None

    def _open(self, path: Path) -> Optional[SongSound]:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            return None
        from AppKit import NSSound
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(str(resolved))
        nssound = NSSound.alloc().initWithContentsOfURL_byReference_(url, True)
        if nssound is None:
            return None
        return _NSSong(nssound)
