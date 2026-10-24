#15
"""Upbeat Instrumental i Drum Hero.

Sub_Clair på Pixabay, nummer 588419. Fri under Pixabay Content License.
Filen er målt med afinfo: 108,048 sekunder, 256 kbps, 48 kHz, stereo.
#24 Noderne er denne fils afkodede slag i sekunder, ikke et fast BPM.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SONG_PATH = ROOT / "assets" / "drum_hero" / "upbeat-instrumental-588419.mp3"
SONG_TITLE = "Upbeat Instrumental"
SONG_ARTIST = "Sub_Clair"
SONG_PAGE = "https://pixabay.com/music/upbeat-upbeat-instrumental-588419/"


def _under_test() -> bool:
    #15 Samme regel som game_audio: unittest sender ikke lyd ud.
    return any("unittest" in arg for arg in sys.argv)


class DrumTrack:
    """Sangen kører, mens runden spiller, og stopper på resultatskærmen."""

    def __init__(self, path: Path = SONG_PATH) -> None:
        self.path = path
        self.started = False
        self._sent = False
        self._heard = False
        self._sound: Optional[object] = None

    def ensure_playing(self) -> None:
        self.started = True
        if _under_test():
            return
        sound = self._open()
        if sound is None:
            return
        if not self._sent:
            #15 BOOL loops i NSSound. Sangen gentages, til runden stopper den.
            sound.setLoops_(True)
            sound.play()
            self._sent = True
            return
        if bool(sound.isPlaying()):
            self._heard = True
            return
        if self._heard:
            self._heard = False
            sound.play()

    def use(self, path: Path) -> None:
        #16 Et andet nummer åbnes forfra. Samme nummer startes også forfra.
        same = path == self.path
        self.stop()
        if same:
            return
        self._sound = None
        self.path = path

    def stop(self) -> None:
        self.started = False
        self._sent = False
        self._heard = False
        sound = self._sound
        if sound is None:
            return
        sound.stop()

    def _open(self) -> Optional[object]:
        if self._sound is not None:
            return self._sound
        if not self.path.is_file():
            return None
        from AppKit import NSSound
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(str(self.path.resolve()))
        sound = NSSound.alloc().initWithContentsOfURL_byReference_(url, True)
        if sound is None:
            return None
        self._sound = sound
        return sound
