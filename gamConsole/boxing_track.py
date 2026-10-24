#20
"""Action Driving Rock som baggrund i Boksning.

Verclub_Music på Pixabay, nummer 623120. Fri under Pixabay Content License.
Filen er målt med afinfo: 115,540 sekunder, 256 kbps, 44,1 kHz, stereo.
Runden er 3 minutter, så sangen gentages.
"""

from __future__ import annotations

from pathlib import Path

from gamConsole.drum_track import DrumTrack

ROOT = Path(__file__).resolve().parents[1]
SONG_PATH = ROOT / "assets" / "boksning" / "action-driving-rock-623120.mp3"
SONG_TITLE = "Action Driving Rock"
SONG_ARTIST = "Verclub_Music"
SONG_PAGE = "https://pixabay.com/music/rock-action-driving-rock-623120/"


class BoxingTrack(DrumTrack):
    """Samme stoppelige afspilning som Drum Hero, med boksespillets fil."""

    def __init__(self) -> None:
        super().__init__(SONG_PATH)
