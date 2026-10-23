from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass(frozen=True)
class GameEntry:
    id: str
    title: str
    description: str
    mode_arg: Optional[str]  # CvGames / builtin mode, eller ekstern nøgle
    source_url: str
    launcher: str = "cvgames"  # cvgames | builtin | just_dance | tilt_of_time | wiisports | handora


WEBCAM_GAMES_REPO = "https://github.com/aurelijusb/webcam-games"

GAMES: List[GameEntry] = [
    #8
    GameEntry(
        id="webcam-3d",
        title="Husbygning",
        description="Sæt stolper, modulvægge, toprem, døre, vinduer og tag på spøgelses-mærkerne.",
        mode_arg="3d",
        source_url=WEBCAM_GAMES_REPO,
    ),
    GameEntry(
        id="webcam-bub",
        title="Bubble shot",
        description="Bevægelsesstyret boble-spil.",
        mode_arg="bub",
        source_url=WEBCAM_GAMES_REPO,
    ),
    GameEntry(
        id="webcam-debug",
        title="Debug view",
        description="Viser hvordan tracking virker internt.",
        mode_arg="debug",
        source_url=WEBCAM_GAMES_REPO,
    ),
    #1 #9
    GameEntry(
        id="flying-users-ar",
        title="Humdingers killinger (AR)",
        description="Kast burgere efter Humdingers killinger. Et spil varer 3 minutter.",
        mode_arg="flying_users",
        source_url="https://github.com/nbjta/VideoGameConsole",
        launcher="builtin",
    ),
    GameEntry(
        id="just-dance",
        title="Just Dance (Python)",
        description="Olincollege clone (standard: Cheap Thrills). TensorFlow — opstart ~30 sek.",
        mode_arg="cheapthrills",
        source_url="https://github.com/olincollege/just-dance",
        launcher="just_dance",
    ),
    #2
    GameEntry(
        id="tilt-of-time",
        title="Tilt of Time",
        description="Vip hovedet og fang kultur-ikoner med webcam.",
        mode_arg="tilt",
        source_url="https://github.com/ebinrajiiit/Tilt_of_Time",
        launcher="tilt_of_time",
    ),
    #3
    GameEntry(
        id="wiisports",
        title="Nostalgia AR",
        description="Bordtennis og frugt med webcam i browseren.",
        mode_arg="lobby",
        source_url="https://github.com/devgaddebuilds/wiisports",
        launcher="wiisports",
    ),
    #6
    GameEntry(
        id="handora",
        title="Handora",
        description="Hånd, ansigt og krop. Laser og Hit-The-Ball i browseren.",
        mode_arg="hand_gesture",
        source_url="https://github.com/BOSS294/Handora",
        launcher="handora",
    ),
    #9
    GameEntry(
        id="gurli-gris",
        title="Gurli Gris",
        description="Gurli i landskabet. Et spil varer 3 minutter. Vift en gren på en boble i 0,5 sekund.",
        mode_arg="gurli_gris",
        source_url="https://github.com/nbjta/VideoGameConsole",
        launcher="builtin",
    ),
]


def games_for_menu() -> List[GameEntry]:
    #9 Gurli og Humdingers forrest i karusellen.
    front_ids = ("gurli-gris", "flying-users-ar")
    front = [g for g in GAMES if g.id in front_ids]
    front.sort(key=lambda g: front_ids.index(g.id))
    rest = [g for g in GAMES if g.id not in front_ids]
    return front + rest
