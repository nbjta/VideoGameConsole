from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass(frozen=True)
class GameEntry:
    id: str
    title: str
    description: str
    mode_arg: Optional[str]  # CvGames / builtin mode, eller ekstern nøgle
    source_url: str
    launcher: str = "cvgames"  # cvgames | builtin | just_dance | jdwebcam


WEBCAM_GAMES_REPO = "https://github.com/aurelijusb/webcam-games"

GAMES: List[GameEntry] = [
    GameEntry(
        id="webcam-3d",
        title="3D rotation",
        description="Rotér 3D-objekter med kamerabevægelse (OpenCV + OpenGL).",
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
        id="webcam-vj",
        title="WebCam VJ",
        description="Motion-baseret billed-fading.",
        mode_arg="vj",
        source_url=WEBCAM_GAMES_REPO,
    ),
    GameEntry(
        id="webcam-debug",
        title="Debug view",
        description="Viser hvordan tracking virker internt.",
        mode_arg="debug",
        source_url=WEBCAM_GAMES_REPO,
    ),
    GameEntry(
        id="just-dance",
        title="Just Dance (Python)",
        description="Olincollege clone: dans med webcam og score (tung: TensorFlow + assets).",
        mode_arg="cheapthrills",
        source_url="https://github.com/olincollege/just-dance",
        launcher="just_dance",
    ),
    GameEntry(
        id="jdwebcam",
        title="JDWebcam controller",
        description="Webcam som virtuel controller til Just Dance 2017 PC (ikke et spil i sig selv).",
        mode_arg=None,
        source_url="https://github.com/comeraperuibe944/JDWebcam",
        launcher="jdwebcam",
    ),
]


def games_for_menu() -> List[GameEntry]:
    return list(GAMES)
