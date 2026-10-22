import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

REPO_DIR = Path(__file__).resolve().parents[1] / "vendor" / "webcam-games"


def cvgames_binary() -> Optional[Path]:
    candidates = [
        REPO_DIR / "CvGames",
        REPO_DIR / "cmake-build-debug" / "CvGames",
        REPO_DIR / "build" / "CvGames",
    ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def is_webcam_games_installed() -> bool:
    return REPO_DIR.is_dir() and (REPO_DIR / "src" / "main.cpp").is_file()


def build_instructions() -> str:
    return (
        "Webcam-games (CvGames) er ikke klar endnu.\n\n"
        f"Kør fra projektroden:\n  ./scripts/setup_webcam_games.sh\n\n"
        "Kræver: Xcode CLI tools, OpenCV, GLUT, pkg-config.\n"
        "Kilde: vendor/webcam-games\n"
        "Se README.md for detaljer."
    )


def launch_cvgames(mode: str, camera: int = 0) -> Tuple[bool, str]:
    binary = cvgames_binary()
    if binary is None:
        if not is_webcam_games_installed():
            return False, build_instructions()
        return (
            False,
            "Kilde findes, men CvGames er ikke bygget.\n\n"
            "Kør: ./scripts/setup_webcam_games.sh\n"
            "eller: cd vendor/webcam-games && make",
        )

    cmd: List[str] = [str(binary), mode, f"--camera={camera}"]
    env = os.environ.copy()
    data_dir = REPO_DIR / "Data"
    if data_dir.is_dir():
        env["CVGAMES_DATA"] = str(data_dir)

    try:
        subprocess.Popen(
            cmd,
            cwd=str(REPO_DIR),
            env=env,
            stdout=None,
            stderr=None,
        )
    except OSError as exc:
        return False, f"Kunne ikke starte CvGames: {exc}"
    return True, f"Startede {binary.name} {mode} (kamera {camera})."


def open_repo_in_browser() -> None:
    import webbrowser

    webbrowser.open("https://github.com/aurelijusb/webcam-games")
