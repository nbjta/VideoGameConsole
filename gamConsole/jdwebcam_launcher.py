from pathlib import Path
from typing import Tuple

import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor" / "JDWebcam"


def is_installed() -> bool:
    return (REPO / "main.py").is_file()


def setup_hint() -> str:
    return "Kør: ./scripts/setup_jdwebcam.sh\nKræver Just Dance 2017 PC på netværk."


def launch(console_ip: str = "") -> Tuple[bool, str]:
    if not is_installed():
        return False, f"JDWebcam er ikke klonet.\n\n{setup_hint()}"
    cmd = [sys.executable, "main.py"]
    if console_ip:
        cmd.append(f"--{console_ip}")
    try:
        subprocess.Popen(cmd, cwd=str(REPO))
    except OSError as exc:
        return False, str(exc)
    return True, "Startede JDWebcam. Forbind til JD 2017 PC (smartphone-skærm)."
