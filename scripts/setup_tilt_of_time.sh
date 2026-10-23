#!/usr/bin/env bash
#2 Tilt of Time fra https://github.com/topics/webcam-game
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/vendor/tilt-of-time"
REPO="https://github.com/ebinrajiiit/Tilt_of_Time.git"

if ! command -v git >/dev/null 2>&1; then
  echo "git mangler." >&2
  exit 1
fi

if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 "$REPO" "$DEST"
else
  echo "Opdaterer eksisterende clone …"
  git -C "$DEST" pull --ff-only || true
fi

PY="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"
if [[ ! -x "$PY" ]]; then
  PY="python3"
  PIP="pip3"
fi

"$PIP" install "mediapipe>=0.10" "numpy>=1.24" "Pillow>=10.0"

# MediaPipe 1.1.1 afhænger af opencv-contrib-python 5, som erstatter cv2.
# gamConsole kræver opencv-python 4.x. MediaPipe importerer stadig med 4.14.
if "$PIP" show opencv-contrib-python >/dev/null 2>&1; then
  "$PIP" uninstall -y opencv-contrib-python
fi
if ! "$PY" -c "import cv2; raise SystemExit(0 if str(getattr(cv2, '__version__', '')).startswith('4.') else 1)"; then
  "$PIP" install --force-reinstall --no-deps "opencv-python>=4.8.0,<5"
fi
"$PY" -c "import cv2; assert str(cv2.__version__).startswith('4.'), cv2.__version__"

echo "OK: $DEST — menupunkt: Tilt of Time"
