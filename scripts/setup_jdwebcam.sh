#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/vendor/JDWebcam"
REPO="https://github.com/comeraperuibe944/JDWebcam.git"

if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 "$REPO" "$DEST"
else
  git -C "$DEST" pull --ff-only || true
fi

PIP="$ROOT/.venv/bin/pip"
if [[ ! -x "$PIP" ]]; then PIP="pip3"; fi
"$PIP" install websockets opencv-python mediapipe numpy "protobuf>=4"

echo "OK: $DEST — kræver Just Dance 2017 PC. Kør: cd vendor/JDWebcam && python main.py"
