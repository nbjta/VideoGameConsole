#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/vendor/just-dance"
REPO="https://github.com/olincollege/just-dance.git"

if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 "$REPO" "$DEST"
else
  git -C "$DEST" pull --ff-only || true
fi

echo "Installer afhængigheder (kan tage lang tid — TensorFlow) …"
"$ROOT/.venv/bin/pip" install -r "$DEST/requirements.txt" 2>/dev/null || pip3 install -r "$DEST/requirements.txt"

echo "OK: $DEST — start fra menu eller: cd vendor/just-dance && python3 just_dance_gui.py"
