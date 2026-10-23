#!/usr/bin/env bash
#6 Handora fra https://github.com/BOSS294/Handora
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/vendor/handora"
REPO="https://github.com/BOSS294/Handora.git"

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

INDEX="$DEST/modules/hand_gesture/index.html"
if [[ ! -f "$INDEX" ]]; then
  echo "Mangler $INDEX efter clone." >&2
  exit 1
fi

echo "OK: $DEST — menupunkt: Handora (http://127.0.0.1:5190/modules/hand_gesture/index.html)"
