#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor/webcam-games"
REPO="https://github.com/aurelijusb/webcam-games.git"

if ! command -v git >/dev/null 2>&1; then
  echo "git mangler." >&2
  exit 1
fi

if [[ ! -d "$VENDOR/.git" ]]; then
  mkdir -p "$(dirname "$VENDOR")"
  git clone --depth 1 "$REPO" "$VENDOR"
else
  echo "Opdaterer eksisterende clone …"
  git -C "$VENDOR" pull --ff-only || true
fi

if ! command -v pkg-config >/dev/null 2>&1; then
  echo ""
  echo "pkg-config findes ikke. På macOS:"
  echo "  brew install pkg-config opencv freeglut"
  echo ""
  exit 1
fi

echo "Bygger CvGames …"
"$ROOT/scripts/build_cvgames.sh"
