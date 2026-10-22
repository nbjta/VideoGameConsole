#!/usr/bin/env bash
# Homebrew macOS: OpenCV 5 exposes opencv5.pc; upstream Makefile expects opencv + globs .h files.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/vendor/webcam-games"

if [[ ! -f "$SRC/src/main.cpp" ]]; then
  echo "Mangler $SRC — kør scripts/setup_webcam_games.sh først." >&2
  exit 1
fi

export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}"
for pc in /opt/homebrew/opt/opencv/lib/pkgconfig /usr/local/opt/opencv/lib/pkgconfig; do
  if [[ -d "$pc" ]]; then
    export PKG_CONFIG_PATH="$pc${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  fi
done

OPENCV_PKG=""
for name in opencv4 opencv5 opencv; do
  if pkg-config --exists "$name" 2>/dev/null; then
    OPENCV_PKG="$name"
    break
  fi
done

if [[ -z "$OPENCV_PKG" ]]; then
  echo "OpenCV ikke fundet via pkg-config. Sæt PKG_CONFIG_PATH eller installer: brew install opencv" >&2
  exit 1
fi

GL_LIBS="$(pkg-config --cflags --libs gl 2>/dev/null || true)"
GLUT_CFLAGS=""
GLUT_LIBS="-lglut"
if pkg-config --exists freeglut 2>/dev/null; then
  GLUT_CFLAGS="$(pkg-config --cflags freeglut)"
  GLUT_LIBS="$(pkg-config --libs freeglut)"
elif [[ -d /opt/homebrew/opt/freeglut ]]; then
  GLUT_CFLAGS="-I/opt/homebrew/opt/freeglut/include"
  GLUT_LIBS="-L/opt/homebrew/opt/freeglut/lib -lglut"
fi

CPP_FILES=()
while IFS= read -r f; do
  CPP_FILES+=("$f")
done < <(find "$SRC/src" -name '*.cpp' | sort)
if [[ ${#CPP_FILES[@]} -eq 0 ]]; then
  echo "Ingen .cpp filer i $SRC/src" >&2
  exit 1
fi

echo "Bygger CvGames (opencv pkg: $OPENCV_PKG, ${#CPP_FILES[@]} kilder) …"
g++ -std=c++11 \
  "${CPP_FILES[@]}" \
  $GLUT_CFLAGS \
  $(pkg-config --cflags --libs "$OPENCV_PKG") \
  $GL_LIBS \
  $GLUT_LIBS \
  -o "$SRC/CvGames"

echo "OK: $SRC/CvGames"
