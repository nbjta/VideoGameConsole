#!/usr/bin/env bash
#3 Nostalgia AR fra https://github.com/devgaddebuilds/wiisports
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/vendor/wiisports"
REPO="https://github.com/devgaddebuilds/wiisports.git"

if ! command -v git >/dev/null 2>&1; then
  echo "git mangler." >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "npm mangler. Nostalgia AR er et Vite-projekt." >&2
  exit 1
fi

if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 "$REPO" "$DEST"
else
  echo "Opdaterer eksisterende clone …"
  git -C "$DEST" pull --ff-only || true
fi

# Lobbyens PLAY-knapper peger som standard på /pingpong/ og /fruit/, som 404'er lokalt.
cat > "$DEST/nostalgia-ar/.env.local" <<'EOF'
VITE_PINGPONG_URL=http://127.0.0.1:5173/
VITE_FRUIT_URL=http://127.0.0.1:5174/
VITE_LOBBY_URL=http://127.0.0.1:5180/
EOF

write_mediapipe_headers() {
  local dir="$1"
  local cfg="$dir/vite.config.ts"
  if [[ -f "$cfg" ]]; then
    echo "Beholder eksisterende $cfg"
    return
  fi
  cat > "$cfg" <<'EOF'
import { defineConfig } from "vite";

// Samme COOP/COEP som upstream vercel.json, ellers starter MediaPipe WASM ikke.
export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
});
EOF
}

write_mediapipe_headers "$DEST/backend_handtracking"
write_mediapipe_headers "$DEST/frontend_game"

for app in nostalgia-ar backend_handtracking frontend_game; do
  echo "npm install i $app …"
  npm install --prefix "$DEST/$app"
done

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
"$PYTHON" -c "import sys; sys.path.insert(0, '$ROOT'); from gamConsole.wiisports_launcher import ensure_offline_supabase; ensure_offline_supabase()"

echo "OK: $DEST — menupunkt: Nostalgia AR (lobby :5180, bordtennis :5173, frugt :5174)"
