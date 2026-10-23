# VideoGameConsole (gamConsole)

Menu-baseret “spilkonsol” til webcam-spil. Spillene fra [aurelijusb/webcam-games](https://github.com/aurelijusb/webcam-games) startes via den native `CvGames`-binary, når den er bygget.

## Hurtig start

```bash
cd /Users/Natalie/VideoGameConsole
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x run_gamConsole.sh scripts/setup_webcam_games.sh
./run_gamConsole.sh
```

Standard-UI er **OpenCV-vindue** (virker på macOS, hvor Tk ofte viser hvid skærm). Eksperimentel Tk-menu:

```bash
GAMCONSOLE_UI=tk ./run_gamConsole.sh
```

**Luk:** Klik på den røde **LUK**-bjælke nederst i vinduet, eller tast **Q** / **Esc** (vinduet skal have fokus). Hænger det: `./scripts/stop_gamconsole.sh` eller **Ctrl+C** i terminalen.

## Webcam-games (CvGames)

Projektet er C++11 med OpenCV, OpenGL og GLUT. README i upstream nævner `bob`, men executable bruger **`bub`** som mode (som i `src/main.cpp`).

| Menu | CvGames mode |
|------|----------------|
| Husbygning | `3d` |
| Bubble shot | `bub` |
| Debug view | `debug` |

Installér og byg:

```bash
./scripts/setup_webcam_games.sh
```

Manuelt (macOS):

```bash
brew install pkg-config opencv freeglut
./scripts/setup_webcam_games.sh
```

**OpenCV-version:** Upstream [webcam-games](https://github.com/aurelijusb/webcam-games) bruger legacy C-API (`opencv/cv.h`, `opencv/highgui.h`). Homebrew OpenCV 4/5 indeholder ikke disse headers, så `CvGames` bygger typisk **ikke** på moderne macOS uden en større port. gamConsole-menuen og Python-webcam virker stadig; `./scripts/build_cvgames.sh` er tilpasset pkg-config (`opencv5`) og `.cpp`-only build.

Automatiske tests (uden GUI):

```bash
source .venv/bin/activate
python tests/smoke_test.py -v
```

Test uden menu:

```bash
./vendor/webcam-games/CvGames 3d --camera=0
```

## Struktur

- `gamConsole/` — Tk-menu, webcam-preview, launcher
- `vendor/webcam-games/` — clone (oprettes af setup-script)
- `vendor/tilt-of-time/` — clone af Tilt of Time
- `vendor/handora/` — clone af Handora
- `scripts/setup_webcam_games.sh` — clone + `make`
- `scripts/setup_tilt_of_time.sh` — clone + MediaPipe
- `scripts/setup_handora.sh` — clone af Handora

## Tilt of Time

[Tilt of Time](https://github.com/ebinrajiiit/Tilt_of_Time) er et MIT-licenseret webcam-spil fra [emnet webcam-game](https://github.com/topics/webcam-game). Vip hovedet for at fange faldende ikoner. Det kører i **eget OpenCV-vindue**, fordi upstream lukker alle OpenCV-vinduer når spillet slutter.

```bash
./scripts/setup_tilt_of_time.sh
```

Setup installerer MediaPipe. Den pakke trækker `opencv-contrib-python` 5 med ind, som erstatter `cv2`. Scriptet fjerner den igen og lægger `opencv-python` 4.x tilbage. MediaPipe virker med OpenCV 4.14. Uden MediaPipe falder spillet selv tilbage til OpenCV-ansigtsgenkendelse.

Spillet åbner altid kamera 0 i upstream. Menuens kamera-index sendes med, når det ikke er 0.

## Nostalgia AR (wiisports)

[Nostalgia AR](https://github.com/devgaddebuilds/wiisports) er browser-arkaden fra [emnet webcam-game](https://github.com/topics/webcam-game): bordtennis og frugtskæring styret med hænderne. Repoet har ingen LICENSE-fil. Basketball i samme repo er ikke med i deres lobby, så menuen starter den ikke.

```bash
./scripts/setup_wiisports.sh
```

Menuen starter tre Vite-servere og åbner lobbyen i et WebKit-vindue i gamConsole på `http://127.0.0.1:5180/`. Adressen sættes direkte i vinduet. Deres PLAY-knapper går til bordtennis på port 5173 og frugt på port 5174. Uden de lokale adresser peger lobbyen på `/pingpong/` og `/fruit/`, som giver 404. Lobbyen starter uden Supabase. Sign In og Leaderboard virker først, når `VITE_SUPABASE_URL` og `VITE_SUPABASE_ANON_KEY` står i `vendor/wiisports/nostalgia-ar/.env.local`. **Hjem** lukker browseren og stopper serverne. Siden beder selv om kamera via `getUserMedia` uden `deviceId`, og gamConsole godkender den forespørgsel. macOS kan stadig vise sin egen kamera-dialog første gang.

## Handora

[Handora](https://github.com/BOSS294/Handora) er en browser-side med hånd, ansigt og krop via MediaPipe. Laser og Hit-The-Ball åbnes med **Open Games** inde i siden. Repoet har ingen LICENSE-fil. `alert-beep.mp3` og `alarm-siren.mp3` står i HTML, men ligger ikke i repoet, så de to lyde giver 404.

```bash
./scripts/setup_handora.sh
```

Menuen starter `python -m http.server` på `http://127.0.0.1:5190/modules/hand_gesture/index.html` og åbner siden i gamConsole-browseren. **Hjem** lukker browseren og stopper serveren. Siden beder selv om kamera, når man trykker **Start Camera**. MediaPipe-modellerne hentes fra jsDelivr, så internet skal være tændt. Menuens kamera-index sendes ikke med, fordi siden kalder `getUserMedia` uden `deviceId`.

## Just Dance (plan)

Se [docs/JUST_DANCE.md](docs/JUST_DANCE.md) for vurdering af [olincollege/just-dance](https://github.com/olincollege/just-dance).

```bash
./scripts/setup_just_dance.sh   # fuldt spil (tungt)
```

## Licens

Menu-kode: MIT. Webcam-games og Tilt of Time: MIT (upstream).
