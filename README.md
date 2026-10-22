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
| 3D rotation | `3d` |
| Bubble shot | `bub` |
| WebCam VJ | `vj` |
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
- `scripts/setup_webcam_games.sh` — clone + `make`

## Just Dance (plan)

Se [docs/JUST_DANCE.md](docs/JUST_DANCE.md) for vurdering af [JDWebcam](https://github.com/comeraperuibe944/JDWebcam) og [olincollege/just-dance](https://github.com/olincollege/just-dance).

```bash
./scripts/setup_just_dance.sh   # fuldt spil (tungt)
./scripts/setup_jdwebcam.sh       # controller til JD 2017 PC
```

## Licens

Menu-kode: MIT. Webcam-games: MIT (upstream).
