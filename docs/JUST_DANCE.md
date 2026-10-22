# Just Dance i gamConsole — vurdering

Kilder:

- [JDWebcam](https://github.com/comeraperuibe944/JDWebcam) — webcam som **virtuel controller** til officiel **Just Dance 2017 (PC)**.
- [olincollege/just-dance](https://github.com/olincollege/just-dance) — **selvstændigt** Just Dance-lignende spil (webcam + TensorFlow pose/score).

## Kan vi bruge dem?

| Repo | Rolle i gamConsole | Realistisk? |
|------|-------------------|-------------|
| **JDWebcam** | Menu-punkt der starter `main.py` som **sidecar** mens du kører JD 2017 på PC | Ja, som **launcher + setup-script** — ikke som spil *inde* i OpenCV-menuen |
| **olincollege/just-dance** | Menu-punkt der kloner repo og kører `just_dance_gui.py` / `run_game()` | Ja, men **tungt** (TensorFlow, sang/video-filer, ~16 GB RAM anbefalet) og **Tk-GUI** (samme macOS-problemer som vores Tk-menu) |

## JDWebcam — detaljer

- MediaPipe pose → syntetiske accelerometer-data → WebSocket `ws://<PC-IP>:8080/smartphone` (Ubisoft-protokol).
- Kræver: JD 2017 PC, “play with smartphone”-skærm, netværk, `websockets`, `mediapipe`, Python 3.12 anbefales upstream.
- **Ikke** et danse-spil i sig selv — det erstatter telefonen.
- macOS: multi-app kamera er begrænset sammenlignet med Windows-indstillingen i deres README.

## olincollege/just-dance — detaljer

- Fuld flow: vælg sang → `run_game(song=...)` med reference-video + webcam + scoring ([projektbeskrivelse](https://github.com/olincollege/just-dance)).
- Afhængigheder: OpenCV, TensorFlow, playsound, mutagen; store assets under `songs/`, `songs_audio/`, `model/`.
- **Bedste match** hvis målet er “Just Dance-lignende spil i konsollen” — integreres som **vendor-clone + subprocess** (samme mønster som webcam-games).
- Udfordringer: vægt, Tk-start-GUI, kamera samtidig med gamConsole-menu (frigiv kamera før start).

## Anbefalet næste skridt (implementering)

1. **Fase A — launcher only:** menu-punkter + `scripts/setup_just_dance.sh` / `scripts/setup_jdwebcam.sh`.
2. **Fase B — olincollege:** start `just_dance_main.run_game` direkte med valgt sang (OpenCV), undgå Tk GUI.
3. **Fase C — JDWebcam:** valgfrit IP-argument og dokumentation til JD 2017 PC.

Se `gamConsole/games_registry.py` for planlagte menupunkter (når issues er oprettet på board, tilføjes issue-numre i kode).
