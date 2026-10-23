# Just Dance i gamConsole — vurdering

Kilde:

- [olincollege/just-dance](https://github.com/olincollege/just-dance) — **selvstændigt** Just Dance-lignende spil (webcam + TensorFlow pose/score).

## Kan vi bruge det?

| Repo | Rolle i gamConsole | Realistisk? |
|------|-------------------|-------------|
| **olincollege/just-dance** | Menu-punkt der kloner repo og kører `just_dance_gui.py` / `run_game()` | Ja, men **tungt** (TensorFlow, sang/video-filer, ~16 GB RAM anbefalet) og **Tk-GUI** (samme macOS-problemer som vores Tk-menu) |

## olincollege/just-dance — detaljer

- Fuld flow: vælg sang → `run_game(song=...)` med reference-video + webcam + scoring ([projektbeskrivelse](https://github.com/olincollege/just-dance)).
- Afhængigheder: OpenCV, TensorFlow, playsound, mutagen; store assets under `songs/`, `songs_audio/`, `model/`.
- **Bedste match** hvis målet er “Just Dance-lignende spil i konsollen” — integreres som **vendor-clone + subprocess** (samme mønster som webcam-games).
- Udfordringer: vægt, Tk-start-GUI, kamera samtidig med gamConsole-menu (frigiv kamera før start).

## Anbefalet næste skridt (implementering)

1. **Fase A — launcher only:** menu-punkt + `scripts/setup_just_dance.sh`.
2. **Fase B — olincollege:** start `just_dance_main.run_game` direkte med valgt sang (OpenCV), undgå Tk GUI.

Se `gamConsole/games_registry.py` for menupunkter (når issues er oprettet på board, tilføjes issue-numre i kode).
