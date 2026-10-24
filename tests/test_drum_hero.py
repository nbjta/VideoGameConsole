#13
"""Drum Hero i spilmenuen, uden at åbne kamera eller spilvindue."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from gamConsole.drum_hero import (  # noqa: E402
    BEAT,
    CRASH,
    DRUM_COLORS,
    DRUM_COUNT,
    HIGHWAY_END,
    HIHAT,
    HIT_EARLY,
    HIT_LATE,
    LEAD,
    PATTERN,
    SNARE,
    TOM,
    DrumChart,
    DrumHeroGame,
    DrumKit,
    Drummer,
    HitChart,
    StageBackdrop,
    StageFrame,
)
from gamConsole.drum_hero_session import DrumHeroSession  # noqa: E402
from gamConsole.drum_hero_setup import EASY, HARD, MEDIUM, SONGS  # noqa: E402
from gamConsole.drum_track import SONG_ARTIST, SONG_PATH, SONG_TITLE  # noqa: E402
from gamConsole.flying_users_ar import WebcamARGame  # noqa: E402
from gamConsole.games_registry import games_for_menu  # noqa: E402
from gamConsole.menu_opencv import CANVAS_H, QUIT_BAR_H, _compose_menu  # noqa: E402
from gamConsole.pose_body_overlay import (  # noqa: E402
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    BodyPose,
)
from gamConsole.round_score import (  # noqa: E402
    ROUND_SECONDS,
    HighScoreList,
    RoundScore,
    clock_text,
)


def _ears(left: tuple, right: tuple) -> BodyPose:
    points = [(-1, -1)] * 33
    points[7] = left
    points[8] = right
    return BodyPose(640, 360, tuple(points))


def _arms(elbow_y: int, wrist_y: int, wrist_x_out: int) -> BodyPose:
    points = [(-1, -1)] * 33
    points[7] = (220, 80)
    points[8] = (300, 80)
    points[L_SHOULDER] = (180, 150)
    points[R_SHOULDER] = (320, 150)
    points[L_ELBOW] = (180 - wrist_x_out // 2, elbow_y)
    points[R_ELBOW] = (320 + wrist_x_out // 2, elbow_y)
    points[L_WRIST] = (180 - wrist_x_out, wrist_y)
    points[R_WRIST] = (320 + wrist_x_out, wrist_y)
    points[L_HIP] = (210, 240)
    points[R_HIP] = (290, 240)
    points[L_KNEE] = (210, 300)
    points[R_KNEE] = (300, 300)
    points[L_ANKLE] = (210, 350)
    points[R_ANKLE] = (300, 350)
    return BodyPose(640, 360, tuple(points))


def _side() -> BodyPose:
    #13 Slag ud til siden. Venstre stik rammer hi-hat, højre rammer crash.
    return _arms(150, 160, 150)


def _down() -> BodyPose:
    #13 Slag ned. Venstre stik rammer lilletrommen, højre rammer tammen.
    return _arms(230, 300, 40)


def _shorts(image: np.ndarray) -> np.ndarray:
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (blue > 140) & (blue > red + 30) & (green < 140)


def _yellow(image: np.ndarray) -> np.ndarray:
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (red > 200) & (green > 180) & (blue < 80)


class DrumHeroMenuTest(unittest.TestCase):
    def test_registry_button_is_builtin(self) -> None:
        entry = next(game for game in games_for_menu() if game.id == "drum-hero")
        self.assertEqual(entry.launcher, "builtin")
        self.assertEqual(entry.mode_arg, "drum_hero")
        self.assertEqual(entry.title, "Drum Hero")

    def test_selected_card_stays_above_quit_bar(self) -> None:
        games = games_for_menu()
        selected = next(index for index, game in enumerate(games) if game.id == "drum-hero")
        webcam = np.zeros((480, 640, 3), dtype=np.uint8)
        _canvas, layout = _compose_menu(webcam, games, selected, None, False, 0)
        card = next(box for box in layout.card_boxes if box.game_index == selected)
        self.assertLess(card.y2, CANVAS_H - QUIT_BAR_H)
        self.assertIn(selected, {box.game_index for box in layout.card_boxes})

    def test_session_factory_builds_drum_hero(self) -> None:
        from gamConsole.game_sessions import create_session

        entry = next(game for game in games_for_menu() if game.id == "drum-hero")
        with patch("gamConsole.drum_hero_session.cv2.VideoCapture") as cap:
            cap.return_value.isOpened.return_value = False
            session, message = create_session(entry, 0)
        self.assertIsInstance(session, DrumHeroSession)
        self.assertIn("Drum Hero", message)
        assert session is not None
        session.stop()


class StageBackdropTest(unittest.TestCase):
    def test_frame_is_the_stage(self) -> None:
        frame = StageBackdrop().render(640, 360)
        self.assertEqual(frame.shape, (360, 640, 3))
        #13 Loftet er mørkt. Gulvet er lyst.
        top = int(frame[8, 320].mean())
        floor = int(frame[int(360 * 0.82), 320].mean())
        self.assertLess(top, floor - 40)

    def test_thumbnail_shows_the_drummer_and_the_hihat(self) -> None:
        from gamConsole.game_thumbnails import clear_thumbnail_cache, get_game_thumbnail

        clear_thumbnail_cache()
        thumb = get_game_thumbnail("drum-hero", 96)
        self.assertEqual(thumb.shape[:2], (96, 96))
        self.assertGreater(int(_shorts(thumb).sum()), 3)
        self.assertGreater(int(_yellow(thumb).sum()), 3)


class DrummerMoveTest(unittest.TestCase):
    def test_level_head_stays_centered(self) -> None:
        drummer = Drummer()
        box = drummer.box(_ears((200, 80), (280, 80)), 640, 360)
        body_w = box[2] - box[0]
        self.assertEqual(box[0], 640 // 2 - body_w // 2)

    def test_head_tilt_does_not_move_the_drummer(self) -> None:
        pose = _ears((220, 78), (300, 88))
        level = _ears((220, 80), (300, 80))
        drummer = Drummer()
        self.assertEqual(drummer.box(pose, 640, 360)[0], drummer.box(level, 640, 360)[0])

    def test_full_tilt_stays_centered_on_the_stage(self) -> None:
        drummer = Drummer()
        stage = StageFrame(960, 540)
        right = drummer.box(_ears((200, 40), (280, 200)), 960, 540)
        left = drummer.box(_ears((200, 200), (280, 40)), 960, 540)
        self.assertEqual(left[0], right[0])
        self.assertGreaterEqual(right[0], stage.left)
        self.assertLessEqual(right[2], stage.right)

    def test_avatar_is_cut_into_joints(self) -> None:
        drummer = Drummer()
        for name in (
            "head",
            "torso",
            "upper_left",
            "lower_left",
            "hand_left",
            "upper_right",
            "lower_right",
            "hand_right",
            "thigh_left",
            "shin_left",
            "thigh_right",
            "shin_right",
        ):
            self.assertGreater(drummer.avatar.pixels[name], 100)
        #22 Ledcirklerne er lemmets bredde, ikke en klat hen over kroppen.
        for cap in drummer.avatar.caps:
            self.assertGreater(cap.radius, 8)
            self.assertLess(cap.radius, 40)

    #22 Hvilen sidder med knæene ude og hænderne over skuldrene.
    def test_rest_pose_sits_with_the_arms_up(self) -> None:
        from gamConsole.jointed_figure import limb_tip, posed_layout

        avatar = Drummer().avatar
        _angles, anchors = posed_layout(avatar.parts, None, 0.0, 0.0, 1.0)
        self.assertLess(anchors["shin_left"][0], anchors["thigh_left"][0])
        self.assertGreater(anchors["shin_right"][0], anchors["thigh_right"][0])
        left_foot = limb_tip(avatar.parts["shin_left"], anchors["shin_left"], 1.0, 0.0)
        self.assertGreater(left_foot[1], anchors["shin_left"][1])
        left_hand = limb_tip(avatar.parts["hand_left"], anchors["hand_left"], 1.0, 0.0)
        self.assertLess(left_hand[1], anchors["upper_left"][1])

    #22 Billedet har ingen stikker. Hænderne er slagpunktet, og et slag ned sænker dem.
    def test_a_downward_strike_lowers_the_sticks(self) -> None:
        drummer = Drummer()
        box = drummer.box(None, 640, 360)
        rest = drummer.sticks(None, 640, 360)
        down = drummer.sticks(_down(), 640, 360)
        self.assertGreater(down[0][1], rest[0][1] + 8)
        self.assertGreater(down[1][1], rest[1][1] + 8)

    def test_figure_stays_when_the_head_tilts(self) -> None:
        drummer = Drummer()
        level = np.zeros((360, 640, 3), np.uint8)
        tilted = level.copy()
        drummer.draw(level, _ears((220, 90), (300, 90)))
        drummer.draw(tilted, _ears((220, 70), (300, 150)))
        self.assertEqual(_mass_x(tilted), _mass_x(level))


class DrumHitTest(unittest.TestCase):
    def test_four_drums_sit_around_the_ready_sticks(self) -> None:
        drummer = Drummer()
        kit = DrumKit()
        kit.place(drummer, None, 640, 360)
        self.assertEqual(len(kit.drums), DRUM_COUNT)
        left, right = drummer.sticks(None, 640, 360)
        hihat, snare, tom, crash = kit.drums
        self.assertLess(hihat.x, left[0])
        self.assertGreater(snare.x, left[0])
        self.assertLess(tom.x, right[0])
        self.assertGreater(crash.x, right[0])
        self.assertLess(hihat.y, snare.y)
        self.assertLess(crash.y, tom.y)
        line = 360 * HIGHWAY_END
        for drum in kit.drums:
            self.assertGreater(drum.x, drum.radius)
            self.assertLess(drum.x, 640 - drum.radius)
            #14 Hele trommen sidder under hero-stregen.
            self.assertGreater(drum.y - drum.radius, line)
            self.assertLess(drum.y, 360 - drum.radius)

    def test_each_drum_has_its_own_color(self) -> None:
        drummer = Drummer()
        kit = DrumKit()
        kit.place(drummer, None, 640, 360)
        frame = np.zeros((360, 640, 3), np.uint8)
        kit.paint(frame, drummer, None, 0.0, hud=False)
        seen = set()
        for drum in kit.drums:
            pixel = frame[int(round(drum.y)), int(round(drum.x))]
            self.assertTrue(_near(pixel, DRUM_COLORS[drum.lane]))
            seen.add(drum.lane)
        self.assertEqual(len(seen), DRUM_COUNT)

    def test_ready_sticks_score_nothing(self) -> None:
        for lane in range(DRUM_COUNT):
            kit = _kit_at(lane, None, 1.0)
            self.assertEqual(kit.score, 0)
            self.assertEqual(len(kit.notes), 1)

    def test_side_strike_hits_hihat_and_crash(self) -> None:
        self.assertEqual(_kit_at(HIHAT, _side(), 1.0).score, 1)
        self.assertEqual(_kit_at(CRASH, _side(), 1.0).score, 1)

    def test_down_strike_hits_snare_and_tom(self) -> None:
        self.assertEqual(_kit_at(SNARE, _down(), 1.0).score, 1)
        self.assertEqual(_kit_at(TOM, _down(), 1.0).score, 1)

    def test_side_strike_misses_the_snare(self) -> None:
        kit = _kit_at(SNARE, _side(), 1.0)
        self.assertEqual(kit.score, 0)
        self.assertEqual(len(kit.notes), 1)

    def test_early_strike_waits_and_then_scores(self) -> None:
        drummer = Drummer()
        kit = DrumKit()
        kit.add_note(HIHAT, 1.0)
        kit.tick(drummer, _side(), 640, 360, 1.0 - HIT_EARLY - 0.05, auto=False)
        self.assertEqual(kit.score, 0)
        self.assertEqual(len(kit.notes), 1)
        kit.tick(drummer, _side(), 640, 360, 1.0, auto=False)
        self.assertEqual(kit.score, 1)
        self.assertEqual(kit.notes, [])

    def test_a_late_strike_scores_nothing(self) -> None:
        kit = _kit_at(HIHAT, _side(), 1.0 + HIT_LATE + 0.05)
        self.assertEqual(kit.score, 0)
        self.assertEqual(kit.notes, [])

    def test_the_same_strike_hits_on_a_larger_stage(self) -> None:
        drummer = Drummer()
        kit = DrumKit()
        kit.add_note(HIHAT, 1.0)
        kit.tick(drummer, _side(), 960, 540, 1.0, auto=False)
        self.assertEqual(kit.score, 1)

    def test_chart_spawns_snare_then_tom(self) -> None:
        chart = DrumChart()
        first = chart.take(0.0)
        self.assertEqual([(note.lane, note.hit_at) for note in first], [(SNARE, LEAD)])
        second = chart.take(BEAT)
        self.assertEqual([(note.lane, note.hit_at) for note in second], [(TOM, LEAD + BEAT)])

    #16
    def test_easy_is_slower_and_hard_uses_both_arms(self) -> None:
        self.assertEqual(MEDIUM.beat, BEAT)
        self.assertEqual(MEDIUM.hit_early, HIT_EARLY)
        self.assertEqual(MEDIUM.hit_late, HIT_LATE)
        self.assertEqual([MEDIUM.lanes(index)[0] for index in range(4)], list(PATTERN))
        self.assertGreater(EASY.beat, MEDIUM.beat)
        self.assertGreater(EASY.hit_early, MEDIUM.hit_early)
        self.assertLess(HARD.beat, MEDIUM.beat)
        self.assertLess(HARD.hit_late, MEDIUM.hit_late)
        easy = DrumChart(EASY)
        self.assertEqual([(note.lane, note.hit_at) for note in easy.take(0.0)], [(SNARE, LEAD)])
        self.assertEqual(easy.take(EASY.beat)[0].hit_at, LEAD + EASY.beat)
        hard = DrumChart(HARD)
        opening = hard.take(0.0)
        self.assertEqual(sorted(note.lane for note in opening), [HIHAT, CRASH])
        self.assertEqual({note.hit_at for note in opening}, {LEAD})
        drummer = Drummer()
        both = DrumKit(HARD)
        for note in opening:
            both.add_note(note.lane, note.hit_at)
        both.tick(drummer, _side(), 640, 360, LEAD, auto=False)
        self.assertEqual(both.score, 2)
        wide = DrumKit(EASY)
        wide.add_note(HIHAT, 1.0)
        wide.tick(drummer, _side(), 640, 360, 1.0 - 0.30, auto=False)
        self.assertEqual(wide.score, 1)
        tight = DrumKit(HARD)
        tight.add_note(HIHAT, 1.0)
        tight.tick(drummer, _side(), 640, 360, 1.0 - 0.30, auto=False)
        self.assertEqual(tight.score, 0)
        tight.tick(drummer, _side(), 640, 360, 1.0, auto=False)
        self.assertEqual(tight.score, 1)

    #24
    def test_each_song_has_its_own_hits_in_seconds(self) -> None:
        self.assertEqual([len(song.hits) for song in SONGS], [125, 71, 124, 130, 43])
        self.assertEqual([(song.hits[0].milliseconds, song.hits[0].lane) for song in SONGS], [
            (824, TOM),
            (2531, TOM),
            (267, TOM),
            (313, CRASH),
            (3355, CRASH),
        ])
        self.assertEqual(len({song.hits[0].milliseconds for song in SONGS}), 5)
        for song in SONGS:
            lanes = {hit.lane for hit in song.hits}
            self.assertTrue(lanes <= {HIHAT, SNARE, TOM, CRASH})
            self.assertGreaterEqual(_same_lane_gap(song.hits, song.seconds), MEDIUM.hit_early + MEDIUM.hit_late)
            easy = HitChart(EASY, song)
            medium = HitChart(MEDIUM, song)
            hard = HitChart(HARD, song)
            self.assertLess(len(easy._hits), len(medium._hits))
            self.assertEqual([hit.seconds for hit in hard._hits], [hit.seconds for hit in medium._hits])
            self.assertGreaterEqual(_same_lane_gap(easy._hits, song.seconds), EASY.hit_early + EASY.hit_late)
        song = SONGS[0]
        chart = HitChart(MEDIUM, song)
        self.assertEqual(chart.take(0.0), [])
        first = chart.take(song.hits[0].seconds)
        self.assertEqual(first[0].lane, TOM)
        self.assertAlmostEqual(first[0].hit_at, LEAD + 0.824, places=3)
        chart.take(song.hits[-1].seconds)
        wrapped = chart.take(song.seconds + song.hits[0].seconds)
        self.assertEqual(wrapped[0].lane, TOM)
        self.assertAlmostEqual(wrapped[0].hit_at, LEAD + song.seconds + 0.824, places=3)

    def test_note_travels_down_its_lane(self) -> None:
        drummer = Drummer()
        kit = DrumKit()
        kit.place(drummer, None, 640, 360)
        kit.add_note(SNARE, LEAD)
        note = kit.notes[0]
        drum = kit.drums[SNARE]
        early = kit.note_xy(note, 0.0)
        later = kit.note_xy(note, LEAD * 0.5)
        arrived = kit.note_xy(note, LEAD)
        self.assertLess(early[1], later[1])
        self.assertLess(later[1], arrived[1])
        #14 På stregen ligger noden over trommen, i trommens bane.
        self.assertAlmostEqual(arrived[0], drum.x, delta=0.6)
        self.assertAlmostEqual(arrived[1], 360 * HIGHWAY_END, delta=0.6)
        self.assertLess(arrived[1], drum.y - drum.radius)
        self.assertLess(abs(early[0] - 320), abs(drum.x - 320))


class DrumHeroGameTest(unittest.TestCase):
    def test_game_is_webcam_ar(self) -> None:
        self.assertIsInstance(DrumHeroGame(camera=0), WebcamARGame)

    def test_on_frame_shows_the_stage_not_the_webcam(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            game = DrumHeroGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            try:
                webcam = np.full((360, 640, 3), (0, 0, 255), dtype=np.uint8)
                game.begin_round()
                out = game.on_frame(webcam, now=10.0)
            finally:
                game.close()
        self.assertEqual(out.shape, (360, 640, 3))
        self.assertEqual(int(np.all(out == webcam, axis=2).sum()), 0)
        self.assertLess(int(out[4, 4].mean()), 90)
        self.assertGreater(int(_shorts(out).sum()), 20)

    def test_webcam_is_mirrored_like_gurli(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            game = DrumHeroGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            seen = []
            game.player.refresh_pose = lambda frame: seen.append(frame.copy())
            try:
                webcam = np.zeros((40, 80, 3), dtype=np.uint8)
                webcam[:, :8] = (0, 0, 255)
                game.begin_round()
                game.on_frame(webcam, now=5.0)
            finally:
                game.close()
        self.assertEqual(len(seen), 1)
        self.assertGreater(float(seen[0][:, -8:, 2].mean()), 200)

    def test_three_minutes_shows_the_score_and_play_starts_over(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            game = DrumHeroGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            try:
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                self.assertIsInstance(game.match, RoundScore)
                game.begin_round()
                playing = game.on_frame(frame, now=1000.0)
                self.assertIsNone(game.match.result)
                self.assertEqual(clock_text(game.match.round.remaining(1000.0)), "3:00")
                self.assertGreater(int(playing[:, -80:].sum()), 0)
                game.player.kit.score = 4
                game.on_frame(frame, now=1000.0 + ROUND_SECONDS)
                self.assertIsNotNone(game.match.result)
                assert game.match.result is not None
                self.assertEqual(game.match.result.score, 4)
                self.assertIn("1 \u2014 4", game.match.result.highscore_lines())
                assert game.match.result.button.rect is not None
                x1, y1, x2, y2 = game.match.result.button.rect
                game.on_click((x1 + x2) // 2, (y1 + y2) // 2)
                self.assertIsNone(game.match.result)
                self.assertEqual(game.player.kit.score, 0)
                self.assertTrue(game.setup.open)
                self.assertFalse(game.track.started)
                #16 Spil igen viser listen. Spil starter sangen forfra.
                #24 Første frame sætter ventetiden. Sangen starter LEAD sekunder senere.
                game.begin_round()
                game.on_frame(frame, now=2000.0)
                self.assertFalse(game.track.started)
                again = game.on_frame(frame, now=2000.0 + LEAD)
                self.assertIsNone(game.match.result)
                self.assertTrue(game.track.started)
                self.assertGreater(int(again.sum()), 0)
            finally:
                game.close()

    #15
    def test_pixabay_song_plays_for_the_round_and_stops_on_the_result(self) -> None:
        self.assertEqual(SONG_TITLE, "Upbeat Instrumental")
        self.assertEqual(SONG_ARTIST, "Sub_Clair")
        self.assertTrue(SONG_PATH.is_file())
        self.assertGreater(SONG_PATH.stat().st_size, 1_000_000)
        with tempfile.TemporaryDirectory() as folder:
            game = DrumHeroGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            try:
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                self.assertFalse(game.track.started)
                game.begin_round()
                game.on_frame(frame, now=1000.0)
                #24 Sangen venter, til noden kan nå stregen på filens sekund.
                self.assertFalse(game.track.started)
                game.on_frame(frame, now=1000.0 + LEAD)
                self.assertTrue(game.track.started)
                self.assertIsNone(game.track._sound)
                game.on_frame(frame, now=1000.0 + ROUND_SECONDS)
                self.assertFalse(game.track.started)
            finally:
                game.close()
                self.assertFalse(game.track.started)

    #16
    def test_vertical_list_chooses_song_and_difficulty(self) -> None:
        for song in SONGS:
            self.assertTrue(song.path.is_file())
            self.assertGreater(song.path.stat().st_size, 1_000_000)
        #17 Upbeat Funk af Kulakovka er med.
        self.assertEqual(SONGS[3].title, "Upbeat Funk")
        self.assertEqual(SONGS[3].artist, "Kulakovka")
        #18 Energetic Action Sport står sidst.
        self.assertEqual(SONGS[-1].title, "Energetic Action Sport")
        self.assertEqual(SONGS[-1].artist, "JonasBlakewood")
        with tempfile.TemporaryDirectory() as folder:
            game = DrumHeroGame(camera=0, scores=HighScoreList(Path(folder) / "scores.json"))
            try:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                listed = game.on_frame(frame, now=10.0)
                self.assertTrue(game.setup.open)
                self.assertIsNone(game.match.round.started_at)
                self.assertFalse(game.track.started)
                songs = [row for row in game.setup.rows if row.kind == "song"]
                levels = [row for row in game.setup.rows if row.kind == "difficulty"]
                self.assertEqual([row.key for row in songs], ["588419", "617071", "587078", "614901", "622901"])
                self.assertEqual([row.key for row in levels], ["let", "mellem", "svaer"])
                self.assertLess(songs[0].rect[3], songs[1].rect[1])
                self.assertLess(songs[1].rect[3], songs[2].rect[1])
                self.assertLess(songs[2].rect[3], songs[3].rect[1])
                self.assertLess(songs[3].rect[3], songs[4].rect[1])
                self.assertLess(levels[0].rect[3], levels[1].rect[1])
                self.assertLess(levels[1].rect[3], levels[2].rect[1])
                assert game.setup.play_rect is not None
                self.assertLess(levels[2].rect[3], game.setup.play_rect[1])
                chosen = listed[levels[1].rect[1] + 4, levels[1].rect[0] + 4]
                other = listed[levels[0].rect[1] + 4, levels[0].rect[0] + 4]
                self.assertFalse(np.array_equal(chosen, other))
                game.on_click(2, 2)
                self.assertTrue(game.setup.open)
                game.on_click(*_center(songs[1].rect))
                self.assertEqual(game.setup.song.title, "Upbeat - Upbeat Music")
                self.assertEqual(game.setup.song.artist, "NastelBom")
                game.on_click(*_center(levels[2].rect))
                self.assertEqual(game.setup.difficulty.title, "Svær")
                game.on_click(*_center(game.setup.play_rect))
                self.assertFalse(game.setup.open)
                game.on_frame(frame, now=11.0)
                #24 Sangen venter LEAD sekunder. Første slag i Upbeat Music ligger 2,531 s inde.
                self.assertFalse(game.track.started)
                self.assertIsInstance(game.player.kit.chart, HitChart)
                game.on_frame(frame, now=11.0 + LEAD)
                self.assertTrue(game.track.started)
                self.assertEqual(game.track.path, SONGS[1].path)
                self.assertEqual(game.player.kit.difficulty.id, "svaer")
                game.on_frame(frame, now=11.0 + 2.531)
                self.assertEqual(game.player.kit.notes[0].lane, TOM)
                self.assertAlmostEqual(game.player.kit.notes[0].hit_at, 11.0 + LEAD + 2.531, places=3)
                self.assertIsNone(game.track._sound)
            finally:
                game.close()


def _same_lane_gap(hits: tuple, duration: float) -> float:
    by_lane: dict = {}
    for hit in hits:
        by_lane.setdefault(hit.lane, []).append(hit.seconds)
    gaps = []
    for times in by_lane.values():
        if len(times) < 2:
            continue
        gaps.extend(times[i + 1] - times[i] for i in range(len(times) - 1))
        gaps.append(times[0] + duration - times[-1])
    return min(gaps)


def _center(rect: tuple) -> tuple:
    x1, y1, x2, y2 = rect
    return (x1 + x2) // 2, (y1 + y2) // 2


def _kit_at(lane: int, pose: BodyPose | None, now: float) -> DrumKit:
    drummer = Drummer()
    kit = DrumKit()
    kit.add_note(lane, 1.0)
    kit.tick(drummer, pose, 640, 360, now, auto=False)
    return kit


def _near(pixel: np.ndarray, color: tuple, tol: int = 12) -> bool:
    return all(abs(int(pixel[channel]) - color[channel]) <= tol for channel in range(3))


def _mass_x(image: np.ndarray) -> float:
    xs = np.where(_shorts(image))[1]
    if len(xs) == 0:
        raise AssertionError("shortsene blev ikke tegnet")
    return float(xs.mean())


if __name__ == "__main__":
    unittest.main()
