#10
"""Mars-rover i spilmenuen, uden at åbne kamera eller spilvindue."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamConsole.games_registry import games_for_menu  # noqa: E402
from gamConsole.house_build import SNAP_TRIGGER_ERR  # noqa: E402
from gamConsole.mars_rover import MarsPad, MarsRoverBuildScene, ROVER_PLAN  # noqa: E402
from gamConsole.menu_opencv import CANVAS_H, QUIT_BAR_H, _compose_menu  # noqa: E402


def _hand_at(x: float, y: float) -> list:
    return [(x, y)] * 21


def _redder_than_blue(pixel: np.ndarray) -> bool:
    return int(pixel[2]) > int(pixel[0]) and int(pixel[2]) > int(pixel[1])


class MarsRoverMenuTest(unittest.TestCase):
    def test_registry_button_is_builtin(self) -> None:
        entry = next(game for game in games_for_menu() if game.id == "mars-rover")
        self.assertEqual(entry.launcher, "builtin")
        self.assertEqual(entry.mode_arg, "mars_rover")
        self.assertEqual(entry.title, "Mars-rover")

    def test_selected_card_stays_above_quit_bar(self) -> None:
        games = games_for_menu()
        selected = next(index for index, game in enumerate(games) if game.id == "mars-rover")
        webcam = np.zeros((480, 640, 3), dtype=np.uint8)
        _canvas, layout = _compose_menu(webcam, games, selected, None, False, 0)
        card = next(box for box in layout.card_boxes if box.game_index == selected)
        self.assertLess(card.y2, CANVAS_H - QUIT_BAR_H)
        self.assertIn(selected, {box.game_index for box in layout.card_boxes})


class MarsRoverSceneTest(unittest.TestCase):
    def test_background_is_mars_not_grass(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (0, 255, 0)
        MarsPad().paint(frame)
        self.assertTrue(_redder_than_blue(frame[8, 8]))
        self.assertTrue(_redder_than_blue(frame[460, 12]))

    def test_first_piece_is_drawn_on_the_pad(self) -> None:
        plain = np.zeros((480, 640, 3), dtype=np.uint8)
        built = plain.copy()
        MarsPad().paint(plain)
        MarsRoverBuildScene().draw(built)
        self.assertTrue(np.any(plain != built))

    def test_plan_is_rover_parts(self) -> None:
        kinds = [step.kind for step in ROVER_PLAN]
        self.assertEqual(kinds[0], "chassis")
        self.assertEqual(kinds.count("wheel"), 6)
        self.assertLess(kinds.index("chassis"), kinds.index("wheel"))
        self.assertLess(kinds.index("wheel"), kinds.index("mast"))
        self.assertLess(kinds.index("mast"), kinds.index("camera"))
        self.assertIn("arm", kinds)
        self.assertIn("antenna", kinds)
        self.assertIn("hazcam", kinds)
        self.assertIn("drill", kinds)
        self.assertLess(kinds.index("arm"), kinds.index("drill"))
        self.assertEqual(kinds[-1], "power")

    def test_first_part_starts_away_from_the_mark(self) -> None:
        scene = MarsRoverBuildScene()
        self.assertGreater(scene.alignment_error(), SNAP_TRIGGER_ERR)

    def test_snap_advances_to_the_next_part(self) -> None:
        scene = MarsRoverBuildScene()
        step = scene.steps[0]
        scene.beam_u = step.target_u
        scene.beam_v = step.target_v
        scene._update_snap()
        scene._update_snap()
        self.assertEqual(scene.step_index, 1)
        self.assertEqual(len(scene.placed), 1)
        self.assertEqual(scene.placed[0].step.kind, "chassis")

    def test_preview_places_every_part(self) -> None:
        scene = MarsRoverBuildScene()
        scene.preview = True
        scene.place_all_for_preview()
        self.assertTrue(scene.complete)
        self.assertEqual(len(scene.placed), len(ROVER_PLAN))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        scene.draw(frame)
        tires = (frame[:, :, 0] < 50) & (frame[:, :, 1] < 50) & (frame[:, :, 2] < 60)
        self.assertGreater(int(tires.sum()), 30)
        self.assertTrue(_redder_than_blue(frame[8, 8]))

    def test_done_text_stays_in_the_bottom_banner(self) -> None:
        scene = MarsRoverBuildScene()
        scene.place_all_for_preview()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        scene.steer((480, 640, 3), (320, 220), holding=False, landmarks=_hand_at(0.50, 0.46))
        scene.draw(frame)
        banner = frame[470, 8]
        self.assertLess(int(banner.mean()), 50)
        self.assertGreater(int(frame[400, 300].mean()), int(banner.mean()))

    def test_hand_keeps_following_after_the_rover_is_finished(self) -> None:
        scene = MarsRoverBuildScene()
        scene.place_all_for_preview()
        self.assertTrue(scene.complete)
        first = np.zeros((480, 640, 3), dtype=np.uint8)
        second = np.zeros((480, 640, 3), dtype=np.uint8)
        scene.steer((480, 640, 3), (80, 80), holding=False, landmarks=_hand_at(0.15, 0.20))
        scene.draw(first)
        scene.steer((480, 640, 3), (520, 360), holding=False, landmarks=_hand_at(0.82, 0.75))
        scene.draw(second)
        self.assertFalse(np.array_equal(first, second))
        self.assertTrue(scene.complete)

    def test_hand_follows_while_a_part_is_locking_in(self) -> None:
        scene = MarsRoverBuildScene()
        scene.beam_u = 0.42
        scene.beam_v = 0.51
        scene.reward_frames = 10
        scene.steer((480, 640, 3), (600, 40), holding=True, landmarks=_hand_at(0.9, 0.1))
        self.assertEqual(scene._grip_landmarks[0], (0.9, 0.1))
        self.assertAlmostEqual(scene.beam_u, 0.42)
        self.assertAlmostEqual(scene.beam_v, 0.51)

    def test_open_hand_does_not_move_the_part(self) -> None:
        scene = MarsRoverBuildScene()
        scene.beam_u = 0.42
        scene.beam_v = 0.51
        scene.steer((480, 640, 3), (8, 8), holding=False)
        self.assertAlmostEqual(scene.beam_u, 0.42)
        self.assertAlmostEqual(scene.beam_v, 0.51)

    def test_thumbnail_shows_mars(self) -> None:
        from gamConsole.game_thumbnails import clear_thumbnail_cache, get_game_thumbnail

        clear_thumbnail_cache()
        thumb = get_game_thumbnail("mars-rover", 96)
        self.assertEqual(thumb.shape[:2], (96, 96))
        self.assertGreater(float(thumb[:, :, 2].mean()), float(thumb[:, :, 0].mean()))


if __name__ == "__main__":
    unittest.main()
