#8
"""Husbygning: bjælke over betonfundament, uden kamera."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamConsole.hand_pip_overlay import hand_is_clenched  # noqa: E402
from gamConsole.house_build import (  # noqa: E402
    ConcreteFoundation,
    HOUSE_PLAN,
    HouseBuildScene,
    PlacedPiece,
    TimberBeam,
)
from gamConsole.house_paint_assets import brush_sprite  # noqa: E402
from gamConsole.house_tool_cards import ToolCardBar  # noqa: E402


def _open_hand() -> list:
    """21 punkter. Håndled i bunden, fingerspidser langt oppe."""
    points = [(0.50, 0.80)] * 21
    points = list(points)
    points[0] = (0.50, 0.80)
    points[5] = (0.38, 0.58)
    points[9] = (0.50, 0.55)
    points[13] = (0.62, 0.58)
    points[17] = (0.72, 0.62)
    for tip, mcp in ((8, 5), (12, 9), (16, 13), (20, 17)):
        mx, my = points[mcp]
        dx, dy = mx - points[0][0], my - points[0][1]
        points[tip] = (mx + dx * 1.8, my + dy * 1.8)
    points[4] = (0.20, 0.40)
    points[8] = (0.30, 0.12)
    return points


def _fist() -> list:
    """Fingerspidser trukket ind til knoerne."""
    points = _open_hand()
    for tip, mcp in ((4, 5), (8, 5), (12, 9), (16, 13), (20, 17)):
        mx, my = points[mcp]
        points[tip] = (mx + 0.01, my + 0.02)
    return points


class HouseBuildTest(unittest.TestCase):
    def test_beam_is_longer_than_it_is_thick(self) -> None:
        length, height, depth = TimberBeam.half_extents
        self.assertGreater(length, height * 3)
        self.assertGreater(length, depth * 3)

    def test_foundation_covers_the_camera_picture(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (0, 255, 0)
        ConcreteFoundation().paint(frame)
        #8 Hjørnet er blå himmel, siden er græs, og kamera-grønt er dækket.
        corner = frame[8, 8].astype(int)
        self.assertGreater(int(corner[0]), int(corner[2]))
        self.assertLess(int(corner[1]), 180)
        grass = frame[300, 8].astype(int)
        self.assertGreater(int(grass[1]), int(grass[0]))
        self.assertGreater(int(grass[1]), int(grass[2]))
        slab = frame[250, 320].astype(int)
        self.assertLess(int(slab.max()) - int(slab.min()), 35)
        self.assertGreater(int(slab.mean()), 140)

    def test_beam_is_drawn_on_top_of_the_foundation(self) -> None:
        plain = np.zeros((480, 640, 3), dtype=np.uint8)
        built = plain.copy()
        ConcreteFoundation().paint(plain)
        HouseBuildScene().draw(built)
        self.assertTrue(np.any(plain != built))

    def test_steer_holds_still_without_motion(self) -> None:
        scene = HouseBuildScene()
        scene.beam_u = 0.33
        scene.beam_v = 0.44
        scene.steer((480, 640, 3), None)
        self.assertAlmostEqual(scene.beam_u, 0.33)
        self.assertAlmostEqual(scene.beam_v, 0.44)

    def test_steer_follows_the_motion_centroid(self) -> None:
        scene = HouseBuildScene()
        scene.steer((480, 640, 3), (0, 0))
        self.assertLess(scene.beam_u, 0.2)
        self.assertLess(scene.beam_v, 0.2)

    def test_open_hand_stays_visible_without_moving_the_beam(self) -> None:
        scene = HouseBuildScene()
        scene.beam_u = 0.42
        scene.beam_v = 0.51
        scene.steer((480, 640, 3), (8, 8), holding=False, landmarks=_open_hand())
        self.assertAlmostEqual(scene.beam_u, 0.42)
        self.assertAlmostEqual(scene.beam_v, 0.51)
        self.assertIsNotNone(scene._grip_landmarks)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        scene.draw(frame)
        cyan = (frame[:, :, 0] > 200) & (frame[:, :, 1] > 200) & (frame[:, :, 2] < 40)
        self.assertGreater(int(cyan.sum()), 0)

    def test_clench_moves_the_beam(self) -> None:
        scene = HouseBuildScene()
        scene.steer((480, 640, 3), (0, 0), holding=True, landmarks=_fist())
        self.assertLess(scene.beam_u, 0.2)
        self.assertLess(scene.beam_v, 0.2)

    def test_finger_pose_changes_the_hand_on_the_beam(self) -> None:
        open_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fist_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        open_scene = HouseBuildScene()
        fist_scene = HouseBuildScene()
        open_scene.steer((480, 640, 3), (320, 200), holding=True, landmarks=_open_hand())
        fist_scene.steer((480, 640, 3), (320, 200), holding=True, landmarks=_fist())
        open_scene.draw(open_frame)
        fist_scene.draw(fist_frame)
        self.assertFalse(np.array_equal(open_frame, fist_frame))

    def test_squeeze_is_a_fist_or_a_pinch(self) -> None:
        self.assertFalse(hand_is_clenched(_open_hand()))
        self.assertTrue(hand_is_clenched(_fist()))
        pinched = _open_hand()
        pinched[4] = pinched[8]
        self.assertTrue(hand_is_clenched(pinched))

    def test_several_angles_keep_the_frame_shape(self) -> None:
        scene = HouseBuildScene()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for u in (0.2, 0.5, 0.8):
            scene.beam_u = u
            scene.beam_v = 0.5
            scene.draw(frame)
            self.assertEqual(frame.shape, (360, 640, 3))

    def test_snap_advances_to_next_beam(self) -> None:
        scene = HouseBuildScene()
        step = scene.steps[0]
        scene.beam_u = step.target_u
        scene.beam_v = step.target_v
        scene._update_snap()
        while scene.snap_anim > 0:
            scene._update_snap()
        self.assertEqual(scene.step_index, 1)
        self.assertEqual(len(scene.placed), 1)

    def test_plan_posts_then_rim_doors_windows_and_roof(self) -> None:
        kinds = [step.kind for step in HOUSE_PLAN]
        self.assertEqual(kinds[:4], ["post", "post", "post", "post"])
        self.assertEqual(kinds[4:8], ["wall", "wall", "wall", "wall"])
        self.assertEqual(kinds[8:12], ["rim", "rim", "rim", "rim"])
        self.assertEqual(kinds.count("door"), 2)
        self.assertGreaterEqual(kinds.count("window"), 2)
        self.assertEqual(kinds[-1], "roof")
        self.assertLess(kinds.index("post"), kinds.index("wall"))
        self.assertLess(kinds.index("wall"), kinds.index("rim"))
        self.assertLess(kinds.index("rim"), kinds.index("door"))
        self.assertLess(kinds.index("door"), kinds.index("window"))
        self.assertLess(kinds.index("window"), kinds.index("roof"))

    def test_brush_png_loads(self) -> None:
        bgr, mask = brush_sprite()
        self.assertEqual(len(bgr.shape), 3)
        self.assertGreater(int(mask.sum()), 0)

    def test_tool_card_hit_in_top_bar(self) -> None:
        bar = ToolCardBar()
        w = 640
        layouts = bar._layout(w)
        x1, y1, x2, y2, _ = layouts[0]
        self.assertEqual(bar.tool_at((x1 + x2) // 2, (y1 + y2) // 2, w), "hand")
        x1, y1, x2, y2, card = layouts[1]
        self.assertEqual(card.tool_id, "brush_red")
        self.assertEqual(bar.tool_at((x1 + x2) // 2, (y1 + y2) // 2, w), "brush_red")
        self.assertIsNone(bar.tool_at(100, 200, w))

    def test_paint_only_on_placed_material(self) -> None:
        scene = HouseBuildScene()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        top, _ = scene.foundation.paint(frame)
        scene._slab_top = top
        scene.select_tool("brush_red")
        cx, cy = _slab_point_from_top(top, 0.12, 0.14)
        scene.placed.append(PlacedPiece(step=scene.steps[0]))
        scene.paint_at_pixel(cx, cy)
        self.assertEqual(len(scene.paint_dots), 1)
        scene.paint_at_pixel(20, 20)
        self.assertEqual(len(scene.paint_dots), 1)


def _slab_point_from_top(top, u, v):
    from gamConsole.house_build import _slab_point

    return _slab_point(top, u, v)


if __name__ == "__main__":
    unittest.main()
