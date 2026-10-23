#1
"""Leder set bagfra — synkron med pose (original tegning, ikke officiel licens-figur)."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from gamConsole.pose_body_overlay import (
    BodyPose,
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
)

# Ryder-inspirerede farver (BGR)
HAIR = (32, 58, 95)
SKIN = (190, 200, 225)
VEST_RED = (40, 40, 215)
VEST_WHITE = (248, 248, 248)
VEST_CYAN = (255, 190, 45)
PANTS = (120, 55, 25)
SHOE = (245, 245, 250)
SHOE_ACCENT = (40, 90, 220)
BADGE = (220, 220, 230)


def draw_leader_back(
    canvas: np.ndarray,
    feet_x: int,
    feet_y: int,
    *,
    scale: float = 1.0,
) -> None:
    """Fallback centreret på vejen når pose mangler."""
    s = max(0.65, scale)
    hip = (float(feet_x), float(feet_y - 70 * s))
    sh = (hip[0], hip[1] - 55 * s)
    ls = (sh[0] - 38 * s, sh[1])
    rs = (sh[0] + 38 * s, sh[1])
    la = (ls[0] - 8 * s, feet_y - 8 * s)
    ra = (rs[0] + 8 * s, feet_y - 8 * s)
    lk = (ls[0], hip[1] + 45 * s)
    rk = (rs[0], hip[1] + 45 * s)
    pose = _synthetic_pose(canvas.shape[1], canvas.shape[0], ls, rs, la, ra, lk, rk, hip)
    _draw_leader_from_pose(canvas, pose, blend=1.0)


def draw_leader_synced(canvas: np.ndarray, pose: BodyPose, *, blend: float = 1.0) -> None:
    """Tegn leder oven på samme led som spillerens silhuet."""
    _draw_leader_from_pose(canvas, pose, blend=blend)


def _synthetic_pose(w: int, h: int, ls, rs, la, ra, lk, rk, hip) -> BodyPose:
    pts: list = [(-1, -1)] * 33
    def setp(i, xy):
        pts[i] = (int(xy[0]), int(xy[1]))

    setp(L_SHOULDER, ls)
    setp(R_SHOULDER, rs)
    setp(L_HIP, (hip[0] - 18, hip[1]))
    setp(R_HIP, (hip[0] + 18, hip[1]))
    setp(L_ELBOW, ((ls[0] + la[0]) / 2, (ls[1] + la[1]) / 2))
    setp(R_ELBOW, ((rs[0] + ra[0]) / 2, (rs[1] + ra[1]) / 2))
    setp(L_WRIST, la)
    setp(R_WRIST, ra)
    setp(L_KNEE, lk)
    setp(R_KNEE, rk)
    setp(L_ANKLE, (la[0], la[1] + 6))
    setp(R_ANKLE, (ra[0], ra[1] + 6))
    return BodyPose(w, h, tuple(pts))


def _draw_leader_from_pose(canvas: np.ndarray, pose: BodyPose, *, blend: float) -> None:
    overlay = canvas.copy()
    torso = pose.torso_height()
    thick_arm = max(5, int(torso * 0.11))
    thick_leg = max(6, int(torso * 0.13))

    ls, rs = pose.get(L_SHOULDER), pose.get(R_SHOULDER)
    lh, rh = pose.get(L_HIP), pose.get(R_HIP)
    if ls is None or rs is None or lh is None or rh is None:
        return

    sh_mid = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    hip_mid = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
    shoulder_w = max(20, int(np.hypot(rs[0] - ls[0], rs[1] - ls[1]) * 1.05))

    _draw_head_back(overlay, sh_mid, shoulder_w, torso)
    _draw_torso_vest(overlay, ls, rs, lh, rh, shoulder_w, torso)
    _draw_limb_chain(overlay, pose, L_SHOULDER, L_ELBOW, L_WRIST, VEST_WHITE, thick_arm, SKIN)
    _draw_limb_chain(overlay, pose, R_SHOULDER, R_ELBOW, R_WRIST, VEST_WHITE, thick_arm, SKIN)
    _draw_limb_chain(overlay, pose, L_HIP, L_KNEE, L_ANKLE, PANTS, thick_leg, SHOE)
    _draw_limb_chain(overlay, pose, R_HIP, R_KNEE, R_ANKLE, PANTS, thick_leg, SHOE)

    cv2.addWeighted(overlay, blend, canvas, 1.0 - blend, 0, dst=canvas)


def _draw_head_back(overlay: np.ndarray, sh_mid: Tuple[int, int], shoulder_w: int, torso: float) -> None:
    cx, sy = sh_mid
    head_r = max(14, int(shoulder_w * 0.38))
    head_cy = int(sy - torso * 0.32)
    cv2.ellipse(overlay, (cx, head_cy), (head_r, int(head_r * 1.08)), 0, 0, 360, HAIR, -1, cv2.LINE_AA)
    cv2.ellipse(
        overlay,
        (cx, head_cy + int(head_r * 0.15)),
        (int(head_r * 0.82), int(head_r * 0.75)),
        0,
        0,
        360,
        SKIN,
        -1,
        cv2.LINE_AA,
    )
    for tilt, dx in ((-35, -head_r), (-10, -head_r // 2), (10, head_r // 3), (28, head_r)):
        tip_x = cx + int(dx * 0.55)
        base_y = head_cy - int(head_r * 0.2)
        tip_y = head_cy - int(head_r * 1.05)
        cv2.line(overlay, (cx + dx // 3, base_y), (tip_x, tip_y), HAIR, max(3, head_r // 7), cv2.LINE_AA)


def _draw_torso_vest(
    overlay: np.ndarray,
    ls: Tuple[int, int],
    rs: Tuple[int, int],
    lh: Tuple[int, int],
    rh: Tuple[int, int],
    shoulder_w: int,
    torso: float,
) -> None:
    pad = int(shoulder_w * 0.08)
    body = np.array([ls, rs, (rh[0] + pad, rh[1]), (lh[0] - pad, lh[1])], dtype=np.int32)
    cv2.fillConvexPoly(overlay, body, VEST_RED, cv2.LINE_AA)
    y_top = min(ls[1], rs[1])
    y_bot = max(lh[1], rh[1])
    white_h = max(8, int(torso * 0.38))
    cv2.rectangle(overlay, (ls[0], y_top - 2), (rs[0], y_top + white_h), VEST_WHITE, -1, cv2.LINE_AA)
    cyan_h = max(6, int(torso * 0.14))
    cv2.rectangle(overlay, (lh[0] - pad, y_bot - cyan_h), (rh[0] + pad, y_bot + 2), VEST_CYAN, -1, cv2.LINE_AA)
    badge_x = (ls[0] + rs[0]) // 2
    badge_y = y_top + white_h + int(torso * 0.22)
    cv2.circle(overlay, (badge_x, badge_y), max(5, int(torso * 0.07)), BADGE, -1, cv2.LINE_AA)
    cv2.circle(overlay, (badge_x, badge_y), max(3, int(torso * 0.04)), (30, 50, 200), -1, cv2.LINE_AA)


def _draw_limb_chain(
    overlay: np.ndarray,
    pose: BodyPose,
    i0: int,
    i1: int,
    i2: int,
    color_mid: Tuple[int, int, int],
    thick: int,
    color_end: Tuple[int, int, int],
) -> None:
    p0, p1, p2 = pose.get(i0), pose.get(i1), pose.get(i2)
    if p0 is None or p1 is None:
        return
    cv2.line(overlay, p0, p1, color_mid, thick, cv2.LINE_AA)
    if p2 is not None:
        cv2.line(overlay, p1, p2, color_mid, max(4, thick - 1), cv2.LINE_AA)
        if color_end == SHOE:
            cv2.ellipse(overlay, p2, (max(8, thick + 2), max(5, thick)), 0, 0, 360, color_end, -1, cv2.LINE_AA)
            cv2.ellipse(
                overlay,
                (p2[0], p2[1] + 2),
                (max(6, thick), max(3, thick // 2)),
                0,
                0,
                360,
                SHOE_ACCENT,
                -1,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(overlay, p2, max(5, thick // 2 + 2), color_end, -1, cv2.LINE_AA)
