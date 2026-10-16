"""
File: niryo_tictactoe/common/trajectory.py
Description: Arc trajectories in polar space (robot base at the origin of the XY plane).

A 6-axis arm is naturally "polar": joint 1 sets the azimuth theta, joints 2-3 set the radial
reach r and the height z.  Interpolating (r, theta) instead of (x, y) gives a smooth sweep, and a
parabolic z-profile lifts the tool over the chips so the suction cup never drags across them.
The last centimetres are NOT part of the arc: the tool descends vertically (perpendicular to the
chip surface) from a hover point, see ArmController._descend().
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import math


def xy_to_polar(x: float, y: float):
    return math.hypot(x, y), math.atan2(y, x)


def polar_to_xy(r: float, theta: float):
    return r * math.cos(theta), r * math.sin(theta)


def arc_waypoints(start_xy, end_xy, z_start: float, z_end: float,
                  lift: float = 0.05, n: int = 7):
    """(x, y, z) waypoints from start to end (start itself excluded, end included).

    r and theta change linearly (shortest sweep) and
    z(s) = z0 + (z1 - z0) s + 4 * lift * s (1 - s): a parabola with its apex `lift` metres
    above the straight chord and its ends at z0 / z1.
    """
    r0, t0 = xy_to_polar(*start_xy)
    r1, t1 = xy_to_polar(*end_xy)
    sweep = (t1 - t0 + math.pi) % (2 * math.pi) - math.pi
    points = []
    for i in range(1, n + 1):
        s = i / n
        x, y = polar_to_xy(r0 + (r1 - r0) * s, t0 + sweep * s)
        points.append((x, y, z_start + (z_end - z_start) * s + 4.0 * lift * s * (1.0 - s)))
    return points
