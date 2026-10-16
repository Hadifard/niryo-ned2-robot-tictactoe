"""
File: niryo_tictactoe/common/depth.py
Description: Metric distance from a known object size (chip diameter) and a learned safety model.

Pinhole camera:  d_px = f_px * D / Z      ->      Z = f_px * D / d_px
Two chips seen at the same time:  Z2 / Z1 = d1 / d2   (the smaller one is farther away).
The wrist camera therefore needs no depth sensor: the chip diameter D is the metric ruler.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import numpy as np

CHIP_DIAMETER_M = 0.030


def distance_from_diameter(d_px: float, f_px: float, diameter: float = CHIP_DIAMETER_M) -> float:
    return f_px * diameter / d_px


def metres_per_pixel(d_px: float, diameter: float = CHIP_DIAMETER_M) -> float:
    return diameter / d_px


def relative_distance(d1_px: float, d2_px: float) -> float:
    """Z2 / Z1 from the two apparent diameters."""
    return d1_px / d2_px


class DepthModel:
    """Learns Z = a / d + b from a few calibration passes, then replaces the measurement.

    During the first passes the robot measures the true height (touching the board / chips);
    afterwards the fitted model predicts distance from the apparent chip size alone.
    """

    def __init__(self):
        self.a = self.b = None
        self.rms = None

    def fit(self, d_px, z_m) -> "DepthModel":
        d, z = np.asarray(d_px, float), np.asarray(z_m, float)
        X = np.stack([1.0 / d, np.ones_like(d)], axis=1)
        (self.a, self.b), *_ = np.linalg.lstsq(X, z, rcond=None)
        self.rms = float(np.sqrt(np.mean((X @ [self.a, self.b] - z) ** 2)))
        return self

    @property
    def ready(self) -> bool:
        return self.a is not None

    def predict(self, d_px: float) -> float:
        return self.a / d_px + self.b


class SafetyMonitor:
    """Keeps the cup away from chips, the board and the other arm."""

    def __init__(self, tool_radius: float = 0.012, margin: float = 0.01):
        self.tool_radius, self.margin = tool_radius, margin

    def clearance(self, tool_xyz, obstacles) -> float:
        """Smallest free distance to cylinders (x, y, radius, top_z); negative = collision."""
        x, y, z = tool_xyz
        worst = np.inf
        for ox, oy, radius, top in obstacles:
            lateral = np.hypot(x - ox, y - oy) - radius - self.tool_radius
            vertical = z - top
            worst = min(worst, max(lateral, vertical))
        return float(worst)

    def is_safe(self, tool_xyz, obstacles) -> bool:
        return self.clearance(tool_xyz, obstacles) >= self.margin
