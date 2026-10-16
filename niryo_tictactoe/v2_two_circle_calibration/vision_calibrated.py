"""
File: niryo_tictactoe/v2_two_circle_calibration/vision_calibrated.py
Description: Variant 2 vision: everything is positioned relative to the two calibration circles.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from common.colors import DRAW_BGR, color_mask, classify_disc, preprocess
from v2_two_circle_calibration.calibration import (BoardLayout, CalibrationError, HandEye,
                                                   Similarity2D, locate_paper)


@dataclass
class Observation:
    cells: list                  # 9 x colour name
    token: Optional[str]         # 'A' / 'B' / None
    slots: dict                  # 'c1'.. -> (colour, (px, py) refined centroid or None)
    sim: Similarity2D            # paper(mm) -> pixel
    angle_deg: float
    px_per_mm: float


class CalibratedVision:
    def __init__(self, layout: BoardLayout, hand_eye: Optional[HandEye] = None):
        self.layout, self.hand_eye = layout, hand_eye
        self._previous = None

    # ------------------------------------------------------------------ one frame
    def observe(self, frame) -> Observation:
        hsv = preprocess(frame)
        sim, a, b = locate_paper(hsv, self.layout, self._previous)
        self._previous = (sim, a, b)
        px_per_mm = sim.scale
        cell_r = 0.38 * self.layout.pitch * px_per_mm
        chip_r = 0.35 * self.layout.chip_diameter * px_per_mm

        cells = []
        for i in range(9):
            color, _ = classify_disc(hsv, sim(self.layout.cell(i)), cell_r,
                                     colors=("green", "blue", "red"))
            cells.append(color)

        token = None
        for side, spot in self.layout.token_spots.items():
            color, _ = classify_disc(hsv, sim(spot), chip_r, colors=("red",))
            if color == "red":
                token = side

        slots = {}
        for name, spot in self.layout.slots.items():
            center = sim(spot)
            color, _ = classify_disc(hsv, center, chip_r, colors=("green", "blue", "red"))
            slots[name] = (color, self._refine(hsv, center, color, chip_r * 1.6) if color != "none" else None)
        return Observation(cells, token, slots, sim, sim.angle_deg, px_per_mm)

    @staticmethod
    def _refine(hsv, center, color, radius):
        """Centroid of the chip's colour blob near the nominal slot: vision-guided pick."""
        h, w = hsv.shape[:2]
        x0, y0 = int(max(0, center[0] - radius)), int(max(0, center[1] - radius))
        x1, y1 = int(min(w, center[0] + radius)), int(min(h, center[1] + radius))
        mask = color_mask(hsv[y0:y1, x0:x1], color)
        m = cv2.moments(mask, binaryImage=True)
        if m["m00"] < 20:
            return None
        return (x0 + m["m10"] / m["m00"], y0 + m["m01"] / m["m00"])

    # --------------------------------------------------------- vote over frames
    def observe_stable(self, camera, frames: int = 5, agreement: float = 0.7) -> Optional[Observation]:
        obs = []
        for _ in range(frames):
            try:
                obs.append(self.observe(camera.read()))
            except CalibrationError:
                pass                                   # arm/hand covers a circle: skip frame
        if len(obs) < agreement * frames:
            return None
        need = agreement * len(obs)
        cells = []
        for i in range(9):
            value, count = Counter(o.cells[i] for o in obs).most_common(1)[0]
            if count < need:
                return None
            cells.append(value)
        token, count = Counter(o.token for o in obs).most_common(1)[0]
        if count < need:
            return None
        last = obs[-1]
        return Observation(cells, token, last.slots, last.sim, last.angle_deg, last.px_per_mm)

    # ------------------------------------------------------------ robot positions
    def to_robot(self, obs: Observation, paper_xy):
        if self.hand_eye is None:
            raise CalibrationError("hand-eye calibration missing (run tools/calibrate_hand_eye.py)")
        return self.hand_eye(obs.sim(paper_xy))

    def draw(self, frame, obs: Observation, label: bool = True):
        out = frame.copy()
        L, sim = self.layout, obs.sim
        for name, p in (("A", L.circle_a), ("B", L.circle_b)):
            x, y = map(int, sim(p))
            cv2.circle(out, (x, y), 9, (255, 255, 255), 2)
            cv2.putText(out, name, (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for i in range(9):
            x, y = map(int, sim(L.cell(i)))
            cv2.circle(out, (x, y), int(0.38 * L.pitch * obs.px_per_mm), DRAW_BGR[obs.cells[i]], 2)
            cv2.putText(out, f"p{i + 1}", (x - 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        for name, (color, _) in obs.slots.items():
            x, y = map(int, sim(L.slots[name]))
            cv2.circle(out, (x, y), int(0.35 * L.chip_diameter * obs.px_per_mm), DRAW_BGR[color], 1)
        for side, spot in L.token_spots.items():
            x, y = map(int, sim(spot))
            cv2.rectangle(out, (x - 12, y - 12), (x + 12, y + 12),
                          (0, 0, 255) if obs.token == side else (120, 120, 120), 2)
        if label:
            cv2.putText(out, f"rot {obs.angle_deg:6.1f} deg  {obs.px_per_mm:4.2f} px/mm", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return out
