"""
File: niryo_tictactoe/v1_fixed_layout/vision_fixed.py
Description: Variant 1 vision: look at 9 + 2 fixed pixel positions and name their colour.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import cv2

from common.colors import CHIP_COLORS, DRAW_BGR, classify_disc, preprocess


@dataclass
class Reading:
    cells: list                  # 9 x ('red'|'green'|'blue'|'none')
    token: Optional[str]         # 'A', 'B' or None (nobody holds the red token)


class FixedVision:
    def __init__(self, cell_px: dict, token_px: dict, radius: int = 12):
        self.cell_px, self.token_px, self.radius = cell_px, token_px, radius

    def read_once(self, frame) -> Reading:
        hsv = preprocess(frame)
        cells = []
        for i in range(9):
            color, _ = classify_disc(hsv, self.cell_px[f"p{i + 1}"], self.radius,
                                     colors=("green", "blue", "red"))
            cells.append(color)
        token = None
        for side, px in self.token_px.items():
            color, _ = classify_disc(hsv, px, self.radius, colors=("red",))
            if color == "red":
                token = side
        return Reading(cells, token)

    def red_in_roi(self, frame, center, radius: int, min_fraction: float = 0.3) -> bool:
        color, _ = classify_disc(preprocess(frame), center, radius, colors=("red",), min_fraction=min_fraction)
        return color == "red"

    def read_stable(self, camera, frames: int = 7, agreement: float = 0.7) -> Optional[Reading]:
        """Majority vote over several frames; None if the camera view is still changing
        (an arm or a hand in front of the board)."""
        votes = [self.read_once(camera.read()) for _ in range(frames)]
        need = agreement * frames
        cells = []
        for i in range(9):
            value, count = Counter(v.cells[i] for v in votes).most_common(1)[0]
            if count < need:
                return None
            cells.append(value)
        value, count = Counter(v.token for v in votes).most_common(1)[0]
        return Reading(cells, value) if count >= need else None

    def draw(self, frame, reading: Reading):
        out = frame.copy()
        for i in range(9):
            name, px = f"p{i + 1}", self.cell_px[f"p{i + 1}"]
            cv2.circle(out, px, self.radius, DRAW_BGR[reading.cells[i]], 2)
            cv2.putText(out, name, (px[0] - 12, px[1] - self.radius - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        for side, px in self.token_px.items():
            cv2.rectangle(out, (px[0] - 14, px[1] - 14), (px[0] + 14, px[1] + 14),
                          (0, 0, 255) if reading.token == side else (120, 120, 120), 2)
        return out
