"""
File: niryo_tictactoe/v3_robust_background_subtraction/vision_robust.py
Description: Variant 3 - lighting-robust vision with image background subtraction.

Pipeline (all classical OpenCV, about 50 ms per 640x480 frame measured on a 2-core cloud CPU, no GPU/YOLO):

 1. Find the two circles -> paper pose (variant 2), warp the paper to a canonical
    top view at a FIXED scale, so every cell is always the same patch of pixels.
 2. Learn the EMPTY board once (median of a few frames).
 3. Every frame: match brightness/colour of the white paper to the reference
    (per-channel gain), then subtract the reference in Lab space.
    Chroma (a,b) counts fully, lightness only 30 %, so shadows and dimming barely
    matter but a coloured chip stands out.
 4. Occupancy = foreground fraction inside the cell disc; colour = hue vote of the
    foreground pixels only.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from common.colors import color_mask, white_patch
from v2_two_circle_calibration.calibration import (BoardLayout, CalibrationError, Similarity2D,
                                                   locate_paper)

CHIP_ORDER = ("green", "blue", "red")


@dataclass
class RobustObservation:
    cells: list
    token: Optional[str]
    foreground: np.ndarray          # binary mask in the rectified view (for display)
    rectified: np.ndarray
    sim: Similarity2D               # paper(mm) -> camera pixel


class RobustVision:
    def __init__(self, layout: BoardLayout, px_per_mm: float = 2.0, margin_mm: float = 30.0,
                 chroma_threshold: float = 14.0, lightness_weight: float = 0.3,
                 occupancy: float = 0.45):
        self.layout, self.s = layout, px_per_mm
        self.margin = margin_mm
        xs = [layout.circle_a[0], layout.circle_b[0]] + [p[0] for p in layout.slots.values()]
        ys = [layout.circle_a[1], layout.circle_b[1]] + [p[1] for p in layout.slots.values()]
        self.origin_mm = (min(xs) - margin_mm, min(ys) - margin_mm)
        self.size = (int((max(xs) - min(xs) + 2 * margin_mm) * px_per_mm),
                     int((max(ys) - min(ys) + 2 * margin_mm) * px_per_mm))
        self.thr, self.wl, self.occupancy = chroma_threshold, lightness_weight, occupancy
        self.background: Optional[np.ndarray] = None       # float32 Lab, rectified
        self._previous = None

    # ------------------------------------------------------------------ geometry
    def _paper_to_canvas(self) -> Similarity2D:
        return Similarity2D(complex(self.s, 0),
                            complex(-self.origin_mm[0] * self.s, -self.origin_mm[1] * self.s))

    def _locate(self, frame):
        hsv = cv2.cvtColor(white_patch(cv2.GaussianBlur(frame, (5, 5), 0)), cv2.COLOR_BGR2HSV)
        sim, a, b = locate_paper(hsv, self.layout, self._previous)
        self._previous = (sim, a, b)
        return sim

    def rectify(self, frame, sim: Similarity2D) -> np.ndarray:
        canvas_from_px = sim.inverse().then(self._paper_to_canvas())
        return cv2.warpAffine(frame, canvas_from_px.as_affine(), self.size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    # -------------------------------------------------------------- reference image
    def learn_background(self, frames) -> None:
        stack = []
        for frame in frames:
            sim = self._locate(frame)
            stack.append(self.rectify(frame, sim))
        median = np.median(np.stack(stack), axis=0).astype(np.uint8)
        self.background = cv2.cvtColor(median, cv2.COLOR_BGR2Lab).astype(np.float32)
        self.bg_bgr = median
        # pixels of white paper = bright and unsaturated in the empty reference
        hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
        self.paper_mask = ((hsv[..., 1] < 60) & (hsv[..., 2] > 150)).astype(np.uint8)
        self.paper_mask = cv2.erode(self.paper_mask, np.ones((5, 5), np.uint8))

    # ---------------------------------------------------------------- per frame
    def _gain_match(self, rect: np.ndarray, exclude: Optional[np.ndarray] = None) -> np.ndarray:
        """Scale each channel so the paper looks like it did in the reference."""
        mask = self.paper_mask.astype(bool)
        if exclude is not None:
            mask &= ~exclude
        if mask.sum() < 200:
            return rect
        out = rect.astype(np.float32)
        for c in range(3):
            ref = float(np.median(self.bg_bgr[..., c][mask]))
            cur = float(np.median(rect[..., c][mask]))
            out[..., c] *= ref / max(cur, 1.0)
        return np.clip(out, 0, 255).astype(np.uint8)

    def foreground(self, rect: np.ndarray) -> tuple:
        matched = self._gain_match(rect)
        for _ in range(2):                                   # 2nd pass: no chips
            lab = cv2.cvtColor(matched, cv2.COLOR_BGR2Lab).astype(np.float32)
            d = lab - self.background
            diff = np.sqrt((self.wl * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
            fg = (cv2.GaussianBlur(diff, (5, 5), 0) > self.thr).astype(np.uint8)
            grown = cv2.dilate(fg, np.ones((9, 9), np.uint8)).astype(bool)
            matched = self._gain_match(rect, exclude=grown)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return fg, matched

    def _classify_foreground(self, matched_hsv, fg, center_mm, radius_mm):
        c = self._paper_to_canvas()(center_mm)
        r = int(radius_mm * self.s)
        cx, cy = int(round(c[0])), int(round(c[1]))
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(self.size[0], cx + r + 1), min(self.size[1], cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            return "none"
        disc = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.circle(disc, (cx - x0, cy - y0), r, 1, -1)
        covered = (fg[y0:y1, x0:x1] & disc).sum() / max(disc.sum(), 1)
        if covered < self.occupancy:
            return "none"
        sel = (fg[y0:y1, x0:x1] & disc).astype(np.uint8) * 255
        roi = matched_hsv[y0:y1, x0:x1]
        votes = {name: cv2.countNonZero(cv2.bitwise_and(color_mask(roi, name), sel)) for name in CHIP_ORDER}
        best = max(votes, key=votes.get)
        return best if votes[best] > 0.3 * cv2.countNonZero(sel) else "none"

    def observe(self, frame) -> RobustObservation:
        if self.background is None:
            raise RuntimeError("call learn_background() with frames of the EMPTY board first")
        sim = self._locate(frame)
        rect = self.rectify(frame, sim)
        fg, matched = self.foreground(rect)
        hsv = cv2.cvtColor(cv2.GaussianBlur(matched, (3, 3), 0), cv2.COLOR_BGR2HSV)
        radius = 0.32 * self.layout.pitch
        cells = [self._classify_foreground(hsv, fg, self.layout.cell(i), radius) for i in range(9)]
        token = None
        for side, spot in self.layout.token_spots.items():
            token_color = self._classify_foreground(hsv, fg, spot, 0.3 * self.layout.chip_diameter)
            if token_color == "red":
                token = side
        return RobustObservation(cells, token, fg, matched, sim)

    def observe_stable(self, camera, frames: int = 5, agreement: float = 0.7):
        obs = []
        for _ in range(frames):
            try:
                obs.append(self.observe(camera.read()))
            except CalibrationError:
                pass
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
        return RobustObservation(cells, token, obs[-1].foreground, obs[-1].rectified, obs[-1].sim)
