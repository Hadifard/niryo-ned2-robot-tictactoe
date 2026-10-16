"""
File: niryo_tictactoe/common/colors.py
Description: HSV colour model shared by all vision variants.

OpenCV HSV: H in 0..179, S and V in 0..255.
Measured on the real lab footage: red chips H=2..4, blue H=110, green H=68 and
the pink calibration circle H=174 - so pink takes the upper hue band and red the lower.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import cv2
import numpy as np

HSV_RANGES = {
    "red":   [((0, 110, 60), (12, 255, 255))],     # measured on the real chips: H = 2..4
    "green": [((38, 70, 50), (85, 255, 255))],
    "blue":  [((100, 110, 40), (130, 255, 255))],
    "pink":  [((145, 70, 80), (179, 255, 255))],    # calibration circle A (measured H = 174)
}
CHIP_COLORS = ("red", "green", "blue")
DRAW_BGR = {"red": (0, 0, 255), "green": (0, 220, 0), "blue": (255, 80, 0),
            "pink": (255, 0, 255), "none": (200, 200, 200)}


def color_mask(hsv: np.ndarray, name: str) -> np.ndarray:
    mask = None
    for lo, hi in HSV_RANGES[name]:
        part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    return mask


# ------------------------------------------------------------ lighting normalisation
def gray_world(bgr: np.ndarray) -> np.ndarray:
    """Remove a global colour cast (warm/cold room light) with the gray-world model."""
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1e-3)
    return np.clip(img * gains, 0, 255).astype(np.uint8)


def white_patch(bgr: np.ndarray, percentile: float = 97.0, target: float = 235.0) -> np.ndarray:
    """Scale every channel so the brightest 3 % (the white paper) maps to `target`.

    Makes a dim or colour-cast image look like a well-lit one before thresholding.
    """
    img = bgr.astype(np.float32)
    top = np.percentile(img.reshape(-1, 3), percentile, axis=0)
    return np.clip(img * (target / np.maximum(top, 1.0)), 0, 255).astype(np.uint8)


def equalize_lightness(bgr: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE on the L channel only: evens out brightness, keeps hue/saturation."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def preprocess(bgr: np.ndarray, blur: int = 5, balance: bool = True, clahe: bool = True) -> np.ndarray:
    """BGR frame -> HSV frame, ready for colour classification."""
    out = cv2.GaussianBlur(bgr, (blur, blur), 0) if blur else bgr
    if balance:
        out = white_patch(out)
    if clahe:
        out = equalize_lightness(out)
    return cv2.cvtColor(out, cv2.COLOR_BGR2HSV)


# ----------------------------------------------------------------- disc classifier
def classify_disc(hsv: np.ndarray, center, radius: float, colors=CHIP_COLORS,
                  min_fraction: float = 0.35):
    """Which colour fills a disc? Returns (name, {colour: fraction}).

    A pixel vote inside a circular ROI is far more stable than the mean HSV of a
    square patch: the mean of two hues is a third, wrong hue.
    """
    h, w = hsv.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = max(2, int(round(radius)))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x1 <= x0 or y1 <= y0:
        return "none", {}
    roi = hsv[y0:y1, x0:x1]
    disc = np.zeros(roi.shape[:2], np.uint8)
    cv2.circle(disc, (cx - x0, cy - y0), r, 255, -1)
    total = max(int(cv2.countNonZero(disc)), 1)
    fractions = {c: cv2.countNonZero(cv2.bitwise_and(color_mask(roi, c), disc)) / total
                 for c in colors}
    best = max(fractions, key=fractions.get)
    return (best if fractions[best] >= min_fraction else "none"), fractions
