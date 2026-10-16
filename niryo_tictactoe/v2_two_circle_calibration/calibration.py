"""
File: niryo_tictactoe/v2_two_circle_calibration/calibration.py
Description: Variant 2 - find the playing paper from just TWO coloured circles.

A pink circle (A) and a green circle (B) are printed in two corners of the
paper.  Two points are exactly enough to recover a 2D similarity transform
(rotation + uniform scale + translation), so the paper may be rotated, shifted
or anywhere in the camera view - the chip trays, grid and turn spots follow.

Everything is done with complex numbers:  p' = m * p + t
    |m| = scale, arg(m) = rotation, t = translation.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from common.colors import color_mask


class CalibrationError(RuntimeError):
    pass


# ------------------------------------------------------------------- similarity
@dataclass(frozen=True)
class Similarity2D:
    m: complex
    t: complex

    @classmethod
    def from_two_points(cls, src_a, src_b, dst_a, dst_b) -> "Similarity2D":
        sa, sb = complex(*src_a), complex(*src_b)
        da, db = complex(*dst_a), complex(*dst_b)
        if abs(sb - sa) < 1e-9:
            raise CalibrationError("the two reference points coincide")
        m = (db - da) / (sb - sa)
        return cls(m, da - m * sa)

    def __call__(self, p):
        z = self.m * complex(*p) + self.t
        return (z.real, z.imag)

    def inverse(self) -> "Similarity2D":
        return Similarity2D(1 / self.m, -self.t / self.m)

    def then(self, other: "Similarity2D") -> "Similarity2D":
        """self first, then other."""
        return Similarity2D(other.m * self.m, other.m * self.t + other.t)

    @property
    def scale(self) -> float:
        return abs(self.m)

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.m.imag, self.m.real))

    def as_affine(self) -> np.ndarray:
        """2x3 matrix for cv2.warpAffine."""
        return np.array([[self.m.real, -self.m.imag, self.t.real],
                         [self.m.imag, self.m.real, self.t.imag]], np.float64)


# ----------------------------------------------------------------------- layout
@dataclass
class BoardLayout:
    """Everything is in millimetres on the PAPER, origin = pink circle A, x right, y down.

    The defaults describe the printed paper of this project; edit them (or load a
    JSON file) when the paper is redesigned.  Only the numbers change, not the code.
    """
    circle_a: tuple = (0.0, 0.0)
    circle_b: tuple = (252.0, 180.0)
    circle_diameter: float = 14.0
    chip_diameter: float = 30.0
    grid_center: tuple = (126.0, 90.0)
    pitch: float = 42.0
    slots: dict = field(default_factory=dict)
    token_spots: dict = field(default_factory=lambda: {"A": (70.0, 185.0), "B": (180.0, -5.0)})

    def __post_init__(self):
        if not self.slots:                       # c1..c5 left edge, c6..c10 right edge
            for i in range(5):
                self.slots[f"c{i + 1}"] = (12.0, 42.0 + 32.0 * i)
                self.slots[f"c{i + 6}"] = (240.0, 10.0 + 32.0 * i)

    def cell(self, index: int):
        row, col = divmod(index, 3)
        return (self.grid_center[0] + (col - 1) * self.pitch,
                self.grid_center[1] + (row - 1) * self.pitch)

    @classmethod
    def load(cls, path: str) -> "BoardLayout":
        with open(path) as fh:
            data = json.load(fh)
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in data.items()})


# ------------------------------------------------------------- circle detection
@dataclass
class Circle:
    center: tuple
    radius: float
    circularity: float


def _candidates(mask, min_area: float, max_area: float, min_circularity: float):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area < min_area or area > max_area or perimeter == 0:
            continue
        circularity = 4 * math.pi * area / perimeter ** 2
        if circularity < min_circularity:
            continue
        m = cv2.moments(c)
        found.append(Circle((m["m10"] / m["m00"], m["m01"] / m["m00"]),
                            math.sqrt(area / math.pi), circularity))
    return found


def find_circle(hsv, color: str, min_area=30, max_area=2500, min_circularity=0.72,
                near=None, max_jump=80):
    """Best circle of one colour.

    Chips of the same colour may be on the paper too; the calibration circle is the
    SMALLEST round blob.  When the previous position `near` is known, the blob
    closest to it wins (tracking if the paper is nudged during the game).
    """
    candidates = _candidates(color_mask(hsv, color), min_area, max_area, min_circularity)
    if not candidates:
        return None
    if near is not None:
        close = [c for c in candidates if math.dist(c.center, near) < max_jump]
        if close:
            return min(close, key=lambda c: math.dist(c.center, near))
    return min(candidates, key=lambda c: c.radius)


def locate_paper(hsv, layout: BoardLayout, previous=None, **kwargs) -> tuple:
    """Detect both circles and return (paper->pixel Similarity2D, circle_a, circle_b)."""
    near_a = previous[1].center if previous else None
    near_b = previous[2].center if previous else None
    a = find_circle(hsv, "pink", near=near_a, **kwargs)
    b = find_circle(hsv, "green", near=near_b, **kwargs)
    if a is None or b is None:
        raise CalibrationError(f"calibration circles not found (pink={a is not None}, green={b is not None})")
    sim = Similarity2D.from_two_points(layout.circle_a, layout.circle_b, a.center, b.center)
    return sim, a, b


# ----------------------------------------------------------------- pixel -> robot
@dataclass
class HandEye:
    """Maps camera pixels (at the fixed observation pose) to robot XY in metres.

    Fitted once from the same two circles: jog the tool onto circle A, then B,
    note the robot XY (tools/calibrate_hand_eye.py).  Valid for a camera that
    looks (almost) straight down - which the Niryo wrist camera does at HOME_POSE.
    """
    px_to_robot: Similarity2D

    @classmethod
    def fit(cls, px_a, px_b, robot_a, robot_b) -> "HandEye":
        return cls(Similarity2D.from_two_points(px_a, px_b, robot_a, robot_b))

    def __call__(self, px):
        return self.px_to_robot(px)

    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump({"m": [self.px_to_robot.m.real, self.px_to_robot.m.imag],
                       "t": [self.px_to_robot.t.real, self.px_to_robot.t.imag]}, fh)

    @classmethod
    def load(cls, path: str) -> "HandEye":
        with open(path) as fh:
            d = json.load(fh)
        return cls(Similarity2D(complex(*d["m"]), complex(*d["t"])))
