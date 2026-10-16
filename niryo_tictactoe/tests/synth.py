"""
File: niryo_tictactoe/tests/synth.py
Description: Synthetic camera frames of the playing paper (for tests without a robot).
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from v2_two_circle_calibration.calibration import BoardLayout, HandEye, Similarity2D

BGR = {"red": (30, 55, 205), "green": (60, 175, 60), "blue": (200, 80, 30), "pink": (180, 105, 255)}


class Scene:
    """State of the world + a camera that renders it."""

    def __init__(self, layout: BoardLayout, rot_deg=0.0, shift_px=(0, 0), px_per_mm=1.2,
                 gain=(1.0, 1.0, 1.0), noise=2.0, shadow=0.0, seed=0, vignette=0.15, irregular=False):
        self.layout = layout
        self.irregular = irregular
        self.rot, self.shift, self.s = rot_deg, shift_px, px_per_mm
        self.gain, self.noise, self.shadow, self.vignette = gain, noise, shadow, vignette
        self.rng = np.random.default_rng(seed)
        self.cells = ["none"] * 9
        self.slots = {n: ("green" if int(n[1:]) <= 5 else "blue") for n in layout.slots}
        self.token = "A"
        self.size = (640, 480)
        self.sim = self._make_sim()
        # ground-truth camera-pixel -> robot mapping (a rotated, scaled, shifted plane)
        self.true_hand_eye = HandEye(Similarity2D(complex(0.000644, 0.000173), complex(0.085, -0.21)))

    def _make_sim(self) -> Similarity2D:
        centre_paper = complex(*self.layout.grid_center)
        m = complex(math.cos(math.radians(self.rot)), math.sin(math.radians(self.rot))) * self.s
        centre_px = complex(320 + self.shift[0], 240 + self.shift[1])
        return Similarity2D(m, centre_px - m * centre_paper)

    # ------------------------------------------------------------------ drawing
    def _poly(self, pts_mm):
        return np.array([self.sim(p) for p in pts_mm], np.int32)

    def _chip(self, img, center_mm, color, diameter_mm):
        if self.irregular and diameter_mm >= 25:         # real chips: 5 mm of seat play, wobbly outline
            import zlib
            r_ = np.random.default_rng(zlib.crc32(f"{color}{center_mm}".encode()))
            center_mm = (center_mm[0] + r_.uniform(-2.5, 2.5), center_mm[1] + r_.uniform(-2.5, 2.5))
            c = self.sim(center_mm)
            ang = np.linspace(0, 2 * np.pi, 22, endpoint=False)
            rad = diameter_mm / 2 * self.s * (1 + r_.normal(0, 0.05, 22))
            pts = np.stack([c[0] + rad * np.cos(ang), c[1] + rad * np.sin(ang)], 1).astype(np.int32)
            cv2.fillPoly(img, [pts], BGR[color], cv2.LINE_AA)
            cv2.polylines(img, [pts], True, tuple(int(v * 0.6) for v in BGR[color]), 2, cv2.LINE_AA)
            return
        c = self.sim(center_mm)
        r = int(diameter_mm / 2 * self.s)
        cv2.circle(img, (int(c[0]), int(c[1])), r, BGR[color], -1, cv2.LINE_AA)
        light = tuple(min(255, int(v * 1.25)) for v in BGR[color])
        cv2.circle(img, (int(c[0] - r * 0.25), int(c[1] - r * 0.25)), int(r * 0.45), light, -1, cv2.LINE_AA)
        cv2.circle(img, (int(c[0]), int(c[1])), r, tuple(int(v * 0.6) for v in BGR[color]), 2, cv2.LINE_AA)

    def render(self) -> np.ndarray:
        L = self.layout
        img = np.full((480, 640, 3), (112, 117, 120), np.uint8)
        paper = [(-30, -30), (282, -30), (282, 210), (-30, 210)]
        cv2.fillConvexPoly(img, self._poly(paper), (236, 240, 242), cv2.LINE_AA)
        # grid
        for k in (-1.5, -0.5, 0.5, 1.5):
            a = (L.grid_center[0] + k * L.pitch, L.grid_center[1] - 1.5 * L.pitch)
            b = (L.grid_center[0] + k * L.pitch, L.grid_center[1] + 1.5 * L.pitch)
            cv2.line(img, tuple(map(int, self.sim(a))), tuple(map(int, self.sim(b))), (30, 30, 30), 2, cv2.LINE_AA)
            a = (L.grid_center[0] - 1.5 * L.pitch, L.grid_center[1] + k * L.pitch)
            b = (L.grid_center[0] + 1.5 * L.pitch, L.grid_center[1] + k * L.pitch)
            cv2.line(img, tuple(map(int, self.sim(a))), tuple(map(int, self.sim(b))), (30, 30, 30), 2, cv2.LINE_AA)
        for name, p in L.slots.items():                         # empty slot outlines
            c = self.sim(p)
            cv2.circle(img, (int(c[0]), int(c[1])), int(L.chip_diameter / 2 * self.s), (150, 150, 150), 1, cv2.LINE_AA)
        for p, color in ((L.circle_a, "pink"), (L.circle_b, "green")):
            self._chip(img, p, color, L.circle_diameter)
        for i, color in enumerate(self.cells):
            if color != "none":
                self._chip(img, L.cell(i), color, L.chip_diameter)
        for name, color in self.slots.items():
            if color != "none":
                self._chip(img, L.slots[name], color, L.chip_diameter)
        if self.token:
            self._chip(img, L.token_spots[self.token], "red", L.chip_diameter)
        return self._camera(img)

    def _camera(self, img):
        out = img.astype(np.float32)
        out *= np.array(self.gain, np.float32)                      # brightness / colour cast
        if self.vignette:
            yy, xx = np.mgrid[0:480, 0:640]
            d = np.hypot(xx - 320, yy - 240) / 400
            out *= (1 - self.vignette * d ** 2)[..., None]
        if self.shadow:                                              # soft diagonal shadow band
            yy, xx = np.mgrid[0:480, 0:640]
            band = 1 / (1 + np.exp(-(xx - 440 + 0.3 * yy) / 25.0))
            out *= (1 - self.shadow * band)[..., None]
        out += self.rng.normal(0, self.noise, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------- robot interaction
    def robot_xy(self, paper_xy):
        return self.true_hand_eye(self.sim(paper_xy))

    def targets(self) -> dict:
        t = {f"p{i + 1}": self.robot_xy(self.layout.cell(i)) for i in range(9)}
        t.update({n: self.robot_xy(p) for n, p in self.layout.slots.items()})
        t.update({f"token{s}": self.robot_xy(p) for s, p in self.layout.token_spots.items()})
        return t

    def nearest(self, xy, max_dist=0.012):
        best = min(self.targets().items(), key=lambda kv: math.dist(kv[1], xy))
        return best[0] if math.dist(best[1], xy) < max_dist else None


class SimulatedRobotWorld:
    """Connects a MockRobot to a Scene: grasp removes a chip, release puts it down."""

    def __init__(self, scene: Scene):
        self.scene, self.held = scene, None

    def _get(self, key):
        s = self.scene
        if key.startswith("token"):
            return "red" if s.token == key[5:] else "none"
        if key.startswith("p"):
            return s.cells[int(key[1:]) - 1]
        return s.slots[key]

    def _set(self, key, value):
        s = self.scene
        if key.startswith("token"):
            s.token = key[5:] if value == "red" else (None if s.token == key[5:] else s.token)
        elif key.startswith("p"):
            s.cells[int(key[1:]) - 1] = value
        else:
            s.slots[key] = value

    def on_event(self, kind, pose):
        key = self.scene.nearest(pose[:2])
        if key is None:
            return
        if kind == "grasp" and self._get(key) != "none":
            self.held = self._get(key)
            self._set(key, "none")
        elif kind == "release" and self.held:
            self._set(key, self.held)
            self.held = None


class SimCamera:
    def __init__(self, scene: Scene, on_read=None):
        self.scene, self.on_read = scene, on_read

    def read(self):
        if self.on_read:
            self.on_read()
        return self.scene.render()
