"""
File: niryo_tictactoe/v3_robust_background_subtraction/diff_tracker.py
Description: Chip tracking by image subtraction - no colour detection at all.

Twice per second the robot photographs the board.  Consecutive images are subtracted; what remains is
the new or moved chip.  The LARGEST closed region is the chip, its centroid is its position.

* A chip appears  -> region where the new image is darker than the paper.
* A chip vanishes -> region where the new image is brighter.
* Owner (which colour) is NOT measured: every chip is followed from the first move of the game.
  A chip placed right after the robot moved is the robot's; any other new chip belongs to the
  opponent; a chip that is lifted and put down keeps the owner it already had.
* The seat of a chip has about 5 mm of play, so the difference image is noisy and the regions are not
  perfect discs - which is why the largest area and its centre of mass are used, not a circle fit.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class Chip:
    id: int
    center: tuple
    owner: str                                    # 'robot' | 'opponent'


@dataclass
class Event:
    kind: str                                     # 'placed' | 'moved' | 'removed'
    chip: Chip
    area: float = 0.0


@dataclass
class DiffChipTracker:
    min_area: float = 250.0
    threshold: float = 18.0
    same_spot_px: float = 40.0
    chips: dict = field(default_factory=dict)
    _next_id: int = 0

    @staticmethod
    def _gray(frame) -> np.ndarray:
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g.astype(np.float32), (7, 7), 0)

    def regions(self, mask: np.ndarray):
        """Closed regions of a binary mask, largest first: [(area, (cx, cy), contour)]."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in found:
            area = cv2.contourArea(c)
            if area >= self.min_area:
                m = cv2.moments(c)
                out.append((area, (m["m10"] / m["m00"], m["m01"] / m["m00"]), c))
        return sorted(out, key=lambda r: -r[0])

    def difference(self, prev, cur):
        """(appeared_mask, vacated_mask, signed_difference) after removing global lighting change."""
        d = self._gray(prev) - self._gray(cur)           # positive: now darker (a chip arrived)
        d = d - np.median(d)                             # lamp brightness drift cancels out
        appeared = (d > self.threshold).astype(np.uint8) * 255
        vacated = (d < -self.threshold).astype(np.uint8) * 255
        return appeared, vacated, d

    def update(self, prev, cur, mover: str) -> list:
        """Process one image pair; `mover` = who moved a chip in between ('robot' or 'opponent')."""
        appeared, vacated, _ = self.difference(prev, cur)
        events, lifted = [], None
        gone = self.regions(vacated)
        if gone:                                         # largest vanished area -> which known chip?
            _, centre, _ = gone[0]
            near = [c for c in self.chips.values() if math.dist(c.center, centre) < self.same_spot_px]
            if near:
                lifted = min(near, key=lambda c: math.dist(c.center, centre))
                del self.chips[lifted.id]
        new = self.regions(appeared)
        if new:
            area, centre, _ = new[0]                     # the largest closed area is the chip
            if lifted is not None:                       # moved chip: keep its identity and owner
                chip = Chip(lifted.id, centre, lifted.owner)
                kind = "moved"
            else:
                chip = Chip(self._next_id, centre, mover)
                self._next_id += 1
                kind = "placed"
            self.chips[chip.id] = chip
            events.append(Event(kind, chip, area))
        elif lifted is not None:
            events.append(Event("removed", lifted))
        return events

    def distances(self):
        """Pairwise centre-to-centre distances in pixels: [(id_a, id_b, d_px)]."""
        ids = sorted(self.chips)
        return [(a, b, math.dist(self.chips[a].center, self.chips[b].center))
                for i, a in enumerate(ids) for b in ids[i + 1:]]
