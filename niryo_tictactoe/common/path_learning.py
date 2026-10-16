"""
File: niryo_tictactoe/common/path_learning.py
Description: Learning minimum-length parabolic paths (2-D and 3-D) with trial and error.

The robot tries lift heights on simulated transfers, keeps the shortest collision-free one and fits a
ridge-regularised polynomial model  lift = f(distance, obstacle height).  Afterwards every transfer
uses the prediction instead of searching.  A suddenly appearing obstacle triggers a lateral detour
search: the arc is bent sideways in the horizontal plane until it clears, with the shortest length.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.depth import SafetyMonitor


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float = 0.02
    top: float = 0.08

    def as_tuple(self):
        return (self.x, self.y, self.radius, self.top)


def path_points(start, end, lift: float, lateral: float = 0.0, n: int = 60) -> np.ndarray:
    """3-D parabola: straight chord in XY (+ sideways bulge), z = chord + 4 * lift * s (1 - s)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    s = np.linspace(0.0, 1.0, n)[:, None]
    chord = start + (end - start) * s
    side = np.array([-(end[1] - start[1]), end[0] - start[0], 0.0])
    side = side / (np.linalg.norm(side) + 1e-12)
    bulge = 4.0 * s * (1.0 - s)
    return chord + bulge * (lateral * side) + (bulge * lift) * np.array([0.0, 0.0, 1.0])


def length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def min_clearance(points, obstacles, monitor: SafetyMonitor) -> float:
    if not obstacles:
        return np.inf
    return min(monitor.clearance(p, [o.as_tuple() for o in obstacles]) for p in points)


def _features(dist, height):
    return np.array([1.0, dist, height, dist * height, dist ** 2, height ** 2])


class PathLearner:
    def __init__(self, monitor: SafetyMonitor | None = None, ridge: float = 1e-6):
        self.monitor, self.ridge, self.w = monitor or SafetyMonitor(), ridge, None
        self.history = []                          # (trial_index, lift, length, collision_free)

    def _random_transfer(self, rng):
        a = np.array([rng.uniform(0.10, 0.14), rng.uniform(-0.10, 0.10), 0.06])
        b = np.array([rng.uniform(0.20, 0.40), rng.uniform(-0.10, 0.10), 0.06])
        mid = (a + b) / 2
        top = rng.uniform(0.07, 0.13)
        return a, b, [Obstacle(mid[0], mid[1], 0.02, top)], top

    def best_lift(self, a, b, obstacles, grid=np.linspace(0.0, 0.14, 29)):
        """Trial and error: shortest collision-free lift among the tried ones."""
        best = None
        for lift in grid:
            pts = path_points(a, b, lift)
            ok = min_clearance(pts, obstacles, self.monitor) >= self.monitor.margin
            if ok and (best is None or length(pts) < best[1]):
                best = (lift, length(pts))
        return best[0] if best else grid[-1]

    def train(self, trials: int = 150, seed: int = 0) -> "PathLearner":
        rng = np.random.default_rng(seed)
        X, y = [], []
        for i in range(trials):
            a, b, obstacles, top = self._random_transfer(rng)
            lift = self.best_lift(a, b, obstacles)
            X.append(_features(np.linalg.norm(b[:2] - a[:2]), top))
            y.append(lift)
            self.history.append((i, lift, length(path_points(a, b, lift)), True))
        X, y = np.array(X), np.array(y)
        self.w = np.linalg.solve(X.T @ X + self.ridge * np.eye(X.shape[1]), X.T @ y)
        return self

    def save(self, path: str) -> None:
        """Store the learned regression weights (JSON) so training is done only once."""
        import json
        json.dump({"features": ["1", "dist", "height", "dist*height", "dist^2", "height^2"],
                   "weights": [float(v) for v in self.w], "ridge": self.ridge}, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path: str) -> "PathLearner":
        import json
        data = json.load(open(path))
        learner = cls(ridge=data["ridge"])
        learner.w = np.array(data["weights"])
        return learner

    def predict_lift(self, dist: float, height: float) -> float:
        return float(max(0.0, _features(dist, height) @ self.w))

    def plan(self, a, b, obstacles, laterals=np.linspace(-0.10, 0.10, 41)):
        """Learned lift first; if blocked (new obstacle) search the shortest lateral detour."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        top = max([o.top for o in obstacles], default=0.05)
        lift = self.predict_lift(float(np.linalg.norm(b[:2] - a[:2])), top)
        pts = path_points(a, b, lift)
        if min_clearance(pts, obstacles, self.monitor) >= self.monitor.margin:
            return pts, dict(lift=lift, lateral=0.0, replanned=False)
        best = None
        for lat in laterals:
            for lf in (lift, lift * 0.5, 0.0):
                cand = path_points(a, b, lf, lat)
                if min_clearance(cand, obstacles, self.monitor) >= self.monitor.margin:
                    if best is None or length(cand) < best[0]:
                        best = (length(cand), cand, lf, lat)
        if best is None:
            return pts, dict(lift=lift, lateral=0.0, replanned=True, failed=True)
        return best[1], dict(lift=best[2], lateral=best[3], replanned=True)
