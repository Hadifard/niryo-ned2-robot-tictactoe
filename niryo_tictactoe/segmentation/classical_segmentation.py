"""
File: niryo_tictactoe/segmentation/classical_segmentation.py
Description: Instance segmentation of coloured chips WITHOUT a neural network.

Why: in early 2024 YOLO/SAM-style models were too heavy for the project laptops
(a few frames per second on CPU).  A colour mask + distance-transform watershed
gives one clean mask per chip - even when two chips touch - in a few milliseconds.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from common.colors import CHIP_COLORS, DRAW_BGR, color_mask, preprocess


@dataclass
class Instance:
    color: str
    mask: np.ndarray          # uint8 0/255, full frame size
    center: tuple
    area: float
    bbox: tuple               # x, y, w, h


def segment_color(hsv: np.ndarray, name: str, min_area: int = 150, max_area: float = 1e9,
                  min_circularity: float = 0.0) -> list:
    mask = color_mask(hsv, name)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if cv2.countNonZero(mask) < min_area:
        return []

    # seeds = peaks of the distance transform: one per chip, even for touching chips
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, seeds = cv2.threshold(dist, 0.72 * dist.max(), 255, cv2.THRESH_BINARY)
    seeds = seeds.astype(np.uint8)
    n, markers = cv2.connectedComponents(seeds)
    markers = markers + 1
    unknown = cv2.subtract(mask, seeds)
    markers[unknown == 255] = 0
    labels = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))

    instances = []
    for label in range(2, n + 1):
        inst = np.uint8(labels == label) * 255
        area = cv2.countNonZero(inst)
        if area < min_area or area > max_area:
            continue                                   # skin, robot body, screens ...
        found, _ = cv2.findContours(inst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(found, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0 or 4 * np.pi * area / perimeter ** 2 < min_circularity:
            continue                                   # chips are round, hands are not
        m = cv2.moments(inst, binaryImage=True)
        instances.append(Instance(name, inst, (m["m10"] / m["m00"], m["m01"] / m["m00"]),
                                  float(area), cv2.boundingRect(inst)))
    return instances


def segment_chips(bgr: np.ndarray, colors=CHIP_COLORS, min_area: int = 150, max_area: float = 1e9,
                  min_circularity: float = 0.0) -> list:
    hsv = preprocess(bgr)
    result = []
    for name in colors:
        result += segment_color(hsv, name, min_area, max_area, min_circularity)
    return result


def overlay(bgr: np.ndarray, instances: list, alpha: float = 0.45) -> np.ndarray:
    out = bgr.copy()
    layer = bgr.copy()
    for inst in instances:
        layer[inst.mask > 0] = DRAW_BGR[inst.color]
    out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
    for inst in instances:
        contours, _ = cv2.findContours(inst.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, DRAW_BGR[inst.color], 2)
        x, y, w, h = inst.bbox
        cv2.putText(out, f"{inst.color} {int(inst.area)}px", (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "frame.png"
    image = cv2.imread(path)
    found = segment_chips(image)
    print(f"{len(found)} chips:", [(i.color, tuple(round(v) for v in i.center)) for i in found])
    cv2.imwrite("segmented.png", overlay(image, found))
