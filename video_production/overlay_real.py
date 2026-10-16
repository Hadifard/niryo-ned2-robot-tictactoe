"""
File: video_production/overlay_real.py
Description: Run the project's REAL vision code (variant 2) on the lab footage and draw the result.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
import sys, math
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "niryo_tictactoe"))
import cv2, numpy as np
from common.colors import preprocess, classify_disc, DRAW_BGR
from v2_two_circle_calibration.calibration import Similarity2D, find_circle
from segmentation.classical_segmentation import segment_chips

# Cell centres measured once on a clean frame, stored in "circle units":
# A (pink) = 0, B (green) = 1   ->  exactly the 2-point similarity of calibration.py
A0, B0 = (66.0, 206.0), (306.0, 377.0)
TL, TR, BL, BR = (120, 213), (264, 229), (116, 371), (262, 376)
_to_units = Similarity2D.from_two_points(A0, B0, (0, 0), (1, 0))
LAYOUT_U = {}
names = [["p7", "p4", "p1"], ["p8", "p5", "p2"], ["p9", "p6", "p3"]]
for r in range(3):
    for c in range(3):
        u, v = (c + .5) / 3, (r + .5) / 3
        top = (TL[0] + u * (TR[0] - TL[0]), TL[1] + u * (TR[1] - TL[1]))
        bot = (BL[0] + u * (BR[0] - BL[0]), BL[1] + u * (BR[1] - BL[1]))
        LAYOUT_U[names[r][c]] = _to_units((top[0] + v * (bot[0] - top[0]), top[1] + v * (bot[1] - top[1])))
D0 = math.dist(A0, B0)

def process(frame, prev=None, draw_cells=True):
    """frame: BGR (portrait lab video). Returns (annotated, info dict) """
    h, w = frame.shape[:2]
    sc = 360.0 / w                                   # work at 360 px width
    small = cv2.resize(frame, None, fx=sc, fy=sc)
    hsv = preprocess(small)
    a = find_circle(hsv, "pink", min_area=25, max_area=1500, near=prev[0] if prev else None, max_jump=60)
    b = find_circle(hsv, "green", min_area=25, max_area=1500, near=prev[1] if prev else None, max_jump=60)
    out = frame.copy()
    info = {"found": False}
    if a is None or b is None:
        return out, info, None
    sim = Similarity2D.from_two_points((0, 0), (1, 0), a.center, b.center)   # units -> small px
    dist = math.dist(a.center, b.center)
    rad = 0.062 * dist
    k = 1 / sc
    P = lambda p: (int(p[0] * k), int(p[1] * k))
    for nm, c in (("A", a), ("B", b)):
        cv2.circle(out, P(c.center), int(c.radius * k * 1.6), (255, 255, 255), max(2, int(3 * w / 1080)))
    cv2.line(out, P(a.center), P(b.center), (0, 255, 255), max(2, int(2 * w / 1080)), cv2.LINE_AA)
    found = {}
    for name, u in LAYOUT_U.items():
        p = sim(u)
        if not draw_cells:
            continue
        color, _ = classify_disc(hsv, p, rad, colors=("green", "blue", "red"), min_fraction=0.30)
        found[name] = color
        cv2.circle(out, P(p), int(rad * k), DRAW_BGR[color], max(2, int(3 * w / 1080)), cv2.LINE_AA)
        cv2.putText(out, name, (P(p)[0] - int(14 * k), P(p)[1] + int(5 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * k, (255, 255, 255), max(1, int(1.5 * w / 1080)), cv2.LINE_AA)
    ang = math.degrees(math.atan2(b.center[1] - a.center[1], b.center[0] - a.center[0]))
    info = {"found": True, "angle": ang - math.degrees(math.atan2(B0[1] - A0[1], B0[0] - A0[0])),
            "scale": dist / D0, "cells": found, "a": a.center, "b": b.center}
    return out, info, (a.center, b.center)
