"""
File: niryo_tictactoe/tools/click_cells.py
Description: Measure the pixel table of variant 1: click the centre of p1..p9, then token spots A and B.

    python -m tools.click_cells --image frame.png      (or --ip to grab a live frame)
Prints a dictionary ready to paste into v1_fixed_layout/config_fixed.py.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import argparse

import cv2

NAMES = [f"p{i}" for i in range(1, 10)] + ["token A", "token B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--ip", default="169.254.200.200")
    args = ap.parse_args()
    if args.image:
        frame = cv2.resize(cv2.imread(args.image), (640, 480))
    else:
        from common.camera import NiryoCamera
        frame = NiryoCamera(args.ip).read()

    points = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(NAMES):
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

    cv2.namedWindow("click")
    cv2.setMouseCallback("click", on_mouse)
    while len(points) < len(NAMES):
        view = frame.copy()
        cv2.putText(view, f"click: {NAMES[len(points)]}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("click", view)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            return
    cv2.destroyAllWindows()
    print("CELL_PX =", {f"p{i + 1}": points[i] for i in range(9)})
    print("TOKEN_PX =", {"A": points[9], "B": points[10]})


if __name__ == "__main__":
    main()
