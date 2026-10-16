"""
File: niryo_tictactoe/tools/calibrate_hand_eye.py
Description: One-time hand-eye calibration (about one minute).

1. Put the empty paper under the camera, robot at HOME_POSE.
2. The script detects the pink (A) and green (B) circle in the camera image.
3. Learning mode is switched on: move the tool tip by hand onto circle A, press
   Enter; then onto circle B, press Enter.
4. The pixel -> robot mapping is saved to hand_eye.json.

    python -m tools.calibrate_hand_eye --ip 169.254.200.200
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import argparse

from common.camera import NiryoCamera
from common.colors import preprocess
from common.robot_control import connect_robot
from v2_two_circle_calibration.calibration import BoardLayout, HandEye, locate_paper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="169.254.200.200")
    ap.add_argument("--out", default="hand_eye.json")
    args = ap.parse_args()

    robot = connect_robot(args.ip)
    camera = NiryoCamera(args.ip)
    _, a, b = locate_paper(preprocess(camera.read()), BoardLayout())
    print(f"circle A (pink) at pixel {a.center}, circle B (green) at pixel {b.center}")

    robot.arm.set_learning_mode(True)
    touched = []
    for name in ("A", "B"):
        input(f"Move the tool tip onto circle {name}, then press Enter ...")
        pose = robot.arm.get_pose()
        touched.append((pose.x, pose.y))
        print(f"  robot XY for {name}: {touched[-1]}")
    robot.arm.set_learning_mode(False)

    HandEye.fit(a.center, b.center, touched[0], touched[1]).save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
