"""
File: niryo_tictactoe/tools/record_corrections.py
Description: Teach the robot where each cell REALLY is (fixes systematic error and backlash).

For every cell the arm moves to the nominal position; you nudge it by hand
(learning mode) until the cup is centred over the hole, press Enter, and the
difference to the nominal pose is stored in corrections.json.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import argparse
import json

from common.robot_control import PICK_Z, ArmController, connect_robot
from v1_fixed_layout import config_fixed as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=cfg.ROBOT_IP)
    ap.add_argument("--out", default="corrections.json")
    args = ap.parse_args()

    robot = connect_robot(args.ip)
    arm = ArmController(robot)
    corrections = {}
    for i in range(9):
        x, y = cfg.cell_xy(i)
        arm.robot.arm.set_learning_mode(False)
        arm.robot.arm.move_pose(arm._pose(x, y, PICK_Z))
        arm.robot.arm.set_learning_mode(True)
        input(f"p{i + 1}: centre the cup over the hole, then press Enter ... ")
        pose = robot.arm.get_pose()
        corrections[f"p{i + 1}"] = [round(pose.x - x, 4), round(pose.y - y, 4), round(pose.z - PICK_Z, 4)]
    robot.arm.set_learning_mode(False)
    with open(args.out, "w") as fh:
        json.dump(corrections, fh, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
