"""
File: niryo_tictactoe/v2_two_circle_calibration/play_calibrated.py
Description: Variant 2 - paper anywhere, found again before every move from two circles.

python -m v2_two_circle_calibration.play_calibrated --ip 169.254.200.200         --me green --opp blue --side A --start --hand-eye hand_eye.json
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import argparse

from common.camera import NiryoCamera
from common.game_runner import GameRunner
from common.robot_control import HOME_POSE, ArmController, connect_robot
from v2_two_circle_calibration.calibration import BoardLayout, HandEye
from v2_two_circle_calibration.vision_calibrated import CalibratedVision


class CalibratedBackend:
    def __init__(self, camera, vision: CalibratedVision):
        self.camera, self.vision = camera, vision
        self.obs = None

    def read_board(self):
        obs = self.vision.observe_stable(self.camera)
        if obs is None:
            return None
        self.obs = obs                           # always keep the freshest paper pose
        return obs.cells, obs.token

    def _paper_to_robot(self, paper_xy):
        return self.vision.to_robot(self.obs, paper_xy)

    def cell_xy(self, index):
        return self._paper_to_robot(self.vision.layout.cell(index))

    def slot_xy(self, name):
        color, px = self.obs.slots[name]
        if px is not None and self.vision.hand_eye is not None:
            return self.vision.hand_eye(px)       # where the chip REALLY is
        return self._paper_to_robot(self.vision.layout.slots[name])

    def token_xy(self, side):
        return self._paper_to_robot(self.vision.layout.token_spots[side])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="169.254.200.200")
    ap.add_argument("--me", default="green", choices=["blue", "green"])
    ap.add_argument("--opp", default="blue", choices=["blue", "green"])
    ap.add_argument("--agent", default="minimax", choices=["minimax", "random", "hybrid"])
    ap.add_argument("--side", default="A", choices=["A", "B"])
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--layout", default=None, help="optional layout JSON (mm)")
    ap.add_argument("--hand-eye", default="hand_eye.json")
    ap.add_argument("--corrections", default="corrections.json")
    args = ap.parse_args()

    layout = BoardLayout.load(args.layout) if args.layout else BoardLayout()
    robot = connect_robot(args.ip)
    arm = ArmController.with_corrections_file(robot, args.corrections)
    arm.go_home()
    camera = NiryoCamera(args.ip)
    vision = CalibratedVision(layout, HandEye.load(args.hand_eye))
    backend = CalibratedBackend(camera, vision)
    my_slots = [f"c{i}" for i in (range(1, 6) if args.side == "A" else range(6, 11))]
    runner = GameRunner(arm, backend, args.me, args.opp, args.agent,
                        my_side=args.side, opp_side="B" if args.side == "A" else "A",
                        my_slots=my_slots, i_start=args.start)
    runner.run()
    runner.clear_board()
    arm.go_home()


if __name__ == "__main__":
    main()
