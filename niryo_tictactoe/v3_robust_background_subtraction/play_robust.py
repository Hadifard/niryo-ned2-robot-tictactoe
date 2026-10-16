"""
File: niryo_tictactoe/v3_robust_background_subtraction/play_robust.py
Description: Variant 3 - lighting-robust play (background subtraction on the rectified board).

Start with an EMPTY board so the reference image can be learned:

    python -m v3_robust_background_subtraction.play_robust --ip 169.254.200.200         --me green --opp blue --side A --start --hand-eye hand_eye.json
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import argparse

from common.camera import NiryoCamera
from common.game_runner import GameRunner
from common.robot_control import ArmController, connect_robot
from v2_two_circle_calibration.calibration import BoardLayout, HandEye
from v3_robust_background_subtraction.vision_robust import RobustVision


class RobustBackend:
    def __init__(self, camera, vision: RobustVision, hand_eye: HandEye):
        self.camera, self.vision, self.hand_eye = camera, vision, hand_eye
        self.obs = None

    def read_board(self):
        obs = self.vision.observe_stable(self.camera)
        if obs is None:
            return None
        self.obs = obs
        return obs.cells, obs.token

    def _to_robot(self, paper_xy):
        return self.hand_eye(self.obs.sim(paper_xy))

    def cell_xy(self, index):
        return self._to_robot(self.vision.layout.cell(index))

    def slot_xy(self, name):
        return self._to_robot(self.vision.layout.slots[name])

    def token_xy(self, side):
        return self._to_robot(self.vision.layout.token_spots[side])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="169.254.200.200")
    ap.add_argument("--me", default="green", choices=["blue", "green"])
    ap.add_argument("--opp", default="blue", choices=["blue", "green"])
    ap.add_argument("--agent", default="minimax", choices=["minimax", "random", "hybrid"])
    ap.add_argument("--side", default="A", choices=["A", "B"])
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--layout", default=None)
    ap.add_argument("--hand-eye", default="hand_eye.json")
    ap.add_argument("--corrections", default="corrections.json")
    args = ap.parse_args()

    layout = BoardLayout.load(args.layout) if args.layout else BoardLayout()
    robot = connect_robot(args.ip)
    arm = ArmController.with_corrections_file(robot, args.corrections)
    arm.go_home()
    camera = NiryoCamera(args.ip)

    vision = RobustVision(layout)
    print("learning the empty board ...")
    vision.learn_background([camera.read() for _ in range(15)])

    backend = RobustBackend(camera, vision, HandEye.load(args.hand_eye))
    my_slots = [f"c{i}" for i in (range(1, 6) if args.side == "A" else range(6, 11))]
    runner = GameRunner(arm, backend, args.me, args.opp, args.agent,
                        my_side=args.side, opp_side="B" if args.side == "A" else "A",
                        my_slots=my_slots, i_start=args.start)
    runner.run()
    runner.clear_board()
    arm.go_home()


if __name__ == "__main__":
    main()
