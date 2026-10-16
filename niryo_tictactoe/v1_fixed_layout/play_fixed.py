"""
File: niryo_tictactoe/v1_fixed_layout/play_fixed.py
Description: Variant 1 - play against a human or another robot on a fixed layout.

python -m v1_fixed_layout.play_fixed --ip 169.254.200.200 --me green --opp blue         --agent minimax --side A --start
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import argparse

from common.camera import NiryoCamera
from common.game_runner import GameRunner
from common.robot_control import ArmController, connect_robot
from v1_fixed_layout import config_fixed as cfg
from v1_fixed_layout import recorded_positions as taught
from v1_fixed_layout.vision_fixed import FixedVision


class FixedBackend:
    def __init__(self, camera, vision: FixedVision):
        self.camera, self.vision = camera, vision

    def read_board(self):
        reading = self.vision.read_stable(self.camera)
        return None if reading is None else (reading.cells, reading.token)

    def cell_xy(self, index):
        return cfg.cell_xy(index)

    def slot_xy(self, name):
        return cfg.CHIP_SLOT_XY[name]

    def token_xy(self, side):
        return cfg.TOKEN_XY[side]

    def token_at_rest(self, side):
        """Standby: the arm is folded and the camera only watches the red-token region."""
        roi = cfg.REST_TOKEN_PX[side]
        return self.vision.red_in_roi(self.camera.read(), roi[:2], roi[2])


class TaughtBackend(FixedBackend):
    """Dark room: positions come from the poses taught by hand, vision is only used for the board."""

    def cell_xy(self, index):
        return tuple(taught.pickup_coordinates[f"p{index + 1}"][:2])

    def slot_xy(self, name):
        return tuple(taught.pickup_coordinates[name][:2])

    def token_xy(self, side):
        return tuple(taught.pickup_coordinates["i1" if side == "A" else "i2"][:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=cfg.ROBOT_IP)
    ap.add_argument("--me", default="green", choices=["blue", "green"])
    ap.add_argument("--opp", default="blue", choices=["blue", "green"])
    ap.add_argument("--agent", default="minimax", choices=["minimax", "random", "hybrid"])
    ap.add_argument("--side", default="A", choices=["A", "B"], help="token spot of this robot")
    ap.add_argument("--start", action="store_true", help="this robot plays first")
    ap.add_argument("--corrections", default="corrections.json")
    ap.add_argument("--taught", action="store_true", help="use the taught poses (low light)")
    args = ap.parse_args()

    robot = connect_robot(args.ip)
    arm = ArmController.with_corrections_file(robot, args.corrections)
    arm.go_home()
    camera = NiryoCamera(args.ip)
    vision = FixedVision(cfg.CELL_PX, cfg.TOKEN_PX, cfg.ROI_RADIUS)
    my_slots = [f"c{i}" for i in (range(1, 6) if args.side == "A" else range(6, 11))]
    backend = (TaughtBackend if args.taught else FixedBackend)(camera, vision)
    runner = GameRunner(arm, backend, args.me, args.opp, args.agent,
                        my_side=args.side, opp_side="B" if args.side == "A" else "A",
                        my_slots=my_slots, i_start=args.start)
    runner.run()
    runner.clear_board()
    arm.go_home()


if __name__ == "__main__":
    main()
