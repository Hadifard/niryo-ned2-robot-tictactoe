"""
File: niryo_tictactoe/demo_simulation.py
Description: Plays one COMPLETE game without any robot or camera. A simulated camera renders the paper (rotated and
shifted on purpose), the real vision code (variant 2: two-circle calibration) reads the board, the real GameRunner
decides with alpha-beta minimax, and a mock robot "moves" chips in the simulated world while a random simulated human
plays the other side. Every chip placement is saved as an annotated camera frame (demo_output/step_XX.png) plus an
animated GIF, so students can see exactly what the robot sees. Usage:  python main.py demo [--rot 22] [--seed 3]
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import argparse
import os
import random

import cv2

from common.game_runner import GameRunner
from common.robot_control import ArmController, MockRobot
from tests.synth import Scene, SimCamera, SimulatedRobotWorld
from v2_two_circle_calibration.calibration import BoardLayout, HandEye
from v2_two_circle_calibration.play_calibrated import CalibratedBackend
from v2_two_circle_calibration.vision_calibrated import CalibratedVision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot", type=float, default=22, help="paper rotation in degrees (try 0..359)")
    ap.add_argument("--seed", type=int, default=3, help="random seed of the simulated human")
    ap.add_argument("--out", default="demo_output")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    layout, rng = BoardLayout(), random.Random(args.seed)
    scene = Scene(layout, args.rot, (10, -8), seed=11, irregular=True)
    scene.slots = {n: ("green" if int(n[1:]) <= 5 else "blue") for n in layout.slots}   # chips in both trays
    world = SimulatedRobotWorld(scene)
    vision = CalibratedVision(layout)
    a, b = scene.sim(layout.circle_a), scene.sim(layout.circle_b)
    vision.hand_eye = HandEye.fit(a, b, scene.true_hand_eye(a), scene.true_hand_eye(b))   # = tools/calibrate_hand_eye.py

    frames = []

    def snapshot(kind, pose):                       # called by the mock robot at every grasp / release
        if kind != "release":
            return
        frame = scene.render()
        view = vision.draw(frame, CalibratedVision(layout).observe(frame))
        cv2.putText(view, f"step {len(frames) + 1}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(args.out, f"step_{len(frames) + 1:02d}.png"), view)
        frames.append(view)

    def on_event(kind, pose):
        world.on_event(kind, pose)
        snapshot(kind, pose)

    robot = MockRobot(on_event)
    arm = ArmController(robot)

    def human_acts():                               # runs while the robot is "waiting": human moves, passes the token back
        if scene.token == "B":
            free = [i for i, c in enumerate(scene.cells) if c == "none"]
            if free:
                scene.cells[rng.choice(free)] = "blue"
                scene.token = "A"

    runner = GameRunner(arm, CalibratedBackend(SimCamera(scene), vision), "green", "blue", "minimax",
                        my_side="A", opp_side="B", my_slots=[f"c{i}" for i in range(1, 6)], i_start=True,
                        poll=0, sleep=lambda s: human_acts(), log=print, timeout=30)
    winner = runner.run()
    print("\nfinal board (O = robot/green, X = simulated human/blue):\n" + str(runner.game))
    print("result:", {"O": "robot wins", "X": "human wins", None: "draw"}[winner])

    try:
        from PIL import Image
        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((480, 360)) for f in frames]
        imgs[0].save(os.path.join(args.out, "game.gif"), save_all=True, append_images=imgs[1:], duration=700, loop=0)
        print(f"saved {len(frames)} frames and {args.out}/game.gif")
    except ImportError:
        print(f"saved {len(frames)} frames in {args.out}/  (pip install pillow to also get game.gif)")


if __name__ == "__main__":
    main()
