"""
File: niryo_tictactoe/check_setup.py
Description: Student setup checker. Verifies the Python version, the required libraries, runs a 2-second vision
and game-logic self-check on a synthetic frame, and (optionally) tests that a Niryo robot answers on the network.
Run it FIRST when something does not work:   python main.py check   or   python check_setup.py --ip 169.254.200.200
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import argparse
import importlib
import socket
import sys


def ok(flag: bool) -> str:
    return "OK     " if flag else "MISSING"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=None, help="robot IP: also test the network connection")
    args = ap.parse_args()

    print(f"Python {sys.version.split()[0]}  ({ok(sys.version_info >= (3, 9))} needs >= 3.9)")
    for name, needed in (("numpy", True), ("cv2", True), ("pyniryo", False), ("pyniryo2", False), ("ultralytics", False)):
        try:
            importlib.import_module(name)
            print(f"  {ok(True)} {name}")
        except ImportError:
            print(f"  {ok(False)} {name}  ({'REQUIRED: pip install opencv-python numpy' if needed else 'only needed with a real robot / YOLO demo'})")

    try:
        from common.game_logic import MinimaxAgent, TicTacToe
        from tests.synth import Scene
        from v2_two_circle_calibration.calibration import BoardLayout
        from v2_two_circle_calibration.vision_calibrated import CalibratedVision
        layout = BoardLayout()
        scene = Scene(layout, 30, (10, 5))
        scene.cells = ["blue"] + ["none"] * 8
        obs = CalibratedVision(layout).observe(scene.render())
        move = MinimaxAgent("O").get_move(TicTacToe(["X"] + [" "] * 8))
        print(f"  {ok(obs.cells[0] == 'blue')} vision self-check (paper rotation found: {obs.angle_deg:.0f} deg, expected 30)")
        print(f"  {ok(move == 4)} game AI self-check (minimax answers a corner opening with the centre)")
    except Exception as exc:                       # noqa: BLE001 - this is a diagnostic tool
        print("  MISSING self-check failed:", exc)

    if args.ip:
        try:
            socket.create_connection((args.ip, 9090), timeout=3).close()
            print(f"  {ok(True)} robot {args.ip} answers on port 9090 (ROS bridge)")
        except OSError as exc:
            print(f"  {ok(False)} robot {args.ip} not reachable: {exc}\n          -> check the Ethernet cable, the PC's IP (same subnet, e.g. 169.254.200.x) and that the robot is on.")


if __name__ == "__main__":
    main()
