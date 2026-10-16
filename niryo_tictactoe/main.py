"""
File: niryo_tictactoe/main.py
Description: MAIN ENTRY POINT for students. One command, many jobs. Everything below runs from THIS folder.

  FIRST STEPS (no robot needed)
    python main.py check                 check Python + libraries (add --ip <robot> to test the network)
    python main.py selftest              run the 28 automatic tests (about 25 s)
    python main.py demo                  play a full simulated game, saves frames + GIF in demo_output/
    python main.py train-path            train the learned-path model (saves models/path_model.json)
    python main.py segment --image f.png classical per-chip segmentation of one picture

  WITH THE REAL ROBOT(S)   (robot default IP 169.254.200.200; give each robot its own --ip)
    python main.py click-cells  [--ip IP]   variant 1 only: click the 9 cells + 2 token spots -> pixel table
    python main.py calibrate    [--ip IP]   variants 2/3: one-minute hand-eye calibration -> hand_eye.json
    python main.py corrections  [--ip IP]   optional: per-cell backlash correction table -> corrections.json
    python main.py play --variant 1|2|3 --me green --opp blue --side A --start [--ip IP]
        Robot 1:  ... --me green --opp blue --side A --start
        Robot 2:  ... --me blue  --opp green --side B          (no --start; start it in a second terminal)

  python main.py guide                 print this guide      (full documentation with pictures: README.md)
Everything after the sub-command is passed to that program, so `python main.py play --help` shows all options.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import runpy
import sys
import unittest

COMMANDS = {
    "check": "check_setup",
    "demo": "demo_simulation",
    "train-path": "tools.train_path_model",
    "segment": "segmentation.classical_segmentation",
    "click-cells": "tools.click_cells",
    "calibrate": "tools.calibrate_hand_eye",
    "corrections": "tools.record_corrections",
}
VARIANTS = {"1": "v1_fixed_layout.play_fixed",
            "2": "v2_two_circle_calibration.play_calibrated",
            "3": "v3_robust_background_subtraction.play_robust"}


def run(module: str, argv: list) -> None:
    sys.argv = [module] + argv
    runpy.run_module(module, run_name="__main__")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "guide", "help"):
        print(__doc__)
        return
    command, rest = sys.argv[1], sys.argv[2:]
    if command == "selftest":
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_all")
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    if command == "play":
        if "--variant" not in rest:
            sys.exit("play needs --variant 1|2|3   (1 fixed layout, 2 two-circle calibration, 3 robust to light)")
        i = rest.index("--variant")
        variant = rest[i + 1]
        if variant not in VARIANTS:
            sys.exit(f"unknown variant '{variant}' (use 1, 2 or 3)")
        run(VARIANTS[variant], rest[:i] + rest[i + 2:])
        return
    if command in COMMANDS:
        run(COMMANDS[command], rest)
        return
    sys.exit(f"unknown command '{command}'.  Try:  python main.py guide")


if __name__ == "__main__":
    main()
