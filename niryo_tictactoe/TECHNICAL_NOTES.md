# Robot vs Robot Tic-Tac-Toe (Niryo Ned2) - computer vision & tolerance engineering

Two Niryo robots play Tic-Tac-Toe against each other (or against a human) using only
their wrist cameras. Project work at OTH Amberg-Weiden, 2024.

Rebuilt and cleaned up from the original lab notebooks. The hard parts of the project were
**not** the game rules but: robot backlash/low repeatability, changing light, finding the
paper without fixed coordinates, knowing whose turn it is, and doing it all on a laptop CPU.

## Three vision variants (same game loop)

| Variant | Folder | Idea | When to use |
|---|---|---|---|
| 1 | `v1_fixed_layout/` | Everything is at a fixed place: 9 pixel positions + 2 token spots, fixed robot coordinates | Robots, board and trays never move |
| 2 | `v2_two_circle_calibration/` | **Two coloured circles** (pink A, green B) give rotation + scale + shift of the paper (2D similarity, complex numbers). Grid, chip trays and turn spots follow the paper; a one-minute hand-eye calibration maps pixels to robot XY | Paper rotated / shifted / anywhere in view |
| 3 | `v3_robust_background_subtraction/` | Variant 2 + rectified top view, per-channel gain matching on the white paper, **background subtraction in Lab** (chroma 1.0, lightness 0.3) | Changing light, shadows |

Shared code in `common/`: game rules and agents (alpha-beta minimax, random, hybrid), HSV colour
model + lighting normalisation, cameras, robot control with backlash handling, and the game loop.
`segmentation/classical_segmentation.py` gives one mask per chip (HSV + distance-transform
watershed, no neural network); `segmentation/yolo_seg_demo.py` is the heavy YOLOv8-seg comparison.

## Motion and standby
`common/trajectory.py` plans arcs in polar space around the robot base (azimuth theta, radius r):
r and theta are interpolated, z follows a parabola, and the last centimetres are a vertical descent
perpendicular to the chip surface (`ArmController._descend`). The robot that is not playing folds into
`REST_JOINTS`, watches only the red-token region, and moves to the fixed `OBSERVE_POSE` to read the board.
`v1_fixed_layout/recorded_positions.py` + `--taught` drive the robot from hand-taught poses in dim light.

## Image subtraction tracking, depth, learned paths
* `v3_.../diff_tracker.py`: no colour detection - consecutive frames are subtracted (twice per second), the largest
  closed area is the chip, its centroid its position; owners are tracked from the first move of the game.
* `common/depth.py`: Z = f*D/d from the chip diameter, Z2/Z1 = d1/d2, a learned Z = a/d + b model and a clearance monitor.
* `common/path_learning.py`: trial-and-error lift search + ridge regression, 3-D parabolic paths and a lateral detour
  search when an obstacle suddenly appears.

## Turn protocol
A red token lies on spot A or B. It is on the side that must play. After my move I carry the
token to the opponent's spot; the opponent puts it back on mine. A move is accepted only when
the token is on my side **and** exactly one new chip of the opponent's colour appeared
(`diff_boards`) - a single bad camera frame can no longer trigger endless "Invalid move".

## Run
```
pip install -r requirements.txt            # opencv-python numpy pyniryo pyniryo2 (+ ultralytics optional)
python -m unittest tests.test_all -v       # 28 tests, no robot needed

# one-time, variants 2 and 3 (about one minute):
python -m tools.calibrate_hand_eye --ip 169.254.200.200
# optional, all variants: per-cell correction table for backlash
python -m tools.record_corrections --ip 169.254.200.200

python -m v1_fixed_layout.play_fixed --me green --opp blue --agent minimax --side A --start
python -m v2_two_circle_calibration.play_calibrated --me green --opp blue --side A --start
python -m v3_robust_background_subtraction.play_robust --me green --opp blue --side A --start
```
Second robot: same command with `--me blue --opp green --side B` (without `--start`).

## What is tested, and what is not
Tested without hardware (`tests/`): game rules and agents (minimax never loses), similarity
transform, circle detection for any rotation/shift/lighting, all three vision variants on synthetic
frames (rotation, shift, dim light, colour cast, shadow), classical segmentation (touching chips), and a
**complete game** against a simulated human with a simulated camera and a mock robot.
Variants 2's circle detection and the segmentation were also run on the original lab footage.

Not tested on a real robot: the `pyniryo2` calls (`move_pose`, `grasp_with_tool`, `get_pose`, ...)
follow the original notebooks, but the new motion code, the hand-eye calibration tool and the
default paper layout (`BoardLayout` in `v2_.../calibration.py`, in millimetres) must be checked and
adjusted on the real setup. Variant 2 assumes a camera looking (almost) straight down, as the
Ned wrist camera does at `HOME_POSE`; with a strongly oblique phone camera the two-point model is
not accurate (this was visible in the hand-held lab footage).
HSV ranges were measured on the real chips (red H 2-4, green 68, blue 110, pink circle 174).
