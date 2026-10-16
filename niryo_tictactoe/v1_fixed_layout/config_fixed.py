"""
File: niryo_tictactoe/v1_fixed_layout/config_fixed.py
Description: Variant 1 - fixed layout.

Robots, board and chip trays never move, so every position is a constant.
Re-measure the pixel table with tools/click_cells.py whenever the camera pose
or the board is changed.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

ROBOT_IP = "169.254.200.200"

# Pixel centre of every cell in the 640x480 camera image seen from the home pose.
CELL_PX = {
    "p1": (247, 160), "p2": (317, 151), "p3": (381, 149),
    "p4": (249, 223), "p5": (321, 219), "p6": (388, 218),
    "p7": (249, 293), "p8": (318, 291), "p9": (398, 288),
}
# The red "turn token" sits on spot A or B; the side holding it plays next.
TOKEN_PX = {"A": (472, 151), "B": (481, 265)}
ROI_RADIUS = 12
# Token regions as seen from the folded REST pose (x, y, radius): measure with tools/click_cells.py
REST_TOKEN_PX = {"A": (472, 151, 24), "B": (481, 265, 24)}

# ----------------------------------------------------------------- robot frame (m)
GRID_CENTER = (0.250, 0.000)
PITCH_X, PITCH_Y = 0.0555, 0.0525      # measured from the recorded p1..p9 poses


def cell_xy(index: int):
    """Robot XY of cell 0..8 (p1 is the +x/+y corner, as in the recorded poses)."""
    row, col = divmod(index, 3)
    return (GRID_CENTER[0] + (1 - row) * PITCH_X,
            GRID_CENTER[1] + (1 - col) * PITCH_Y)


# chip trays: c1..c5 on the near side, c6..c10 on the far side
CHIP_SLOT_XY = {f"c{i + 1}": (0.12, 0.10 - 0.05 * i) for i in range(5)}
CHIP_SLOT_XY.update({f"c{i + 6}": (0.39, 0.10 - 0.05 * i) for i in range(5)})
TOKEN_XY = {"A": (0.2967, -0.1043), "B": (0.2169, -0.1021)}
