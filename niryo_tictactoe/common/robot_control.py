"""
File: niryo_tictactoe/common/robot_control.py
Description: Pick-and-place with a sloppy robot.

The Ned's repeatability and gear backlash are a few millimetres, bigger than the clearance of a
chip in its hole.  Three tricks keep it reliable:

1. Travel between two points on a parabolic ARC in polar space (common/trajectory.py): the cup
   stays above the chips, no sideways dragging.
2. The last centimetres are always a VERTICAL descent from a hover point, so the cup meets the
   chip surface perpendicularly and its lip cannot catch on a neighbour.
3. A per-cell correction table, taught once with tools/record_corrections.py, absorbs the
   remaining systematic error.

Standby rule for two robots on one board: the robot that is not playing folds into REST_JOINTS,
where the other arm cannot hit it; it only watches the red turn token, then goes to the fixed
OBSERVE pose to look at the board.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import json
import time
from typing import Optional

from common.trajectory import arc_waypoints

ORIENTATION = (0.0, 1.57, 0.0)      # roll, pitch, yaw: tool pointing straight down
PICK_Z = 0.050                      # metres, TCP height when the vacuum cup touches a chip
PLACE_Z = 0.060                     # chips are released slightly above the board
HOVER = 0.030                       # extra height used for travelling
OBSERVE_POSE = [0.124, -0.0015, 0.3252, 2.717, 1.459, 2.655]   # fixed camera observation point
HOME_POSE = OBSERVE_POSE
REST_JOINTS = [0.0, 0.5, -1.25, 0.0, 0.0, 0.0]               # folded "sleep" pose (radians)
ARC_LIFT = 0.05                                               # apex of the parabola above the chord (m)


def connect_robot(ip: str = "169.254.200.200", calibrate: bool = True):
    from pyniryo2 import NiryoRobot
    robot = NiryoRobot(ip)
    if calibrate:
        robot.arm.calibrate_auto()
    robot.tool.update_tool()
    return robot


class ArmController:
    def __init__(self, robot, corrections: Optional[dict] = None, lift: float = ARC_LIFT,
                 arc_points: int = 7, settle: float = 0.0):
        self.robot = robot
        self.corrections = corrections or {}      # {"p5": [dx, dy, dz], ...} in metres
        self.lift, self.arc_points, self.settle = lift, arc_points, settle

    @classmethod
    def with_corrections_file(cls, robot, path: str, **kw):
        try:
            with open(path) as fh:
                corrections = json.load(fh)
        except FileNotFoundError:
            corrections = {}
        return cls(robot, corrections, **kw)

    # -- poses -----------------------------------------------------------------
    def _pose(self, x: float, y: float, z: float, name: Optional[str] = None):
        dx, dy, dz = self.corrections.get(name, (0.0, 0.0, 0.0))
        return [x + dx, y + dy, z + dz, *ORIENTATION]

    def go_observe(self):
        self.robot.arm.move_pose(OBSERVE_POSE)

    go_home = go_observe                           # old name

    def go_rest(self):
        self.robot.arm.move_joints(REST_JOINTS)

    # -- motion primitives -----------------------------------------------------
    def _descend(self, x, y, z, name):
        self.robot.arm.move_pose(self._pose(x, y, z + HOVER, name))     # hover point
        self.robot.arm.move_pose(self._pose(x, y, z, name))             # vertical descent
        if self.settle:
            time.sleep(self.settle)

    def travel(self, src_xy, dst_xy, z_from: float, z_to: float):
        """Parabolic arc in polar space between two hover points."""
        for x, y, z in arc_waypoints(src_xy, dst_xy, z_from, z_to, self.lift, self.arc_points):
            self.robot.arm.move_pose(self._pose(x, y, z))

    def pick(self, xy, name: Optional[str] = None):
        x, y = xy
        self._descend(x, y, PICK_Z, name)
        self.robot.tool.grasp_with_tool()
        self.robot.arm.move_pose(self._pose(x, y, PICK_Z + HOVER, name))   # vertical lift

    def place(self, xy, name: Optional[str] = None):
        x, y = xy
        self._descend(x, y, PLACE_Z, name)
        self.robot.tool.release_with_tool()
        self.robot.arm.move_pose(self._pose(x, y, PLACE_Z + HOVER, name))

    def pick_and_place(self, src_xy, dst_xy, src_name=None, dst_name=None):
        self.pick(src_xy, src_name)
        self.travel(src_xy, dst_xy, PICK_Z + HOVER, PLACE_Z + HOVER)
        self.place(dst_xy, dst_name)


# ------------------------------------------------------------------ test double
class _MockArm:
    def __init__(self, log):
        self.log = log
        self.pose = list(HOME_POSE)

    def calibrate_auto(self):
        self.log.append(("calibrate",))

    def move_joints(self, joints):
        self.log.append(("joints", tuple(joints)))

    def move_pose(self, pose):
        self.pose = list(pose)
        self.log.append(("move", tuple(round(v, 4) for v in pose)))

    def get_pose(self):
        return self.pose


class _MockTool:
    def __init__(self, log, on_event):
        self.log, self.on_event = log, on_event

    def update_tool(self):
        pass

    def grasp_with_tool(self):
        self.log.append(("grasp",))
        self.on_event("grasp")

    def release_with_tool(self):
        self.log.append(("release",))
        self.on_event("release")


class MockRobot:
    """Records every command; `on_event(kind, pose)` lets a simulator react."""

    def __init__(self, on_event=None):
        self.log = []
        self._cb = on_event or (lambda *a: None)
        self.arm = _MockArm(self.log)
        self.tool = _MockTool(self.log, lambda kind: self._cb(kind, self.arm.pose))
