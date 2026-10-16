"""
File: niryo_tictactoe/tests/test_all.py
Description: Run with:  python -m unittest tests.test_all -v   (no robot, no pyniryo needed)
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import math
import random
import unittest

import cv2
import numpy as np

from common.colors import classify_disc, preprocess
from common.game_logic import (EMPTY, HybridAgent, MinimaxAgent, RandomAgent, TicTacToe,
                               board_from_colors, diff_boards)
from common.game_runner import GameRunner
from common.robot_control import ArmController, MockRobot, PICK_Z, PLACE_Z, HOVER
from common.trajectory import arc_waypoints, polar_to_xy, xy_to_polar
from common.depth import DepthModel, SafetyMonitor, distance_from_diameter, relative_distance
from common.path_learning import Obstacle, PathLearner, length, min_clearance, path_points
from segmentation.classical_segmentation import segment_chips
from tests.synth import Scene, SimCamera, SimulatedRobotWorld
from v1_fixed_layout.vision_fixed import FixedVision
from v2_two_circle_calibration.calibration import (BoardLayout, HandEye, Similarity2D, locate_paper)
from v2_two_circle_calibration.vision_calibrated import CalibratedVision
from v3_robust_background_subtraction.vision_robust import RobustVision
from v3_robust_background_subtraction.diff_tracker import DiffChipTracker

LAYOUT = BoardLayout()
PATTERN = ["blue", "none", "green", "none", "blue", "red", "green", "none", "blue"]
PATTERN = ["blue", "none", "green", "none", "blue", "none", "green", "none", "blue"]


def play(a, b, first="X"):
    game, turn = TicTacToe(), first
    agents = {"X": a, "O": b}
    while not game.is_over():
        game.make_move(agents[turn].get_move(game), turn)
        turn = "O" if turn == "X" else "X"
    return game.winner()


class GameLogic(unittest.TestCase):
    def test_minimax_never_loses(self):
        rng = random.Random(1)
        for i in range(120):
            first = "X" if i % 2 else "O"
            self.assertNotEqual(play(RandomAgent("X", rng), MinimaxAgent("O"), first), "X")
            self.assertNotEqual(play(MinimaxAgent("X"), RandomAgent("O", rng), first), "O")

    def test_minimax_vs_minimax_is_draw(self):
        self.assertIsNone(play(MinimaxAgent("X"), MinimaxAgent("O")))

    def test_minimax_takes_immediate_win_and_blocks(self):
        g = TicTacToe(["O", "O", " ", "X", "X", " ", " ", " ", " "])
        self.assertEqual(MinimaxAgent("O").get_move(g), 2)
        g = TicTacToe(["X", "X", " ", " ", "O", " ", " ", " ", " "])
        self.assertEqual(MinimaxAgent("O").get_move(g), 2)

    def test_hybrid_switches_to_random(self):
        agent = HybridAgent("O", smart_moves=2, rng=random.Random(3))
        g = TicTacToe()
        agent.get_move(g), agent.get_move(g)
        self.assertEqual(agent.smart_moves_left, 0)

    def test_diff_boards(self):
        prev = [EMPTY] * 9
        now = list(prev); now[4] = "X"
        self.assertEqual(diff_boards(prev, now), ([(4, "X")], []))
        gone = list(now); gone[4] = EMPTY
        self.assertEqual(diff_boards(now, gone)[1], ["p5: chip disappeared"])
        two = list(now); two[0] = "X"
        self.assertEqual(len(diff_boards(now, two)[0]), 1)


class Trajectory(unittest.TestCase):
    def test_polar_roundtrip(self):
        for x, y in ((0.3, 0.05), (0.2, -0.1), (0.12, 0.0)):
            self.assertAlmostEqual(math.dist(polar_to_xy(*xy_to_polar(x, y)), (x, y)), 0, places=9)

    def test_arc_is_parabolic_and_ends_exactly(self):
        pts = arc_waypoints((0.12, 0.10), (0.30, -0.05), 0.08, 0.09, lift=0.05, n=9)
        self.assertEqual(len(pts), 9)
        self.assertAlmostEqual(pts[-1][2], 0.09)
        self.assertAlmostEqual(math.dist(pts[-1][:2], (0.30, -0.05)), 0, places=9)
        zs = [p[2] for p in pts]
        self.assertGreater(max(zs), 0.08 + 0.04)                  # clears the chips
        self.assertTrue(all(z >= 0.08 - 1e-9 for z in zs))        # never dips below the start height

    def test_vertical_final_approach_and_standby(self):
        robot = MockRobot()
        arm = ArmController(robot)
        arm.pick_and_place((0.12, 0.10), (0.25, 0.0), "c1", "p5")
        moves = [m[1] for m in robot.log if m[0] == "move"]
        release = next(i for i, m in enumerate(robot.log) if m[0] == "release")
        before = [m[1] for m in robot.log[:release] if m[0] == "move"]
        hover, down = before[-2], before[-1]
        self.assertEqual(hover[:2], down[:2])                     # purely vertical descent
        self.assertAlmostEqual(hover[2] - down[2], HOVER, places=4)
        self.assertEqual(down[3:], (0.0, 1.57, 0.0))              # tool axis perpendicular to the board
        arm.go_rest()
        self.assertEqual(robot.log[-1][0], "joints")


class DepthAndLearning(unittest.TestCase):
    def test_pinhole_distance_and_ratio(self):
        self.assertAlmostEqual(distance_from_diameter(60, 800), 0.4)          # f=800 px, D=30 mm, d=60 px
        self.assertAlmostEqual(relative_distance(60, 30), 2.0)                # half the size = twice as far

    def test_depth_model_learns_from_few_passes(self):
        d = np.array([120.0, 80.0, 60.0, 48.0])
        z = 800 * 0.030 / d + 0.002
        model = DepthModel().fit(d, z)
        self.assertLess(model.rms, 1e-6)
        self.assertAlmostEqual(model.predict(40.0), 800 * 0.030 / 40.0 + 0.002, places=5)

    def test_path_learner_is_collision_free_and_short(self):
        learner = PathLearner().train(trials=80, seed=1)
        a, b = (0.12, 0.05, 0.06), (0.32, -0.05, 0.06)
        obs = [Obstacle(0.22, 0.0, 0.02, 0.10)]
        pts, info = learner.plan(a, b, obs)
        self.assertGreaterEqual(min_clearance(pts, obs, learner.monitor), learner.monitor.margin - 1e-9)
        naive = path_points(a, b, 0.14)
        self.assertLessEqual(length(pts), length(naive) + 1e-9)

    def test_sudden_obstacle_gets_a_detour(self):
        learner = PathLearner().train(trials=80, seed=2)
        a, b = (0.12, 0.0, 0.06), (0.32, 0.0, 0.06)
        wall = [Obstacle(0.22, 0.0, 0.03, 0.30)]                             # too tall to hop over
        pts, info = learner.plan(a, b, wall)
        self.assertTrue(info["replanned"] and not info.get("failed"))
        self.assertNotEqual(info["lateral"], 0.0)
        self.assertGreaterEqual(min_clearance(pts, wall, learner.monitor), learner.monitor.margin - 1e-9)


class Calibration(unittest.TestCase):
    def test_similarity_roundtrip(self):
        s = Similarity2D.from_two_points((0, 0), (10, 0), (5, 5), (5, 15))
        self.assertAlmostEqual(s.angle_deg, 90, places=6)
        self.assertAlmostEqual(s.scale, 1.0, places=6)
        p = (3.0, 4.0)
        q = s.inverse()(s(p))
        self.assertAlmostEqual(q[0], p[0], places=6)
        self.assertAlmostEqual(q[1], p[1], places=6)

    def _check_pose(self, scene):
        frame = scene.render()
        sim, a, b = locate_paper(preprocess(frame), LAYOUT)
        self.assertLess(math.dist(a.center, scene.sim(LAYOUT.circle_a)), 2.5)
        self.assertLess(math.dist(b.center, scene.sim(LAYOUT.circle_b)), 2.5)
        for i in range(9):
            self.assertLess(math.dist(sim(LAYOUT.cell(i)), scene.sim(LAYOUT.cell(i))), 3.0)

    def test_any_rotation_and_shift(self):
        for rot in (0, 37, 90, 144, 180, 271, 330):
            for shift in ((0, 0), (-40, 25), (35, -30)):
                with self.subTest(rot=rot, shift=shift):
                    self._check_pose(Scene(LAYOUT, rot, shift, seed=rot))

    def test_lighting_changes(self):
        for gain in ((0.55, 0.55, 0.55), (1.0, 1.0, 1.0), (0.8, 0.9, 1.15), (1.2, 1.0, 0.8)):
            with self.subTest(gain=gain):
                self._check_pose(Scene(LAYOUT, 25, (10, 10), gain=gain, shadow=0.0))

    def test_hand_eye_fit_recovers_truth(self):
        scene = Scene(LAYOUT, 33, (12, -8))
        a, b = scene.sim(LAYOUT.circle_a), scene.sim(LAYOUT.circle_b)
        fitted = HandEye.fit(a, b, scene.true_hand_eye(a), scene.true_hand_eye(b))
        for i in range(9):
            px = scene.sim(LAYOUT.cell(i))
            self.assertLess(math.dist(fitted(px), scene.true_hand_eye(px)), 1e-6)


class Variant1Fixed(unittest.TestCase):
    def test_reads_fixed_layout(self):
        scene = Scene(LAYOUT, 0, (0, 0))
        scene.cells = list(PATTERN)
        cell_px = {f"p{i + 1}": tuple(map(int, scene.sim(LAYOUT.cell(i)))) for i in range(9)}
        token_px = {k: tuple(map(int, scene.sim(v))) for k, v in LAYOUT.token_spots.items()}
        vision = FixedVision(cell_px, token_px, radius=int(0.35 * LAYOUT.pitch * scene.s))
        reading = vision.read_stable(SimCamera(scene))
        self.assertEqual(reading.cells, PATTERN)
        self.assertEqual(reading.token, "A")


class Variant2Calibrated(unittest.TestCase):
    def _run(self, rot, shift, gain=(1, 1, 1), shadow=0.0):
        scene = Scene(LAYOUT, rot, shift, gain=gain, shadow=shadow, seed=rot + 7)
        scene.cells = list(PATTERN)
        scene.token = "B"
        obs = CalibratedVision(LAYOUT).observe(scene.render())
        self.assertEqual(obs.cells, PATTERN)
        self.assertEqual(obs.token, "B")
        self.assertLess(abs((obs.angle_deg - rot + 180) % 360 - 180), 1.0)   # wrap-safe
        return obs

    def test_rotated_shifted_board(self):
        for rot, shift in ((0, (0, 0)), (40, (-30, 20)), (135, (25, 25)), (250, (-20, -25)), (330, (0, 30))):
            with self.subTest(rot=rot):
                self._run(rot, shift)

    def test_lighting_variation(self):
        for gain in ((0.6, 0.6, 0.6), (1.1, 1.0, 0.85), (0.85, 0.95, 1.15)):
            with self.subTest(gain=gain):
                self._run(20, (5, 5), gain)

    def test_slots_and_refined_centroids(self):
        scene = Scene(LAYOUT, 15, (0, 0))
        obs = CalibratedVision(LAYOUT).observe(scene.render())
        self.assertEqual(obs.slots["c1"][0], "green")
        self.assertEqual(obs.slots["c6"][0], "blue")
        truth = scene.sim(LAYOUT.slots["c1"])
        self.assertLess(math.dist(obs.slots["c1"][1], truth), 3.0)

    def test_tracks_shifted_paper(self):
        vision = CalibratedVision(LAYOUT)
        for shift in ((0, 0), (15, 5), (30, 10), (45, 14)):
            scene = Scene(LAYOUT, 12, shift)
            scene.cells = list(PATTERN)
            self.assertEqual(vision.observe(scene.render()).cells, PATTERN)


class Variant3Robust(unittest.TestCase):
    def setUp(self):
        empty = Scene(LAYOUT, 18, (8, -6), seed=1)
        empty.cells, empty.slots, empty.token = ["none"] * 9, {n: "none" for n in LAYOUT.slots}, None
        self.vision = RobustVision(LAYOUT)
        self.vision.learn_background([empty.render() for _ in range(5)])

    def _scene(self, **kw):
        scene = Scene(LAYOUT, **kw)
        scene.cells = list(PATTERN)
        scene.slots = {n: "none" for n in LAYOUT.slots}
        scene.token = "B"
        return scene

    def test_same_light(self):
        obs = self.vision.observe(self._scene(rot_deg=18, shift_px=(8, -6), seed=2).render())
        self.assertEqual(obs.cells, PATTERN)
        self.assertEqual(obs.token, "B")

    def test_much_darker_and_colour_cast(self):
        for gain in ((0.5, 0.5, 0.5), (0.7, 0.85, 1.1), (1.15, 1.0, 0.8)):
            with self.subTest(gain=gain):
                obs = self.vision.observe(self._scene(rot_deg=18, shift_px=(8, -6), gain=gain, seed=3).render())
                self.assertEqual(obs.cells, PATTERN)

    def test_shadow_and_paper_moved(self):
        obs = self.vision.observe(self._scene(rot_deg=40, shift_px=(-25, 18), gain=(0.9, 0.9, 0.9),
                                              shadow=0.35, seed=4).render())
        self.assertEqual(obs.cells, PATTERN)


class DiffTracking(unittest.TestCase):
    def _frame(self, cells, gain=1.0, seed=0):
        sc = Scene(LAYOUT, 0, (0, 0), seed=seed, noise=5.0, gain=(gain,) * 3, irregular=True)
        sc.cells = list(cells)
        sc.slots, sc.token = {n: "none" for n in LAYOUT.slots}, None
        return sc, cv2.cvtColor(sc.render(), cv2.COLOR_BGR2GRAY)

    def test_tracks_owner_and_position_without_colour(self):
        tracker = DiffChipTracker()
        cells = ["none"] * 9
        sc, prev = self._frame(cells, 1.0, 1)
        plan = [(4, "robot", "green"), (0, "opponent", "blue"), (8, "robot", "green")]
        for k, (cell, who, colour) in enumerate(plan):
            cells[cell] = colour
            sc, cur = self._frame(cells, 1.0 - 0.06 * k, k + 2)        # lamp gets dimmer: still works
            events = tracker.update(prev, cur, who)
            self.assertEqual(events[0].kind, "placed")
            truth = sc.sim(LAYOUT.cell(cell))
            self.assertLess(math.dist(events[0].chip.center, truth), 0.4 * LAYOUT.pitch * sc.s)   # < 5 mm class
            prev = cur
        self.assertEqual(sorted(c.owner for c in tracker.chips.values()), ["opponent", "robot", "robot"])
        # the opponent lifts its chip from p1 and puts it on p3: identity and owner stay
        cells[0], cells[2] = "none", "blue"
        sc, cur = self._frame(cells, 0.9, 9)
        events = tracker.update(prev, cur, "opponent")
        self.assertEqual(events[0].kind, "moved")
        self.assertEqual(events[0].chip.owner, "opponent")
        self.assertEqual(len(tracker.chips), 3)


class Segmentation(unittest.TestCase):
    def test_touching_chips_are_split(self):
        img = np.full((240, 320, 3), 230, np.uint8)
        cv2.circle(img, (120, 120), 30, (200, 80, 30), -1)
        cv2.circle(img, (172, 120), 30, (200, 80, 30), -1)          # touching blue chips
        cv2.circle(img, (250, 60), 25, (60, 175, 60), -1)
        found = segment_chips(img)
        colours = sorted(i.color for i in found)
        self.assertEqual(colours, ["blue", "blue", "green"])


class EndToEnd(unittest.TestCase):
    """Whole game against a simulated human: camera + robot + turn token + minimax."""

    def _play(self, human_moves, rot=22, shift=(10, -8)):
        scene = Scene(LAYOUT, rot, shift, seed=11)
        scene.slots = {n: ("green" if int(n[1:]) <= 5 else "blue") for n in LAYOUT.slots}
        world = SimulatedRobotWorld(scene)
        robot = MockRobot(world.on_event)
        arm = ArmController(robot)
        queue = list(human_moves)

        def human_acts():                       # called while the runner "sleeps" = polling
            if scene.token == "B" and queue:    # robot passed the token: human moves + passes back
                scene.cells[queue.pop(0)] = "blue"
                scene.token = "A"

        camera = SimCamera(scene)
        vision = CalibratedVision(LAYOUT)
        a, b = scene.sim(LAYOUT.circle_a), scene.sim(LAYOUT.circle_b)
        vision.hand_eye = HandEye.fit(a, b, scene.true_hand_eye(a), scene.true_hand_eye(b))

        from v2_two_circle_calibration.play_calibrated import CalibratedBackend
        runner = GameRunner(arm, CalibratedBackend(camera, vision), "green", "blue", "minimax",
                            my_side="A", opp_side="B", my_slots=[f"c{i}" for i in range(1, 6)],
                            i_start=True, poll=0, sleep=lambda s: human_acts(), log=lambda *_: None,
                            timeout=20)
        winner = runner.run()
        return winner, runner, scene

    def test_robot_never_loses_and_chips_land_in_cells(self):
        winner, runner, scene = self._play([0, 8, 5, 7])
        self.assertIn(winner, ("O", None))
        board = board_from_colors(scene.cells, {"green": "O", "blue": "X"})
        self.assertEqual(board, runner.game.board)

    def test_clear_board_returns_chips_to_tray(self):
        _, runner, scene = self._play([0, 8, 5, 7])
        runner.clear_board()
        self.assertEqual(sum(c == "green" for c in scene.cells), 0)
        self.assertEqual(sum(v == "green" for v in scene.slots.values()), 5)


if __name__ == "__main__":
    unittest.main()
