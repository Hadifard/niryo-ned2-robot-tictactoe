"""
File: analysis/vision_benchmark.py
Description: Benchmarks the three vision variants (v1 fixed pixels, v2 two-circle calibration, v3 background
subtraction) on SYNTHETIC camera frames (tests/synth.py) under rotation, shift, dim light, colour cast and
shadows, and times every vision stage on the local CPU. Writes docs/results/vision_benchmark.json and
docs/images/fig_vision_benchmark.png. Run:  python -m analysis.vision_benchmark
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import json, os, random, time
import numpy as np
from analysis.style import *   # noqa
from common.game_logic import board_from_colors
from segmentation.classical_segmentation import segment_chips
from tests.synth import Scene
from v1_fixed_layout.vision_fixed import FixedVision
from v2_two_circle_calibration.calibration import BoardLayout, CalibrationError
from v2_two_circle_calibration.vision_calibrated import CalibratedVision
from v3_robust_background_subtraction.vision_robust import RobustVision

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
L = BoardLayout()
CONDITIONS = [  # name, kwargs for Scene
    ("reference\n(nominal pose, normal light)", dict(rot_deg=0, shift_px=(0, 0))),
    ("rotated 30 deg", dict(rot_deg=30, shift_px=(0, 0))),
    ("rotated 150 deg\n+ shifted", dict(rot_deg=150, shift_px=(-35, 25))),
    ("dim light\n(gain 0.5)", dict(rot_deg=0, shift_px=(0, 0), gain=(.5, .5, .5))),
    ("very dim\n(gain 0.35)", dict(rot_deg=0, shift_px=(0, 0), gain=(.35, .35, .35))),
    ("warm colour cast", dict(rot_deg=0, shift_px=(0, 0), gain=(1.2, 1.0, .8))),
    ("cold colour cast", dict(rot_deg=0, shift_px=(0, 0), gain=(.8, .9, 1.15))),
    ("shadow band", dict(rot_deg=0, shift_px=(0, 0), shadow=.35)),
    ("combined: rot 40, shift,\ndim cast, shadow", dict(rot_deg=40, shift_px=(-25, 18), gain=(.7, .8, .95), shadow=.3)),
]
N = 20


def random_board(rng):
    return [rng.choice(["none", "none", "green", "blue"]) for _ in range(9)]


def make_scene(kw, board, seed):
    sc = Scene(L, seed=seed, **kw)
    sc.cells, sc.slots, sc.token = list(board), {n: "none" for n in L.slots}, None
    return sc


def main():
    nominal = Scene(L, 0, (0, 0))
    cell_px = {f"p{i + 1}": tuple(map(int, nominal.sim(L.cell(i)))) for i in range(9)}
    token_px = {k: tuple(map(int, nominal.sim(v))) for k, v in L.token_spots.items()}
    v1 = FixedVision(cell_px, token_px, radius=int(0.35 * L.pitch * nominal.s))
    empty = make_scene(dict(rot_deg=0, shift_px=(0, 0)), ["none"] * 9, 1)
    v3 = RobustVision(L); v3.learn_background([empty.render() for _ in range(5)])
    rng = random.Random(7)
    res = {}
    for ci, (name, kw) in enumerate(CONDITIONS):
        acc = {"v1": [], "v2": [], "v3": []}; exact = {"v1": 0, "v2": 0, "v3": 0}
        for k in range(N):
            board = random_board(rng); frame = make_scene(kw, board, 100 * ci + k).render()
            outs = {}
            outs["v1"] = v1.read_once(frame).cells
            try: outs["v2"] = CalibratedVision(L).observe(frame).cells
            except CalibrationError: outs["v2"] = ["fail"] * 9
            try: outs["v3"] = v3.observe(frame).cells
            except CalibrationError: outs["v3"] = ["fail"] * 9
            for key, cells in outs.items():
                ok = sum(a == b for a, b in zip(cells, board)); acc[key].append(ok / 9); exact[key] += ok == 9
        res[name.replace("\n", " ")] = {k: dict(cell_accuracy=float(np.mean(v) * 100), exact_board_pct=100 * exact[k] / N) for k, v in acc.items()}
        print(name.replace("\n", " "), {k: round(v["cell_accuracy"], 1) for k, v in res[name.replace("\n", " ")].items()}, flush=True)

    # timing (median of 30 frames, 640x480, this CPU)
    sc = make_scene(dict(rot_deg=20, shift_px=(5, 5)), ["blue", "none", "green", "none", "blue", "none", "green", "none", "blue"], 5)
    frame = sc.render(); timing = {}
    def tm(fn, n=30):
        ts = []
        for _ in range(n):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
        return float(np.median(ts))
    timing["v1 read_once (fixed pixels)"] = tm(lambda: v1.read_once(frame))
    timing["v2 observe (two circles + 9 cells + 10 trays)"] = tm(lambda: CalibratedVision(L).observe(frame))
    timing["v3 observe (rectify + Lab background subtraction)"] = tm(lambda: v3.observe(frame))
    timing["classical segmentation (HSV + watershed)"] = tm(lambda: segment_chips(frame))
    res["timing_ms"] = timing; res["n_boards_per_condition"] = N
    json.dump(res, open(f"{OUT}/results/vision_benchmark.json", "w"), indent=2)
    print(json.dumps(timing, indent=1))

    names = [c[0] for c in CONDITIONS]; x = np.arange(len(names)); w = .27
    fig, ax = plt.subplots(figsize=(13, 4.6))
    for j, (key, col, lab) in enumerate((("v1", GRAY, "v1 fixed pixels"), ("v2", BLUE, "v2 two-circle calibration"), ("v3", GREEN, "v3 background subtraction"))):
        vals = [res[n.replace("\n", " ")][key]["cell_accuracy"] for n in names]
        ax.bar(x + (j - 1) * w, vals, w, color=col, label=lab)
    ax.set_xticks(x, names, fontsize=8); ax.set_ylabel("cell accuracy (%)"); ax.set_ylim(0, 122)
    ax.set_title(f"Board reading accuracy on synthetic frames ({N} random boards per condition x 9 cells)")
    ax.legend(ncol=3, loc="upper center")
    fig.savefig(f"{OUT}/images/fig_vision_benchmark.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    ks = list(timing); ax.barh(ks, [timing[k] for k in ks], color=BLUE)
    for i, k in enumerate(ks): ax.text(timing[k], i, f" {timing[k]:.1f} ms", va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("median time per 640x480 frame (ms, this cloud CPU)"); ax.set_title("Vision cost: classical methods, no GPU")
    ax.set_xlim(0, max(timing.values()) * 1.25)
    fig.savefig(f"{OUT}/images/fig_vision_timing.png"); plt.close(fig)


if __name__ == "__main__":
    main()
