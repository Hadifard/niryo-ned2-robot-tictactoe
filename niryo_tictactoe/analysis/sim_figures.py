"""
File: niryo_tictactoe/analysis/sim_figures.py
Description: Figures + numbers computed from the project's own modules (no robot needed).

    cd niryo_tictactoe && python -m analysis.sim_figures

Writes ../docs/images/*.png and ../docs/results/sim_results.json
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import json, math, os, random, time
import numpy as np
from analysis.style import *   # noqa
from common.game_logic import MinimaxAgent, RandomAgent, HybridAgent, TicTacToe, other, EMPTY, LINES
from common.trajectory import arc_waypoints
from common.depth import DepthModel, distance_from_diameter, CHIP_DIAMETER_M
from common.path_learning import PathLearner, Obstacle, path_points, length, min_clearance
from v1_fixed_layout import config_fixed as cfg

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
IMG, RES = os.path.join(OUT, "images"), os.path.join(OUT, "results")
os.makedirs(IMG, exist_ok=True); os.makedirs(RES, exist_ok=True)
R = {}

# ------------------------------------------------------------------ 1. game logic
def play(a, b, first):
    g, turn, ag = TicTacToe(), first, {"X": a, "O": b}
    while not g.is_over():
        g.make_move(ag[turn].get_move(g), turn); turn = other(turn)
    return g.winner()

def match(make_robot, make_opp, n=1000, seed=0):
    rng = random.Random(seed); w = d = l = 0
    for i in range(n):
        first = "X" if i % 2 else "O"
        res = play(make_opp("X", rng), make_robot("O", rng), first)   # robot = O (as in GameRunner)
        if res == "O": w += 1
        elif res is None: d += 1
        else: l += 1
    return dict(win=w, draw=d, loss=l, games=n)

mm = lambda p, r: MinimaxAgent(p, r); rnd = lambda p, r: RandomAgent(p, r); hyb = lambda p, r: HybridAgent(p, rng=r)
games = {"vs Random": match(mm, rnd), "vs Hybrid (2 smart moves, then random)": match(mm, hyb),
         "vs Minimax": match(mm, mm)}
games["Random vs Random (reference)"] = match(rnd, rnd)
R["games"] = games

# node counts: plain minimax vs alpha-beta (counting replica of common.game_logic._minimax)
cnt = {"n": 0}
def mini(board, to, me, a, b, depth, prune):
    cnt["n"] += 1
    g = TicTacToe(board); win = g.winner()
    if win == me: return 10 - depth
    if win: return depth - 10
    if not g.available_moves(): return 0
    best = -99 if to == me else 99
    for m in g.available_moves():
        board[m] = to
        v = mini(board, other(to), me, a, b, depth + 1, prune); board[m] = EMPTY
        if to == me: best = max(best, v); a = max(a, best)
        else: best = min(best, v); b = min(b, best)
        if prune and b <= a: break
    return best
cnt["n"] = 0; mini([EMPTY]*9, "O", "O", -99, 99, 0, False); full = cnt["n"]
cnt["n"] = 0; mini([EMPTY]*9, "O", "O", -99, 99, 0, True); ab = cnt["n"]
t0 = time.perf_counter(); MinimaxAgent("O").get_move(TicTacToe()); t_empty = (time.perf_counter() - t0) * 1e3
R["search"] = dict(nodes_plain=full, nodes_alphabeta=ab, reduction=full / ab, first_move_ms=t_empty)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=.45))
labels = list(games); y = np.arange(len(labels))[::-1]
left = np.zeros(len(labels))
for key, col, nm in (("win", GREEN, "robot wins"), ("draw", GRAY, "draw"), ("loss", RED, "robot loses")):
    v = np.array([games[k][key] / games[k]["games"] * 100 for k in labels])
    ax[0].barh(y, v, left=left, color=col, label=nm)
    for yi, vi, li in zip(y, v, left):
        if vi > 6: ax[0].text(li + vi / 2, yi, f"{vi:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    left += v
ax[0].set_yticks(y, labels); ax[0].set_xlim(0, 100); ax[0].set_xlabel("% of 1000 games (first move alternates)")
ax[0].set_title("Alpha-beta minimax agent: results"); ax[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.2)); ax[0].grid(False)
ax[1].bar(["plain\nminimax", "alpha-beta"], [full, ab], color=[GRAY, BLUE])
for i, v in enumerate([full, ab]): ax[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
ax[1].set_title(f"Nodes searched ({full/ab:.0f}x fewer)"); ax[1].set_ylabel("nodes visited", labelpad=8)
fig.savefig(f"{IMG}/fig_game_logic.png"); plt.close(fig)

# ------------------------------------------------------------------ 2. trajectory
src, dst = cfg.CHIP_SLOT_XY["c1"], cfg.cell_xy(4)
fine = arc_waypoints(src, dst, 0.08, 0.09, 0.05, 60); real = arc_waypoints(src, dst, 0.08, 0.09, 0.05, 7)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot([src[0], dst[0]], [src[1], dst[1]], "--", color=GRAY, label="straight chord (Cartesian)")
ax[0].plot([src[0]] + [p[0] for p in fine], [src[1]] + [p[1] for p in fine], color=BLUE, lw=2, label="polar interpolation (r, theta)")
ax[0].plot([src[0]] + [p[0] for p in real], [src[1]] + [p[1] for p in real], "o", color=ORANGE, label="7 waypoints sent to the robot")
ax[0].plot(0, 0, "s", color=DARK, ms=10); ax[0].text(0.005, 0.005, "robot base", fontsize=8)
for nm, p in {**{k: v for k, v in cfg.CHIP_SLOT_XY.items()}, **{f"p{i+1}": cfg.cell_xy(i) for i in range(9)}}.items():
    ax[0].plot(*p, ".", color=GRAY, ms=4)
ax[0].set_aspect("equal"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].set_title("Top view: tray c1 to cell p5"); ax[0].legend(fontsize=8, loc="lower left")
s = np.linspace(0, 1, 60)
z0, z1, lift = 0.08, 0.09, 0.05
ax[1].plot(s, z0 + (z1 - z0) * s + 4 * lift * s * (1 - s), color=BLUE, lw=2, label="parabolic z profile")
ax[1].plot(s, z0 + (z1 - z0) * s, "--", color=GRAY, label="straight line")
ax[1].plot([i / 7 for i in range(1, 8)], [p[2] for p in real], "o", color=ORANGE)
ax[1].annotate("vertical descent\n(last 30 mm)", xy=(1, z1), xytext=(.55, .098), arrowprops=dict(arrowstyle="->"), fontsize=8)
ax[1].set_xlabel("path parameter s"); ax[1].set_ylabel("tool height z (m)"); ax[1].set_title("Height profile"); ax[1].legend(fontsize=8)
fig.savefig(f"{IMG}/fig_trajectory.png"); plt.close(fig)

# ------------------------------------------------------------------ 3. depth from chip size
f_px = 800.0; rng = np.random.default_rng(3)
d_cal = np.array([120., 80., 60., 48.]); z_cal = f_px * CHIP_DIAMETER_M / d_cal + 0.002 + rng.normal(0, 0.001, 4)
d_cal_noisy = d_cal + rng.normal(0, 0.7, 4)
model = DepthModel().fit(d_cal_noisy, z_cal)
dd = np.linspace(35, 140, 200); truth = f_px * CHIP_DIAMETER_M / dd + 0.002
err = np.array([model.predict(d) for d in dd]) - truth
R["depth"] = dict(f_px=f_px, chip_mm=30, calibration_passes=4, fit_rms_mm=model.rms * 1e3,
                  mean_abs_err_mm=float(np.abs(err).mean() * 1e3), max_abs_err_mm=float(np.abs(err).max() * 1e3),
                  a=float(model.a), b=float(model.b))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(dd, truth * 1e3, color=GRAY, lw=3, label="true  Z = f D / d + offset")
ax[0].plot(dd, [model.predict(d) * 1e3 for d in dd], color=BLUE, lw=1.5, label="learned  Z = a/d + b")
ax[0].plot(d_cal_noisy, z_cal * 1e3, "o", color=ORANGE, label="4 calibration passes (noisy)")
ax[0].set_xlabel("apparent chip diameter d (px)"); ax[0].set_ylabel("camera-to-chip distance Z (mm)")
ax[0].set_title("Chip diameter as a metric ruler"); ax[0].legend(fontsize=8)
ax[1].plot(dd, err * 1e3, color=RED); ax[1].axhline(0, color=DARK, lw=.6)
ax[1].set_xlabel("apparent chip diameter d (px)"); ax[1].set_ylabel("prediction error (mm)")
ax[1].set_title(f"After 4 passes: mean |error| = {R['depth']['mean_abs_err_mm']:.2f} mm")
fig.savefig(f"{IMG}/fig_depth_model.png"); plt.close(fig)

# ------------------------------------------------------------------ 4. learned paths
learner = PathLearner().train(trials=150, seed=0)
hist = np.array([(h[0], h[1], h[2]) for h in learner.history])
test_rng = np.random.default_rng(123)
rows = []
fixed_lift = 0.14
for _ in range(300):
    a, b, obs, top = learner._random_transfer(test_rng)
    best = learner.best_lift(a, b, obs)
    pts, info = learner.plan(a, b, obs)
    direct = path_points(a, b, learner.predict_lift(float(np.linalg.norm(b[:2] - a[:2])), top))
    ok_direct = min_clearance(direct, obs, learner.monitor) >= learner.monitor.margin
    rows.append(dict(best=best, pred=learner.predict_lift(float(np.linalg.norm(b[:2]-a[:2])), top),
                     len_plan=length(pts), len_best=length(path_points(a, b, best)), len_fixed=length(path_points(a, b, fixed_lift)),
                     ok_plan=min_clearance(pts, obs, learner.monitor) >= learner.monitor.margin - 1e-9,
                     ok_direct=bool(ok_direct), replanned=info["replanned"],
                     ok_fixed=min_clearance(path_points(a, b, fixed_lift), obs, learner.monitor) >= learner.monitor.margin))
best = np.array([r["best"] for r in rows]); pred = np.array([r["pred"] for r in rows])
ss = ((best - pred) ** 2).sum(); r2 = 1 - ss / ((best - best.mean()) ** 2).sum()
Lplan = np.mean([r["len_plan"] for r in rows]); Lfix = np.mean([r["len_fixed"] for r in rows]); Lbest = np.mean([r["len_best"] for r in rows])
R["path_learning"] = dict(train_trials=150, test_transfers=300, r2=float(r2), mae_mm=float(np.abs(best - pred).mean() * 1e3),
    collision_free_plan_pct=100 * float(np.mean([r["ok_plan"] for r in rows])),
    collision_free_direct_prediction_pct=100 * float(np.mean([r["ok_direct"] for r in rows])),
    replanned_pct=100 * float(np.mean([r["replanned"] for r in rows])),
    collision_free_fixed_max_lift_pct=100 * float(np.mean([r["ok_fixed"] for r in rows])),
    mean_len_plan_mm=float(Lplan * 1e3), mean_len_exhaustive_best_mm=float(Lbest * 1e3), mean_len_fixed_max_lift_mm=float(Lfix * 1e3),
    length_saving_vs_fixed_pct=float(100 * (1 - Lplan / Lfix)))

fig = plt.figure(figsize=(12, 7.5))
ax = fig.add_subplot(2, 3, 1)
ax.plot(hist[:, 0], hist[:, 1] * 1e3, ".", color=BLUE, alpha=.6); ax.set_xlabel("training trial"); ax.set_ylabel("best collision-free lift (mm)")
ax.set_title("Best lift found in each practice transfer")
ax = fig.add_subplot(2, 3, 2)
ax.plot([best.min()*1e3, best.max()*1e3], [best.min()*1e3, best.max()*1e3], "--", color=GRAY)
ax.plot(best * 1e3, pred * 1e3, ".", color=GREEN, alpha=.6); ax.set_xlabel("searched best lift (mm)"); ax.set_ylabel("regression prediction (mm)")
ax.set_title(f"Held-out 300 transfers: R2 = {r2:.3f}")
ax = fig.add_subplot(2, 3, 3)
vals = [Lbest * 1e3, Lplan * 1e3, Lfix * 1e3]
ax.bar(["exhaustive\nsearch", "learned\n(this work)", "fixed max\nlift"], vals, color=[GRAY, BLUE, RED])
for i, v in enumerate(vals): ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(min(vals) * .8, max(vals) * 1.03); ax.set_ylabel("mean path length (mm)"); ax.set_title(f"{R['path_learning']['length_saving_vs_fixed_pct']:.1f}% shorter than fixed lift")
# 3D example
a, b = np.array([0.12, 0.05, 0.06]), np.array([0.32, -0.05, 0.06])
obs = [Obstacle(0.22, 0.0, 0.02, 0.10)]
pts, info = learner.plan(a, b, obs)
ax = fig.add_subplot(2, 3, 4, projection="3d")
def cyl(ax, o, col):
    th = np.linspace(0, 2*np.pi, 30); zz = np.linspace(0, o.top, 2)
    X = o.x + o.radius*np.outer(np.cos(th), np.ones(2)); Y = o.y + o.radius*np.outer(np.sin(th), np.ones(2)); Z = np.outer(np.ones(30), zz)
    ax.plot_surface(X, Y, Z, color=col, alpha=.5)
cyl(ax, obs[0], RED)
ax.plot(*path_points(a, b, 0.0).T, "--", color=GRAY, label="no lift")
ax.plot(*pts.T, color=BLUE, lw=2.5, label=f"learned (lift {info['lift']*1e3:.0f} mm)")
ax.set_title("3-D parabolic path over a chip"); ax.legend(fontsize=7, loc="upper left"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
# sudden obstacle
a2, b2 = np.array([0.12, 0.0, 0.06]), np.array([0.32, 0.0, 0.06]); wall = [Obstacle(0.22, 0.0, 0.03, 0.30)]
pts2, info2 = learner.plan(a2, b2, wall)
R["path_learning"]["detour_demo"] = dict(lateral_mm=float(info2["lateral"] * 1e3), lift_mm=float(info2["lift"] * 1e3),
                                         length_mm=float(length(pts2) * 1e3), straight_mm=float(np.linalg.norm(b2 - a2) * 1e3))
ax = fig.add_subplot(2, 3, 5)
th = np.linspace(0, 2*np.pi, 60); ax.fill(0.22 + 0.03*np.cos(th), 0.03*np.sin(th), color=RED, alpha=.5, label="sudden obstacle (30 cm tall)")
ax.plot(a2[0], a2[1], "o", color=DARK); ax.plot(b2[0], b2[1], "s", color=DARK)
ax.plot([a2[0], b2[0]], [a2[1], b2[1]], "--", color=GRAY, label="blocked chord")
ax.plot(pts2[:, 0], pts2[:, 1], color=BLUE, lw=2.5, label=f"detour: {info2['lateral']*1e3:+.0f} mm sideways")
ax.set_aspect("equal"); ax.set_ylim(-0.07, 0.07); ax.set_title("Lateral detour search (top view)"); ax.legend(fontsize=7, loc="lower center"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax = fig.add_subplot(2, 3, 6)
cats = ["learned plan\n(w/ replanning)", "learned lift\n(direct)", "fixed max\nlift"]
pc = [R["path_learning"]["collision_free_plan_pct"], R["path_learning"]["collision_free_direct_prediction_pct"], R["path_learning"]["collision_free_fixed_max_lift_pct"]]
ax.bar(cats, pc, color=[BLUE, GREEN, RED])
for i, v in enumerate(pc): ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, 112); ax.set_ylabel("% collision-free (300 transfers)"); ax.set_title("Safety")
fig.tight_layout(); fig.savefig(f"{IMG}/fig_path_learning.png"); plt.close(fig)

# ------------------------------------------------------------------ 5. similarity transform illustration
from v2_two_circle_calibration.calibration import BoardLayout, Similarity2D
L = BoardLayout()
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
for a_, (rot, sc, sh, ttl) in zip(ax, [(0, 1.0, (0, 0), "paper as printed (mm)"), (25, 1.15, (40, -20), "rotated 25 deg, +15 % scale, shifted"), (140, 0.85, (-30, 30), "rotated 140 deg, -15 % scale")]):
    m = complex(math.cos(math.radians(rot)), math.sin(math.radians(rot))) * sc
    sim = Similarity2D.from_two_points(L.circle_a, L.circle_b, (m*complex(*L.circle_a)).real+sh[0] + 1j*0 if False else (sh[0], sh[1]), (m*complex(*L.circle_b) + complex(*sh)).real and ((m*complex(*L.circle_b) + complex(*sh)).real, (m*complex(*L.circle_b) + complex(*sh)).imag))
    a_.set_aspect("equal"); a_.invert_yaxis(); a_.set_title(ttl, fontsize=10)
    for i in range(9):
        p = sim(L.cell(i)); a_.add_patch(plt.Circle(p, 0.5*L.chip_diameter*sim.scale, fill=False, color=GRAY))
    for nm, p in L.slots.items(): a_.add_patch(plt.Circle(sim(p), 0.5*L.chip_diameter*sim.scale, fill=False, color=GRAY, ls=":"))
    for side, p in L.token_spots.items(): a_.plot(*sim(p), "s", color=RED)
    a_.plot(*sim(L.circle_a), "o", color="#ec4899", ms=9); a_.plot(*sim(L.circle_b), "o", color=GREEN, ms=9)
    a_.set_xlim(-60, 380); a_.set_ylim(300, -90) if rot == 0 else a_.autoscale()
    a_.set_aspect("equal")
fig.suptitle("Two circles fix rotation, scale and shift in closed form (p' = m p + t, complex m, t): grid, trays and token spots follow", fontsize=10)
fig.savefig(f"{IMG}/fig_two_circle_transform.png"); plt.close(fig)

json.dump(R, open(f"{RES}/sim_results.json", "w"), indent=2)
print(json.dumps(R, indent=1))
