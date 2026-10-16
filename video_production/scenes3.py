"""
File: video_production/scenes3.py
Description: See the code below; this file has no module description yet.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""

import sys, math, time
sys.path.insert(0, "/home/claude/niryo_tictactoe"); sys.path.insert(0, "/home/claude/film")
import cv2, numpy as np
from PIL import Image, ImageDraw
import render2 as R
from render2 import *
from common.depth import DepthModel
from common.path_learning import PathLearner, Obstacle, path_points, min_clearance, length
from v3_robust_background_subtraction.diff_tracker import DiffChipTracker

R.CLIPS["depth_strip"] = Clip(CL + "depth_strip.mp4")
PAPER = cv2.cvtColor(cv2.imread("/mnt/user-data/uploads/1000156090.jpg"), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------------ hero
def make_hero3():
    im, d = static_page("", "", [(30, 440, 1020, 520)])
    d.rectangle((30, 30, 1050, 350), fill=PANEL, outline=LINE, width=2)
    bg = np.asarray(im).copy()

    def draw(canvas, tl, pre):
        f = R.CLIPS["hero"].frame(tl)
        reveal = int(1020 * ease_out((tl - 0.1) / 0.9))
        if reveal > 0:
            canvas[440:960, 30:30 + reveal] = fit_cover(f, 1020, 520, 0.5, 0.35)[:, :reveal]

        def ov(d):
            a = clamp01((tl - 0.2) / 0.9)                                   # ring draws itself
            cx, cy, r = 380, 190, 105
            if a > 0:
                d.arc((cx - r, cy - r, cx + r, cy + r), -90, -90 + 360 * ease_out(a), fill=(60, 130, 255), width=26)
            b = ease_out((tl - 0.9) / 0.7)                                  # cross draws itself
            x0, x1, y0, y1 = 590, 790, 85, 295
            if b > 0:
                p = min(1.0, b * 2)
                d.line((x0, y0, x0 + (x1 - x0) * p, y0 + (y1 - y0) * p), fill=(255, 255, 255), width=26)
                if b > 0.5:
                    q = (b - 0.5) * 2
                    d.line((x1, y0, x1 + (x0 - x1) * q, y0 + (y1 - y0) * q), fill=(255, 255, 255), width=26)
            c = ease_out((tl - 1.6) / 0.7)
            d.text((38, 408), "Computer vision & Robotics", font=font("b", 40), fill=(DARK[0], DARK[1], DARK[2], int(255 * c)), anchor="ls")
            d.rectangle((38, 416, 38 + int(200 * c), 420), fill=ACC)
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# ----------------------------------------------------------------- bg subtraction
def make_bgsub3():
    from tests.synth import Scene as SS
    T = "v3_robust_background_subtraction/diff_tracker.py"
    f1, f2, f3 = (block(ROOT + T, p) for p in (r"    def regions", r"    def difference", r"    def update"))
    panel = CodePanel(30, 524, 1020, 442, [dict(title=T, first=f1[0], text=f1[1], t0=0.4, t1=7.5), dict(title=T, first=f2[0], text=f2[1], t0=8.0, t1=14.0),
                                         dict(title=T, first=f3[0], text=f3[1], t0=14.5, t1=27.5)])
    bg = std_scene("03", "BACKGROUND SUBTRACTION METHOD", [(30, 108, 224, 400), (267, 108, 400, 400)], [(679, 108, 371, 400, "Image subtraction · 2 fps", ACC)])
    # ---- precompute the game: every step = (previous frame, current frame, mover)
    plan = [(4, "robot", "green"), (0, "opponent", "blue"), (8, "robot", "green"), (2, "opponent", "blue"), (6, "robot", "green"), ("mv", "opponent", None)]
    cells, steps, tracker = ["none"] * 9, [], DiffChipTracker(min_area=180, threshold=16)
    rng = np.random.default_rng(5)

    def frame(k):
        sc = SS(LAYOUT, 0, (int(rng.integers(-1, 2)), int(rng.integers(-1, 2))), px_per_mm=1.2, seed=int(rng.integers(0, 999)), noise=6.0,
                gain=(1 - 0.05 * k,) * 3, irregular=True, shadow=0.0)
        sc.cells, sc.slots, sc.token = list(cells), {n: "none" for n in LAYOUT.slots}, None
        return sc.render()
    prev = frame(0)
    for k, (cell, who, colour) in enumerate(plan, 1):
        if cell == "mv":
            cells[2], cells[1] = "none", "blue"                             # the opponent re-seats a chip
        else:
            cells[cell] = colour
        cur = frame(k)
        ev = tracker.update(prev, cur, who)
        ap, va, dd = tracker.difference(prev, cur)
        mask = cv2.bitwise_or(ap, va)
        steps.append(dict(prev=prev, cur=cur, d=dd, mask=mask, ev=ev, who=who, chips={i: (c.center, c.owner) for i, c in tracker.chips.items()},
                          regions=[(r[1], r[2]) for r in tracker.regions(ap)[:1] + tracker.regions(va)[:1]], dist=tracker.distances()))
        prev = cur
    X0, Y0, X1, Y1 = 200, 120, 440, 360

    def crop(img):
        return cv2.resize(img[Y0:Y1, X0:X1], (200, 200), interpolation=cv2.INTER_LINEAR)

    def draw(canvas, tl, pre):
        paste(canvas, R.CLIPS["bg_07"].frame(tl), 30, 108, 224, 400)
        k = min(len(steps) - 1, max(0, int((tl - 1.0) / 3.6)))
        st, local = steps[k], (tl - 1.0) - k * 3.6
        rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        paste(canvas, rgb(crop(st["prev"])), 267, 108, 200, 200)
        paste(canvas, rgb(crop(st["cur"])), 467, 108, 200, 200)
        dn = np.clip(128 + st["d"] * 3.0, 0, 255).astype(np.uint8)
        dimg = cv2.cvtColor(dn, cv2.COLOR_GRAY2RGB) if local > 0.8 else np.full((480, 640, 3), 24, np.uint8)
        paste(canvas, crop(dimg), 267, 308, 200, 200)
        res = np.full((480, 640, 3), 18, np.uint8)
        if local > 1.6:
            res[st["mask"] > 0] = (40, 70, 95)
        res = cv2.resize(res[Y0:Y1, X0:X1], (200, 200), interpolation=cv2.INTER_NEAREST)
        sx = 200 / (X1 - X0)
        owners = {"robot": (0, 210, 255), "opponent": (255, 150, 60)}
        if local > 1.6:
            for cnt_i, (centre, cnt) in enumerate(st["regions"]):
                pts = ((cnt.reshape(-1, 2) - (X0, Y0)) * sx).astype(np.int32)
                cv2.polylines(res, [pts], True, (90, 255, 160) if cnt_i == 0 else (120, 120, 255), 1, cv2.LINE_AA)
            cs = [(((c[0] - X0) * sx, (c[1] - Y0) * sx), o) for c, o in st["chips"].values()]
            for (p, o) in cs:
                x, y = int(p[0]), int(p[1])
                cv2.line(res, (x - 9, y), (x + 9, y), owners[o], 2, cv2.LINE_AA)
                cv2.line(res, (x, y - 9), (x, y + 9), owners[o], 2, cv2.LINE_AA)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    (p, _), (q, _) = cs[i], cs[j]
                    if math.dist(p, q) < 105:
                        cv2.line(res, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (200, 200, 200), 1, cv2.LINE_AA)
                        mm = math.dist(p, q) / sx / 1.2
                        cv2.putText(res, f"{mm:.0f}", (int((p[0] + q[0]) / 2) + 2, int((p[1] + q[1]) / 2) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
        paste(canvas, res, 467, 308, 200, 200)
        title, ln, n = panel.draw(canvas, tl, None)

        def ov(d):
            tag(d, 40, 124, "REAL FOOTAGE", ACC2)
            for (x, y, s, c) in ((277, 118, "PREVIOUS", ACC), (477, 118, "CURRENT", ACC), (277, 318, "SUBTRACTION", GOOD), (477, 318, "AREAS · CENTRES", ACC3)):
                tag(d, x, y, s, c)
            code_head(d, (30, 524), title, ln, n)
            who = st["who"]
            d.text((697, 160), f"step {k + 1}/6   mover: {who}", font=font("mb", 15), fill=owners[who])
            bullets(d, 695, 184, 340, ["Photograph the board twice per second; subtract consecutive images",
                                       "Largest closed area = chip; its centroid is the position",
                                       "Seat play is about 5 mm: noisy, uneven edges, so no circle fit",
                                       "Owner is tracked from the first move; no colour detection"], tl, 1.5, step=3.2, size=14, gap=5)
            a = ease_out((tl - 17.0) / 0.8)
            d.rectangle((695, 400, 1034, 496), fill=(PANEL2[0], PANEL2[1], PANEL2[2], int(255 * a)), outline=(ACC[0], ACC[1], ACC[2], int(255 * a)), width=1)
            d.text((705, 407), "REFERENCE", font=font("b", 12), fill=(ACC[0], ACC[1], ACC[2], int(255 * a)))
            cite = "H. Sarhangi Fard (2022). Improving Moving Vehicles Detection and Vehicles Trajectory Plotting using Optimized Background Subtraction Method. Expert Systems with Applications."
            for j, ln_ in enumerate(wrap(cite, font("r", 12), 322)):
                d.text((705, 424 + j * 15), ln_, font=font("r", 12), fill=(TXT[0], TXT[1], TXT[2], int(255 * a)))
            pa = ease_out((tl - 5.5) / 0.4) * (1 - ease((tl - 7.6) / 0.4))     # first page of the paper, about 2 s
            if pa > 0.02:
                w_, h_ = 190, 259
                x, y = 440, 190
                d.rectangle((x + 6, y + 6, x + w_ + 6, y + h_ + 6), fill=(0, 0, 0, int(110 * pa)))
        pre.append(ov)
        pa = ease_out((tl - 5.5) / 0.4) * (1 - ease((tl - 7.6) / 0.4))
        if pa > 0.02:
            card = cv2.resize(PAPER, (190, 259), interpolation=cv2.INTER_AREA)
            roi = canvas[190:449, 440:630]
            roi[:] = (card.astype(np.float32) * pa + roi.astype(np.float32) * (1 - pa)).astype(np.uint8)
            canvas[190:192, 440:630] = ACC
    return dict(bg=bg, draw=draw)


# -------------------------------------------------------------------------- depth
def make_depth():
    T = "common/depth.py"
    a = block(ROOT + T, r"^class DepthModel")
    b = block(ROOT + T, r"^class SafetyMonitor")
    panel = CodePanel(30, 679, 1020, 287, [dict(title=T, first=a[0], text=a[1], t0=0.5, t1=8.0), dict(title=T, first=b[0], text=b[1], t0=8.4, t1=15.5)])
    bg = std_scene("06", "DEPTH ESTIMATION", [(30, 108, 1020, 288)], [(30, 407, 500, 260), (542, 407, 508, 260)])

    def draw(canvas, tl, pre):
        paste(canvas, R.CLIPS["depth_strip"].frame(tl), 30, 108, 1020, 288)
        title, ln, n = panel.draw(canvas, tl, None)

        def ov(d):
            tag(d, 46, 124, "ORIGINAL", ACC2)
            tag(d, 556, 124, "RELATIVE DEPTH HEAT MAP · DEPTH ANYTHING V2", ACC3)
            code_head(d, (30, 679), title, ln, n)
            # similar-triangle schematic
            ox, oy = 50, 537
            d.text((50, 428), "Pinhole model: similar triangles", font=font("b", 16), fill=ACC)
            Z = 0.30 + 0.14 * math.sin(tl * 0.9)
            f, Dm = 0.05, 0.03
            sx = 440.0
            cam = (ox + 4, oy)
            plane = ox + 40
            chipx = plane + (Z / 0.5) * 330
            dpx = f * Dm / Z * sx * 2.2
            dch = Dm / 0.03 * 52
            d.line((cam[0], oy, chipx + 40, oy), fill=(90, 100, 128), width=1)
            d.ellipse((cam[0] - 6, oy - 6, cam[0] + 6, oy + 6), fill=ACC)
            d.line((plane, oy - 70, plane, oy + 70), fill=(150, 160, 190), width=2)
            d.line((plane, oy - dpx / 2, plane, oy + dpx / 2), fill=(255, 200, 80), width=6)
            d.rectangle((chipx - 4, oy - dch / 2, chipx + 4, oy + dch / 2), fill=(70, 130, 255))
            for sgn in (-1, 1):
                d.line((cam[0], oy, chipx, oy + sgn * dch / 2), fill=(ACC[0], ACC[1], ACC[2], 150), width=1)
            d.text((plane + 4, oy - 78), "image plane", font=font("r", 11), fill=DIM)
            d.text((plane + 3, oy + 78), f"d = {dpx:4.0f} px", font=font("mb", 12), fill=(255, 200, 80))
            d.text((chipx - 20, oy + dch / 2 + 8), "chip D = 30 mm", font=font("r", 11), fill=TXT)
            d.line((cam[0], oy + 100, chipx, oy + 100), fill=(255, 90, 70), width=2)
            d.text(((cam[0] + chipx) / 2, oy + 112), f"Z = {Z * 1000:3.0f} mm", font=font("mb", 14), fill=(255, 120, 100), anchor="mm")
            d.text((plane / 2 + cam[0] / 2, oy - 18), "f", font=font("mb", 13), fill=DIM, anchor="mm")
            # formulas
            lines = [("d = f · D / Z", "pinhole projection"), ("Z = f · D / d", "chip diameter D = 30 mm is the metric ruler"),
                     ("Z₂ / Z₁ = d₁ / d₂", "relative distance of two chips"), ("mm / px = D / d", "scale at the chip plane"),
                     ("z_tool ≥ z_obs + r_cup + margin", "clearance to chips, board and other arm"),
                     ("Z ≈ a / d + b", "learned from a few passes, then no re-measuring")]
            for i, (fm, note) in enumerate(lines):
                al = ease_out((tl - (1.0 + i * 1.6)) / 0.5)
                if al <= 0:
                    continue
                y = 424 + i * 40
                d.text((560, y), fm, font=font("mb", 19), fill=(ACC[0], ACC[1], ACC[2], int(255 * al)))
                d.text((560, y + 21), note, font=font("r", 12), fill=(DIM[0], DIM[1], DIM[2], int(255 * al)))
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# ------------------------------------------------------------------------ motion
def make_motion3():
    learner = PathLearner().train(trials=70, seed=3)
    TT = "common/path_learning.py"
    c1, c2, c3 = (block(ROOT + TT, p) for p in (r"^def path_points", r"    def best_lift", r"    def plan"))
    panel = CodePanel(30, 524, 1020, 442, [dict(title=TT, first=c1[0], text=c1[1], t0=0.4, t1=8.0), dict(title=TT, first=c2[0], text=c2[1], t0=8.4, t1=17.0),
                                         dict(title=TT, first=c3[0], text=c3[1], t0=17.4, t1=31.0)])
    bg = std_scene("07", "MOTION PLANNING · LEARNED 3-D PATHS", [(30, 108, 296, 400)], [(338, 108, 420, 400), (770, 108, 280, 400, "Learning from trials", ACC3)])
    # world scenario (metres, x forward, y left, z up)
    existing = [Obstacle(0.305, 0.0525, 0.016, 0.05), Obstacle(0.195, -0.052, 0.016, 0.05), Obstacle(0.25, 0.0525, 0.016, 0.05)]
    slot, p5, redA, redB = (0.12, 0.10, 0.06), (0.25, 0.0, 0.06), (0.2169, -0.1021, 0.06), (0.2967, -0.1043, 0.06)
    P1, _ = learner.plan(slot, p5, existing)
    P2, _ = learner.plan(p5, redA, existing)
    P3a, _ = learner.plan(redA, redB, [])
    block_o = Obstacle(0.2568, -0.103, 0.015, 0.16)
    P3b, info3 = learner.plan(redA, redB, [block_o])
    # trial set for the learning view
    a_t, b_t, o_t = (0.12, 0.05, 0.06), (0.32, -0.05, 0.06), [Obstacle(0.22, 0.0, 0.02, 0.10)]
    grid = np.linspace(0.0, 0.14, 29)
    trials = [(lf, path_points(a_t, b_t, lf), min_clearance(path_points(a_t, b_t, lf), o_t, learner.monitor) >= learner.monitor.margin) for lf in grid]
    best_lift = learner.best_lift(a_t, b_t, o_t)
    rng = np.random.default_rng(9)
    data = []
    for _ in range(40):
        h = rng.uniform(0.07, 0.13)
        data.append((h, learner.best_lift(a_t, b_t, [Obstacle(0.22, 0.0, 0.02, h)])))
    phi, th, sc3 = math.radians(28), math.radians(48), 1500.0
    CX, CY = 338 + 215, 108 + 262

    def pj(p):
        x, y, z = p
        u = -y * math.cos(phi) + x * math.sin(phi) - 0.25 * math.sin(phi)
        v = z * math.cos(th) + (x * math.cos(phi) + y * math.sin(phi) - 0.25 * math.cos(phi)) * math.sin(th)
        return (CX + sc3 * u, CY - sc3 * v)

    def on(points, s):
        i = min(len(points) - 1, int(s * (len(points) - 1)))
        return points[i]

    def draw(canvas, tl, pre):
        paste(canvas, R.CLIPS["s6_rob"].frame(tl), 30, 108, 296, 400)
        title, ln, n = panel.draw(canvas, tl, None)

        def ov(d):
            tag(d, 46, 124, "ROBOT PICK & PLACE", ACC2)
            code_head(d, (30, 524), title, ln, n)
            # ---- 3-D view
            d.text((352, 130), "Dynamic 3-D view · parabolic paths", font=font("b", 14), fill=ACC)
            bz = 0.04
            corners = [(0.17, -0.10, bz), (0.33, -0.10, bz), (0.33, 0.10, bz), (0.17, 0.10, bz)]
            d.polygon([pj(c) for c in corners], fill=(34, 44, 70), outline=LINE)
            for k in range(1, 3):
                for a_, b_ in (((0.17 + k * 0.16 / 3, -0.10, bz), (0.17 + k * 0.16 / 3, 0.10, bz)), ((0.17, -0.10 + k * 0.2 / 3, bz), (0.33, -0.10 + k * 0.2 / 3, bz))):
                    d.line([pj(a_), pj(b_)], fill=LINE, width=1)
            bx, by = pj((0.0, 0.0, 0.0))
            if by < 480:
                d.ellipse((bx - 18, by - 10, bx + 18, by + 10), fill=(70, 80, 110), outline=ACC, width=2)
            d.text((352, 148), "x: forward from the robot base   y: left   z: up", font=font("r", 11), fill=DIM)

            def chip(p, col, r=0.015):
                x, y = pj((p[0], p[1], bz))
                d.ellipse((x - r * sc3 * 0.9, y - r * sc3 * 0.5, x + r * sc3 * 0.9, y + r * sc3 * 0.5), fill=col)

            def cyl(o, col, alpha=255):
                if o.top <= bz + 0.002:
                    return
                x0, y0 = pj((o.x, o.y, bz))
                x1, y1 = pj((o.x, o.y, o.top))
                rr = o.radius * sc3
                d.rectangle((x0 - rr * 0.9, y1, x0 + rr * 0.9, y0), fill=(col[0], col[1], col[2], alpha))
                d.ellipse((x1 - rr * 0.9, y1 - rr * 0.45, x1 + rr * 0.9, y1 + rr * 0.45), fill=(min(255, col[0] + 40), min(255, col[1] + 40), min(255, col[2] + 40), alpha))
            for o in existing:
                chip((o.x, o.y), (90, 100, 130))
            if tl < 11:                                    # --- learning phase
                k = min(len(trials), int(tl / 0.34) + 1)
                for lf, pts, ok in trials[:k]:
                    d.line([pj(p) for p in pts], fill=(70, 210, 110, 120) if ok else (235, 70, 60, 120), width=1)
                cyl(o_t[0], (120, 130, 160))
                if k >= len(trials):
                    d.line([pj(p) for p in path_points(a_t, b_t, best_lift)], fill=ACC, width=4)
                    d.text((560, 440), f"shortest safe lift {best_lift * 1000:.0f} mm", font=font("b", 13), fill=ACC, anchor="mm")
                d.text((560, 465), f"trial {min(k, len(trials))}/{len(trials)}   green = clear, red = collision", font=font("m", 11), fill=TXT, anchor="mm")
            else:                                          # --- execution phase
                for pts, col, wd in ((P1, ACC, 2), (P2, (150, 160, 190), 1)):
                    d.line([pj(p) for p in pts], fill=col, width=wd)
                red_pos = redA
                if tl >= 24.5:
                    obs_a = ease_out((tl - 24.5) / 0.6)
                    cyl(Obstacle(block_o.x, block_o.y, block_o.radius, block_o.top * obs_a), (235, 150, 60))
                if tl < 24.5 or tl < 27.0:
                    d.line([pj(p) for p in P3a], fill=(235, 70, 60) if tl >= 24.5 else (150, 160, 190), width=2)
                if tl >= 27.0:
                    d.line([pj(p) for p in P3b], fill=(255, 200, 80), width=4)
                    d.text((560, 465), f"replanned: lateral {info3['lateral'] * 1000:+.0f} mm, lift {info3['lift'] * 1000:.0f} mm", font=font("b", 12), fill=(255, 200, 80), anchor="mm")
                elif tl >= 24.5:
                    d.text((560, 465), "obstacle on the path - plan blocked", font=font("b", 12), fill=(235, 90, 70), anchor="mm")
                # chip + tool position
                green_at, tcp = slot, slot
                carrying = None
                if tl < 11.8:
                    tcp = slot
                elif tl < 17.5:
                    s = ease((tl - 11.8) / 5.7)
                    tcp = on(P1, s)
                    carrying = "green"
                elif tl < 19.0:
                    tcp, green_at = p5, p5
                elif tl < 24.0:
                    green_at = p5
                    tcp = on(P2, ease((tl - 19.0) / 5.0))
                elif tl < 28.0:
                    green_at, tcp = p5, redA
                elif tl < 35.0:
                    green_at = p5
                    tcp = on(P3b, ease((tl - 28.0) / 7.0))
                    carrying = "red"
                else:
                    green_at, tcp, red_pos = p5, redB, redB
                if tl >= 17.5 and carrying != "green":
                    green_at = p5
                if carrying == "red":
                    red_pos = None
                if tl < 11.8:
                    green_at = slot
                if carrying != "green":
                    chip(green_at, (70, 200, 100))
                if red_pos is not None:
                    chip(red_pos, (235, 70, 60))
                if carrying:
                    px, py = pj((tcp[0], tcp[1], tcp[2]))
                    d.ellipse((px - 12, py + 6, px + 12, py + 18), fill=(70, 200, 100) if carrying == "green" else (235, 70, 60))
                gx, gy = pj((tcp[0], tcp[1], bz))
                px, py = pj(tcp)
                d.line((px, py, gx, gy), fill=(255, 255, 255, 70), width=1)
                d.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(255, 90, 70))
                d.text((352, 490), {0: "pick from tray", 1: "place on cell p5", 2: "go to the red chip", 3: "relocate the red chip"}[0 if tl < 11.8 else 1 if tl < 19 else 2 if tl < 24 else 3], font=font("m", 12), fill=TXT)
            # ---- learning panel
            px0, py0, pw, ph = 786, 160, 248, 110
            d.rectangle((px0, py0, px0 + pw, py0 + ph), fill=PANEL2, outline=LINE)
            d.text((px0 + 4, py0 - 12), "lift (mm) vs obstacle height (mm)", font=font("r", 10), fill=DIM)
            kk = min(len(data), int(tl * 2.2) + 1)
            for h_, lf in data[:kk]:
                x = px0 + (h_ - 0.07) / 0.06 * pw
                y = py0 + ph - (lf / 0.14) * ph
                d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=ACC3)
            if tl > 12:
                pts = [(px0 + i / 30 * pw, py0 + ph - learner.predict_lift(0.2, 0.07 + 0.06 * i / 30) / 0.14 * ph) for i in range(31)]
                d.line(pts, fill=ACC, width=2)
                d.text((px0 + pw - 4, py0 + 6), "ridge regression", font=font("b", 11), fill=ACC, anchor="ra")
            bullets(d, 786, 292, 250, ["Trial and error finds the shortest collision-free lift", "A regression predicts it for every new transfer",
                                       "New obstacle: lateral detour search, then continue", "Same idea: grasp-success prediction, anomaly detection for moved chips, per-lamp colour models"],
                    tl, 1.0, step=3.5, size=13, gap=4)
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# -------------------------------------------------------------------- timeline
R.SCENES = [("hero", 0, 6, "", ""), ("challenge", 6, 22, "01", "THE CHALLENGE"), ("color", 22, 40, "02", ""), ("bgsub", 40, 70, "03", ""),
            ("calib", 70, 90, "04", ""), ("seg", 90, 110, "05", ""), ("depth", 110, 128, "06", ""), ("motion", 128, 164, "07", ""),
            ("taught", 164, 176, "08", ""), ("standby", 176, 190, "09", ""), ("brain", 190, 200, "10", ""), ("design", 200, 208, "11", ""),
            ("end", 208, 216, "", "")]
R.TOTAL = 216.0
R.BUILD.update(hero=make_hero3, bgsub=make_bgsub3, depth=make_depth, motion=make_motion3)

if __name__ == "__main__":
    if sys.argv[1] == "test":
        R.main(None, 0, 0, only=[int(float(x) * FPS) for x in sys.argv[2:]])
    else:
        R.main(sys.argv[4], int(sys.argv[2]), int(sys.argv[3]))
