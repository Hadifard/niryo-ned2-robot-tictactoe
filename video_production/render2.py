"""
File: video_production/render2.py
Description: See the code below; this file has no module description yet.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""

import sys, math, subprocess, time
sys.path.insert(0, "/home/claude/niryo_tictactoe"); sys.path.insert(0, "/home/claude/film")
import cv2, numpy as np
from PIL import Image, ImageDraw
from lib import *
import common.game_logic as gl
from common.trajectory import xy_to_polar, polar_to_xy, arc_waypoints
from tests.synth import Scene as SynthScene
from v2_two_circle_calibration.calibration import BoardLayout
from v2_two_circle_calibration.vision_calibrated import CalibratedVision
from v3_robust_background_subtraction.vision_robust import RobustVision
from segmentation.classical_segmentation import segment_chips, overlay as seg_overlay
import overlay_real

ROOT = "/home/claude/niryo_tictactoe/"
CL = "/home/claude/film/clips/"
TOTAL = 180.0
LAYOUT = BoardLayout()
DARK, GRAY = (18, 26, 46), (95, 105, 128)
ACCD = (0, 150, 205)
BORDER = (40, 52, 84)
SUB_Y = 972
names = ("hero", "s1_lab", "s1_rob", "s2_v02", "s3_06", "bg_07", "s5_v03", "s6_rob", "s7_v11", "s8_v11", "s6_fin", "s7_sim", "s7_cad", "end_bar")
CLIPS = {k: Clip(CL + k + ".mp4") for k in names}
IMG09 = cv2.cvtColor(cv2.imread("/home/claude/s3/selected for making movie_03/09_Playing Ground Board Drawing 2.JPG"), cv2.COLOR_BGR2RGB)
IMG09 = cv2.resize(IMG09, (1800, int(1800 * IMG09.shape[0] / IMG09.shape[1])), interpolation=cv2.INTER_AREA)
IMG10 = cv2.cvtColor(cv2.imread("/home/claude/s3/selected for making movie_03/10_playing ground was made.jpg"), cv2.COLOR_BGR2RGB)
FACE = cv2.cvtColor(cv2.imread("/home/claude/film/face.png"), cv2.COLOR_BGR2RGB)

SCENES = [("hero", 0, 6, "", ""), ("challenge", 6, 22, "01", "THE CHALLENGE"), ("color", 22, 40, "02", "COLOUR DETECTION"),
          ("bgsub", 40, 64, "03", "BACKGROUND SUBTRACTION"), ("calib", 64, 84, "04", "TR & BL FIDUCIAL REFERENCES FOR CALIBRATION"),
          ("seg", 84, 104, "05", "SEGMENTATION"), ("motion", 104, 128, "06", "MOTION PLANNING · POLAR SPACE"),
          ("taught", 128, 140, "07", "TAUGHT POSITIONS · LOW LIGHT"), ("standby", 140, 154, "08", "STANDBY · OBSERVE · ACT"),
          ("brain", 154, 164, "09", "DECISION · MINIMAX"), ("design", 164, 172, "10", "DESIGN · SIMULATION · BUILD"),
          ("end", 172, 180, "", "")]


# ------------------------------------------------------------------ common bits
def light_bg():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = ((yy + xx) / (H + W))[..., None]
    img = np.array((236, 239, 244), np.float32) + (np.array((212, 217, 226), np.float32) - np.array((236, 239, 244), np.float32)) * t
    img[::40, ::40] -= 14
    return np.clip(img, 0, 255).astype(np.uint8)


def static_page(num, title, frames):
    im = Image.fromarray(light_bg())
    d = ImageDraw.Draw(im, "RGBA")
    if num:
        d.rectangle((30, 24, 110, 84), fill=ACC)
        d.text((70, 54), num, font=font("b", 34), fill=(8, 12, 22), anchor="mm")
        d.text((128, 54), title, font=font("b", 36 if len(title) < 26 else 30 if len(title) < 34 else 23), fill=DARK, anchor="lm")
        if len(title) < 26:
            d.text((1050, 54), "Computer vision · Robotics · Python", font=font("r", 17), fill=GRAY, anchor="rm")
    d.rectangle((0, SUB_Y, W, H), fill=(6, 9, 18))
    d.line((0, SUB_Y, W, SUB_Y), fill=ACC, width=2)
    for (x, y, w, h) in frames:
        d.rectangle((x - 2, y - 2, x + w + 1, y + h + 1), outline=BORDER, width=2)
    return im, d


def panel_static(d, x, y, w, h, title=None, color=ACC):
    d.rectangle((x, y, x + w, y + h), fill=PANEL, outline=LINE, width=2)
    if title:
        d.text((x + 16, y + 22), title, font=font("b", 20), fill=color, anchor="lm")
        d.line((x + 16, y + 42, x + w - 16, y + 42), fill=LINE, width=1)


def tag(d, x, y, s, color=ACC):
    f = font("b", 15)
    d.rectangle((x, y, x + int(f.getlength(s)) + 30, y + 26), fill=(8, 12, 22, 215))
    d.rectangle((x + 8, y + 9, x + 16, y + 17), fill=color)
    d.text((x + 22, y + 13), s, font=f, fill=TXT, anchor="lm")


def bullets(d, x, y, w, items, tl, t0, step=1.0, size=18, color=TXT, gap=8):
    f = font("r", size)
    for k, s in enumerate(items):
        a = ease_out((tl - (t0 + k * step)) / 0.5)
        lines = wrap(s, f, w - 24)
        if a > 0:
            off = int((1 - a) * 12)
            d.rectangle((x, y + 8 + off, x + 8, y + 16 + off), fill=(ACC[0], ACC[1], ACC[2], int(255 * a)))
            for j, ln in enumerate(lines):
                d.text((x + 20, y + off + j * (size + 5)), ln, font=f, fill=(color[0], color[1], color[2], int(255 * a)))
        y += len(lines) * (size + 5) + gap


def code_head(d, panel_xy, title, ln, n):
    x, y = panel_xy
    d.text((x + 100, y + 17), title, font=font("m", 15), fill=DIM, anchor="lm")
    d.text((x + 1020 - 16, y + 17), f"line {ln}/{n}", font=font("m", 14), fill=DIM, anchor="rm")


def std_scene(num, title, frames, panels=()):
    im, d = static_page(num, title, frames)
    for p in panels:
        panel_static(d, *p[:4], *p[4:])
    return np.asarray(im).copy()


# ---------------------------------------------------------------------- hero
def make_hero():
    im, d = static_page("", "", [(30, 400, 1020, 560)])
    bg = np.asarray(im).copy()

    def draw(canvas, tl, pre):
        f = CLIPS["hero"].frame(tl)
        reveal = int(1020 * ease_out((tl - 0.1) / 0.9))
        if reveal > 0:
            canvas[400:960, 30:30 + reveal] = f[:, :reveal]

        def ov(d):
            for i, (s, col) in enumerate((("SEE. DECIDE.", DARK), ("MOVE.", ACCD))):
                a = ease_out((tl - 0.2 - i * 0.3) / 0.7)
                d.text((34 - int((1 - a) * 80), 170 + i * 112), s, font=font("b", 108), fill=(col[0], col[1], col[2], int(255 * a)), anchor="ls")
            a = ease_out((tl - 1.2) / 0.7)
            d.text((38, 356), "Computer vision and robotics", font=font("r", 38), fill=(GRAY[0], GRAY[1], GRAY[2], int(255 * a)), anchor="ls")
            d.rectangle((38, 44, 38 + int(160 * a), 52), fill=ACC)
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# ----------------------------------------------------------------- challenge
def icon(d, kind, cx, cy, tl, col):
    if kind == 0:
        w_ = math.sin(tl * 6) * 7
        d.ellipse((cx - 30, cy + 18, cx + 30, cy + 42), outline=col, width=4)
        d.line((cx, cy + 30, cx + w_, cy - 10), fill=col, width=7)
        d.line((cx + w_, cy - 10, cx + w_ * 2.2 + 26, cy - 28), fill=col, width=7)
    elif kind == 1:
        r = 20 + 4 * math.sin(tl * 4)
        d.ellipse((cx - r, cy - r - 6, cx + r, cy + r - 6), outline=col, width=4)
        for k in range(8):
            a = k * math.pi / 4 + tl * 0.8
            d.line((cx + math.cos(a) * (r + 8), cy - 6 + math.sin(a) * (r + 8), cx + math.cos(a) * (r + 18), cy - 6 + math.sin(a) * (r + 18)), fill=col, width=3)
    elif kind == 2:
        for i in range(5):
            h = 10 + 18 * abs(math.sin(tl * 3 + i * 1.1))
            d.rectangle((cx - 40 + i * 17, cy + 28 - h, cx - 28 + i * 17, cy + 28), fill=col)
    else:
        d.rectangle((cx - 26, cy - 26, cx + 26, cy + 26), outline=col, width=4)
        d.rectangle((cx - 12, cy - 12, cx + 12, cy + 12), fill=col)
        for k in range(-1, 2):
            d.line((cx + k * 16, cy - 36, cx + k * 16, cy - 26), fill=col, width=3)
            d.line((cx + k * 16, cy + 26, cx + k * 16, cy + 36), fill=col, width=3)


CARDS = [
    ("Actuation error", "Gear backlash and limited repeatability cause millimetre-scale pose errors that vision must correct.", ACC2),
    ("Illumination variance", "Daylight and lamps change intensity and hue; fixed colour thresholds are not photometrically invariant.", ACC),
    ("Low-grade sensor", "A noisy, compressed, lens-distorted wrist camera: undistortion and temporal voting are required.", GOOD),
    ("Compute budget", "Three-year-old deep detectors were heavy and weak for this scene and cut the control rate on a laptop CPU.", ACC3),
]


def make_challenge():
    bg = std_scene("01", "THE CHALLENGE", [(30, 108, 711, 400), (753, 108, 297, 400)])

    def draw(canvas, tl, pre):
        paste(canvas, CLIPS["s1_lab"].frame(tl), 30, 108, 711, 400)
        paste(canvas, CLIPS["s1_rob"].frame(tl), 753, 108, 297, 400)

        def ov(d):
            tag(d, 46, 124, "LAB FOOTAGE · LAPTOP CODE")
            tag(d, 769, 124, "ROBOT PICK & PLACE", ACC2)
            pos = [(30, 524), (546, 524), (30, 747), (546, 747)]
            for k, (title, text, col) in enumerate(CARDS):
                a = ease_out((tl - (0.8 + k * 3.2)) / 0.6)
                if a <= 0:
                    continue
                x, y = pos[k]
                y += int((1 - a) * 36)
                al = int(255 * a)
                d.rectangle((x, y, x + 504, y + 211), fill=(PANEL[0], PANEL[1], PANEL[2], al), outline=(col[0], col[1], col[2], al), width=2)
                icon(d, k, x + 78, y + 105, tl, (col[0], col[1], col[2], al))
                d.text((x + 150, y + 36), title, font=font("b", 26), fill=(TXT[0], TXT[1], TXT[2], al))
                for j, ln in enumerate(wrap(text, font("r", 19), 335)):
                    d.text((x + 150, y + 82 + j * 26), ln, font=font("r", 19), fill=(DIM[0], DIM[1], DIM[2], al))
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# -------------------------------------------------------------------- colour
HUE = [("red", 0, 12, 3, (235, 70, 60)), ("green", 38, 85, 68, (70, 200, 90)),
       ("blue", 100, 130, 110, (70, 120, 255)), ("pink", 145, 179, 174, (255, 90, 190))]


def make_color():
    p0 = span(ROOT + "common/colors.py", r"^HSV_RANGES", r"^}")
    p1 = block(ROOT + "common/colors.py", r"^def color_mask")
    p2 = block(ROOT + "common/colors.py", r"^def classify_disc")
    panel = CodePanel(30, 524, 1020, 442, [dict(title="common/colors.py", first=p0[0], text=p0[1] + "\n\n\n" + p1[1], t0=0.4, t1=6.4),
                                         dict(title="common/colors.py", first=p2[0], text=p2[1], t0=6.8, t1=16.2)])
    bg = std_scene("02", "COLOUR DETECTION", [(30, 108, 532, 400)], [(574, 108, 476, 400, "HSV model · measured on real chips", ACC)])

    def draw(canvas, tl, pre):
        paste(canvas, CLIPS["s2_v02"].frame(tl), 30, 108, 532, 400)
        title, ln, n = panel.draw(canvas, tl, None)

        def ov(d):
            tag(d, 46, 124, "COLOUR DETECTION TEST · LAPTOP CAMERA")
            code_head(d, (30, 524), title, ln, n)
            rx, ry, rw = 598, 160, 428
            for i in range(rw):
                c = cv2.cvtColor(np.uint8([[[int(i / rw * 179), 230, 240]]]), cv2.COLOR_HSV2RGB)[0, 0]
                d.line((rx + i, ry, rx + i, ry + 12), fill=(int(c[0]), int(c[1]), int(c[2])))
            d.text((rx, ry + 16), "0", font=font("m", 12), fill=DIM)
            d.text((rx + rw, ry + 16), "179 (OpenCV hue)", font=font("m", 12), fill=DIM, anchor="ra")
            for k, (nm, lo, hi, meas, col) in enumerate(HUE):
                a = ease_out((tl - (0.8 + k * 0.9)) / 0.6)
                if a <= 0:
                    continue
                y = 204 + k * 50
                d.text((rx, y - 2), nm, font=font("b", 18), fill=(col[0], col[1], col[2], int(255 * a)))
                d.text((rx + rw, y - 2), f"H {lo}-{hi}   measured {meas}", font=font("m", 14), fill=(TXT[0], TXT[1], TXT[2], int(255 * a)), anchor="ra")
                x0, x1 = rx + lo / 179 * rw, rx + hi / 179 * rw
                d.rectangle((x0, y + 20, x1, y + 30), fill=(col[0], col[1], col[2], int(255 * a)))
                mx = rx + meas / 179 * rw
                d.polygon([(mx, y + 33), (mx - 5, y + 41), (mx + 5, y + 41)], fill=(255, 255, 255, int(255 * a)))
            bullets(d, 598, 412, 430, ["Value channel absorbs most brightness change; hue and saturation carry the class",
                                       "Circular ROI vote per cell, white-patch balance, CLAHE"], tl, 5.0, step=2.0, size=15, gap=4)
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# ------------------------------------------------------------------ bg subtraction
def make_bgsub():
    f1 = block(ROOT + "v3_robust_background_subtraction/vision_robust.py", r"def learn_background")
    f2 = block(ROOT + "v3_robust_background_subtraction/vision_robust.py", r"def foreground")
    panel = CodePanel(30, 524, 1020, 442, [dict(title="v3_robust_background_subtraction/vision_robust.py", first=f1[0], text=f1[1], t0=0.4, t1=8.0),
                                         dict(title="v3_robust_background_subtraction/vision_robust.py", first=f2[0], text=f2[1], t0=8.6, t1=21.0)])
    bg = std_scene("03", "BACKGROUND SUBTRACTION", [(30, 108, 224, 400), (267, 108, 400, 400)], [(679, 108, 371, 400, "Idea: subtract frames", ACC)])
    empty = SynthScene(LAYOUT, 18, (8, -6), seed=1)
    empty.cells, empty.slots, empty.token = ["none"] * 9, {n: "none" for n in LAYOUT.slots}, None
    vision = RobustVision(LAYOUT)
    vision.learn_background([empty.render() for _ in range(5)])
    script = [(1.5, 4, "blue"), (4.0, 0, "green"), (6.5, 8, "blue"), (9.0, 2, "green"), (11.5, 6, "blue"), (14.0, 1, "green"),
              (16.5, 5, "blue"), (19.0, 7, "green")]
    st = dict(obs=None)

    def draw(canvas, tl, pre):
        paste(canvas, CLIPS["bg_07"].frame(tl), 30, 108, 224, 400)
        g = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(0.45 * tl))
        sc = SynthScene(LAYOUT, 18, (8, -6), gain=(g * 1.05, g, g * 0.95), shadow=0.38 * (0.5 + 0.5 * math.sin(0.3 * tl + 1)), seed=int(tl * 30) % 997)
        sc.cells = ["none"] * 9
        for (t0, c, col) in script:
            if tl >= t0:
                sc.cells[c] = col
        sc.slots, sc.token = {n: "none" for n in LAYOUT.slots}, None
        frame = sc.render()
        try:
            st["obs"] = vision.observe(frame)
        except Exception:
            pass
        obs = st["obs"]
        paste(canvas, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 267, 108, 400, 200, fy=0.5)
        mask = np.zeros(obs.foreground.shape + (3,), np.uint8) + np.array((14, 18, 30), np.uint8)
        mask[obs.foreground > 0] = ACC
        paste(canvas, mask, 267, 308, 400, 200)
        title, ln, n = panel.draw(canvas, tl, None)

        def ov(d):
            tag(d, 40, 124, "REAL FOOTAGE", ACC2)
            tag(d, 281, 124, f"CURRENT FRAME · GAIN {g:.2f}", ACC)
            tag(d, 281, 324, "DIFFERENCE MASK · LAB ΔE", GOOD)
            code_head(d, (30, 524), title, ln, n)
            ok = all((obs.cells[i] == sc.cells[i]) for i in range(9))
            cols = {"none": (60, 70, 96), "blue": (70, 120, 255), "green": (70, 200, 90), "red": (235, 70, 60)}
            for i in range(9):
                r_, c_ = divmod(i, 3)
                x, y = 587 + c_ * 22, 436 + r_ * 22
                d.rectangle((x, y, x + 18, y + 18), fill=cols[obs.cells[i]])
            d.text((577, 470), "9/9 cells correct" if ok else "reading ...", font=font("b", 13), fill=GOOD if ok else DIM, anchor="rm")
            bullets(d, 695, 156, 340, ["Subtract consecutive frames: only the changed chips remain",
                                       "Works in dim or shadowed regions where colour is weak",
                                       "Blob centroid = chip position and the pick target"], tl, 1.5, step=3.0, size=15, gap=5)
            a = ease_out((tl - 11.0) / 0.8)
            d.rectangle((695, 360, 1034, 496), fill=(PANEL2[0], PANEL2[1], PANEL2[2], int(255 * a)), outline=(ACC[0], ACC[1], ACC[2], int(255 * a)), width=1)
            d.text((705, 368), "REFERENCE", font=font("b", 12), fill=(ACC[0], ACC[1], ACC[2], int(255 * a)))
            cite = "H. Sarhangi Fard (2022). Improving Moving Vehicles Detection and Vehicles Trajectory Plotting using Optimized Background Subtraction Method. Expert Systems with Applications."
            for j, ln_ in enumerate(wrap(cite, font("r", 13), 320)):
                d.text((705, 386 + j * 17), ln_, font=font("r", 13), fill=(TXT[0], TXT[1], TXT[2], int(255 * a)))
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# --------------------------------------------------------------------- calibration
SYN_SCRIPT = [(1.0, 4, "blue"), (3.0, 0, "green"), (5.0, 8, "blue"), (7.0, 2, "green"), (9.0, 6, "blue"), (11.0, 1, "green"),
              (13.0, 5, "blue"), (15.0, 7, "green")]


def make_calib():
    f1 = span(ROOT + "v2_two_circle_calibration/calibration.py", r"^@dataclass\(frozen=True\)", r"return \(z\.real, z\.imag\)")
    f2 = block(ROOT + "v2_two_circle_calibration/calibration.py", r"^def find_circle")
    f3 = block(ROOT + "v2_two_circle_calibration/calibration.py", r"^def locate_paper")
    T = "v2_two_circle_calibration/calibration.py"
    panel = CodePanel(30, 524, 1020, 442, [dict(title=T, first=f1[0], text=f1[1], t0=0.3, t1=7.0), dict(title=T, first=f2[0], text=f2[1], t0=7.4, t1=13.6),
                                         dict(title=T, first=f3[0], text=f3[1], t0=14.0, t1=18.6)])
    bg = std_scene("04", "TR & BL FIDUCIAL REFERENCES FOR CALIBRATION", [(30, 108, 225, 400), (267, 108, 400, 400)], [(679, 108, 371, 400, "p' = m·p + t   (4 DoF)", ACC)])
    st = dict(prev=None, vision=CalibratedVision(LAYOUT), obs=None)

    def draw(canvas, tl, pre):
        src_t = (tl * 0.45) % 8.5
        bgr = cv2.cvtColor(CLIPS["s3_06"].frame(tl % 19.0), cv2.COLOR_RGB2BGR)
        out, info, prev = overlay_real.process(bgr, st["prev"], draw_cells=(4.3 <= src_t <= 5.7))
        st["prev"] = prev
        paste(canvas, cv2.cvtColor(out, cv2.COLOR_BGR2RGB), 30, 108, 225, 400)
        sc = SynthScene(LAYOUT, rot_deg=(tl * 19.0) % 360, shift_px=(int(22 * math.sin(tl * 0.8)), int(16 * math.cos(tl * 0.55))), px_per_mm=1.15, seed=int(tl * 30) % 1000)
        sc.cells = ["none"] * 9
        for (t0, c, col) in SYN_SCRIPT:
            if tl >= t0:
                sc.cells[c] = col
        sc.slots, sc.token = {n: "none" for n in LAYOUT.slots}, ("A" if int(tl / 4) % 2 == 0 else "B")
        frame = sc.render()
        try:
            obs = st["vision"].observe(frame)
            frame = st["vision"].draw(frame, obs, label=False)
            st["obs"] = obs
        except Exception:
            pass
        paste(canvas, cv2.cvtColor(frame[:, 80:560], cv2.COLOR_BGR2RGB), 267, 108, 400, 400)
        title, ln, n = panel.draw(canvas, tl, None)
        obs = st["obs"]

        def ov(d):
            tag(d, 44, 124, "REAL FOOTAGE", ACC2)
            tag(d, 281, 124, "ANY ROTATION · ANY POSITION")
            code_head(d, (30, 524), title, ln, n)
            if obs is not None:
                d.text((697, 160), f"rotation  {((obs.angle_deg + 180) % 360) - 180:7.1f} deg", font=font("mb", 20), fill=ACC)
                d.text((697, 190), f"scale     {obs.px_per_mm:7.2f} px/mm", font=font("mb", 20), fill=GOOD)
            bullets(d, 697, 232, 336, ["m = (b − a) / (B − A): closed form from two circles",
                                       "Grid, trays and token spots follow the paper",
                                       "Hand-eye step: pixel → robot XY (camera looks down)"], tl, 2.0, step=3.5, size=16, gap=6)
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# --------------------------------------------------------------------- segmentation
def make_seg():
    f2 = block(ROOT + "segmentation/classical_segmentation.py", r"^def segment_color")
    f3 = block(ROOT + "segmentation/yolo_seg_demo.py", r"^def main")
    panel = CodePanel(30, 524, 1020, 442, [dict(title="segmentation/classical_segmentation.py", first=f2[0], text=f2[1], t0=0.4, t1=10.0),
                                         dict(title="segmentation/yolo_seg_demo.py  (deep model baseline)", first=f3[0], text="\n".join(f3[1].split("\n")[:30]), t0=10.6, t1=17.4)])
    bg = std_scene("05", "SEGMENTATION", [(30, 108, 532, 400), (574, 108, 225, 400)], [(811, 108, 239, 400, "Why classical", GOOD)])

    def draw(canvas, tl, pre):
        paste(canvas, CLIPS["s5_v03"].frame(tl), 30, 108, 532, 400)
        bgr = cv2.cvtColor(CLIPS["s3_06"].frame((tl * 0.9 + 1.0) % 19.0), cv2.COLOR_RGB2BGR)
        out = seg_overlay(bgr, segment_chips(bgr, min_area=450, max_area=3200, min_circularity=0.72))
        paste(canvas, cv2.cvtColor(out, cv2.COLOR_BGR2RGB), 574, 108, 225, 400)
        title, ln, n = panel.draw(canvas, tl, None)

        def ov(d):
            tag(d, 46, 124, "SEGMENTATION TEST · LAPTOP CAMERA · DEEP MODEL", ACC3)
            tag(d, 588, 124, "CLASSICAL · THIS CODE", GOOD)
            code_head(d, (30, 524), title, ln, n)
            a = ease_out((tl - 1.0) / 0.6)
            d.text((827, 160), "HSV + watershed", font=font("b", 15), fill=(GOOD[0], GOOD[1], GOOD[2], int(255 * a)))
            d.rectangle((827, 182, 827 + int(40 * a), 194), fill=(GOOD[0], GOOD[1], GOOD[2], int(255 * a)))
            d.text((827, 200), "≈60 ms/frame, 1 CPU core", font=font("r", 13), fill=(TXT[0], TXT[1], TXT[2], int(255 * a)))
            a = ease_out((tl - 2.2) / 0.6)
            d.text((827, 232), "COCO-trained network", font=font("b", 15), fill=(ACC3[0], ACC3[1], ACC3[2], int(255 * a)))
            d.rectangle((827, 254, 827 + int(190 * a), 266), fill=(ACC3[0], ACC3[1], ACC3[2], int(255 * a)))
            d.text((827, 272), "heavy, GPU for real time", font=font("r", 13), fill=(TXT[0], TXT[1], TXT[2], int(255 * a)))
            bullets(d, 827, 306, 206, ["Class prior mismatch: knows chairs and animals, not our chips",
                                       "Distance-transform peaks split touching chips"], tl, 3.6, step=3.0, size=14, gap=4)
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# ------------------------------------------------------------------- motion planning
BASE_XY = None


def make_motion():
    t1 = block(ROOT + "common/trajectory.py", r"^def arc_waypoints")
    t2a = block(ROOT + "common/robot_control.py", r"    def _descend")
    t2b = block(ROOT + "common/robot_control.py", r"    def travel")
    t2c = block(ROOT + "common/robot_control.py", r"    def pick_and_place")
    panel = CodePanel(30, 524, 1020, 442, [dict(title="common/trajectory.py", first=t1[0], text=t1[1], t0=0.4, t1=9.0),
                                         dict(title="common/robot_control.py", first=t2a[0], text=t2a[1] + "\n\n" + t2b[1] + "\n\n" + t2c[1], t0=9.6, t1=20.0)])
    bg = std_scene("06", "MOTION PLANNING · POLAR SPACE", [(30, 108, 296, 400)],
                   [(338, 108, 336, 400, "Polar space (r, θ, z)", ACC), (686, 108, 364, 400, "Arc + vertical approach", ACC2)])
    P1, P2 = (0.17, 0.12), (0.33, -0.09)
    ZP, HOV, LIFT = 0.05, 0.03, 0.05
    S = 600.0

    def tcp(tc):
        """position (x, y, z, phase) for cycle phase tc in 0..1"""
        if tc < 0.12:
            return P1[0], P1[1], ZP + HOV * ease(tc / 0.12), "vertical lift"
        if tc < 0.80:
            s = ease((tc - 0.12) / 0.68)
            x, y, z = arc_waypoints(P1, P2, ZP + HOV, ZP + HOV + 0.01, LIFT, 1)[0] if False else (None, None, None)
            r0, th0 = xy_to_polar(*P1); r1, th1 = xy_to_polar(*P2)
            x, y = polar_to_xy(r0 + (r1 - r0) * s, th0 + (th1 - th0) * s)
            return x, y, ZP + HOV + 0.01 * s + 4 * LIFT * s * (1 - s), "parabolic arc in (r, θ)"
        if tc < 0.92:
            return P2[0], P2[1], ZP + HOV + 0.01 - (HOV + 0.01 - 0.01) * ease((tc - 0.80) / 0.12), "vertical descent ⟂ surface"
        return P2[0], P2[1], ZP + 0.01, "release"

    def draw(canvas, tl, pre):
        paste(canvas, CLIPS["s6_rob"].frame(tl), 30, 108, 296, 400)
        title, ln, n = panel.draw(canvas, tl, None)
        tc = (tl % 7.0) / 7.0
        x, y, z, phase = tcp(tc)

        def ov(d):
            tag(d, 46, 124, "ROBOT PICK & PLACE", ACC2)
            code_head(d, (30, 524), title, ln, n)
            # ---- polar top view
            bx, by = 338 + 168, 108 + 372
            P = lambda xy: (bx - xy[1] * S, by - xy[0] * S)
            for r in (0.1, 0.2, 0.3, 0.4):
                d.arc((bx - r * S, by - r * S, bx + r * S, by + r * S), 215, 325, fill=LINE, width=1)
                d.text((bx + 4, by - r * S - 14), f"{r:.1f} m", font=font("m", 11), fill=(90, 100, 128))
            for deg in (-30, -15, 0, 15, 30):
                a = math.radians(deg)
                d.line((bx, by, bx - math.sin(a) * 0.42 * S, by - math.cos(a) * 0.42 * S), fill=(40, 50, 76), width=1)
            d.ellipse((bx - 12, by - 12, bx + 12, by + 12), fill=(70, 80, 110), outline=ACC, width=2)
            d.text((bx, by + 22), "robot base", font=font("r", 12), fill=DIM, anchor="mm")
            r0, th0 = xy_to_polar(*P1); r1, th1 = xy_to_polar(*P2)
            pts = [P((r0 + (r1 - r0) * s) * 0 + polar_to_xy(r0 + (r1 - r0) * s, th0 + (th1 - th0) * s)[0],
                     polar_to_xy(r0 + (r1 - r0) * s, th0 + (th1 - th0) * s)[1]) if False else P(polar_to_xy(r0 + (r1 - r0) * s, th0 + (th1 - th0) * s)) for s in np.linspace(0, 1, 40)]
            d.line(pts, fill=(ACC[0], ACC[1], ACC[2], 150), width=3)
            for xy, lab in ((P1, "pick"), (P2, "place")):
                px, py = P(xy)
                d.ellipse((px - 9, py - 9, px + 9, py + 9), outline=(255, 255, 255), width=2)
                d.text((px + 12, py - 8), lab, font=font("r", 13), fill=TXT)
            cx, cy = P((x, y))
            d.line((bx, by, cx, cy), fill=(255, 200, 80, 160), width=1)
            d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(255, 90, 70))
            r_, th_ = xy_to_polar(x, y)
            d.text((350, 160), f"r = {r_:.3f} m", font=font("mb", 15), fill=ACC)
            d.text((350, 182), f"θ = {math.degrees(th_):6.1f}°", font=font("mb", 15), fill=ACC)
            d.text((350, 204), "joint 1 sweeps θ", font=font("r", 12), fill=DIM)
            d.text((350, 220), "joints 2-3 set r and z", font=font("r", 12), fill=DIM)
            # ---- side view  (z vs path)
            ox, oy = 686, 108
            def Z(zv):
                return oy + 336 - (zv - 0.03) * 2300
            xa, xb = ox + 60, ox + 312
            d.line((ox + 14, Z(0.04), ox + 352, Z(0.04)), fill=(120, 130, 160), width=2)
            d.text((ox + 20, Z(0.04) + 6), "board surface", font=font("r", 11), fill=DIM)
            for cxp in (xa, xa + 63, xa + 126, xa + 189, xb):
                d.rectangle((cxp - 15, Z(0.052), cxp + 15, Z(0.04)), fill=(70, 130, 255) if cxp in (xa, xb) else (90, 100, 130))
            path = [(xa, Z(ZP)), (xa, Z(ZP + HOV))]
            for s in np.linspace(0, 1, 40):
                path.append((xa + (xb - xa) * s, Z(ZP + HOV + 0.01 * s + 4 * LIFT * s * (1 - s))))
            path += [(xb, Z(ZP + HOV + 0.01)), (xb, Z(0.06))]
            d.line(path, fill=(ACC2[0], ACC2[1], ACC2[2], 200), width=3)
            d.line((ox + 14, Z(0.052 + 0.02), ox + 352, Z(0.052 + 0.02)), fill=(255, 255, 255, 60), width=1)
            d.text((ox + 352, Z(0.072) - 6), "clearance above chips", font=font("r", 11), fill=DIM, anchor="rm")
            # tcp x position along the path
            if tc < 0.12:
                tx = xa
            elif tc < 0.80:
                tx = xa + (xb - xa) * ease((tc - 0.12) / 0.68)
            else:
                tx = xb
            ty = Z(z)
            d.polygon([(tx - 12, ty - 22), (tx + 12, ty - 22), (tx + 7, ty - 4), (tx - 7, ty - 4)], fill=(255, 90, 70))
            d.rectangle((tx - 3, ty - 46, tx + 3, ty - 22), fill=(200, 205, 220))
            d.text((ox + 182, oy + 68), phase, font=font("b", 14), fill=(255, 200, 80), anchor="mm")
            d.text((ox + 182, oy + 90), f"z = {z * 1000:5.0f} mm   pitch = 90°", font=font("m", 12), fill=TXT, anchor="mm")
            d.text((ox + 182, oy + 378), "z(s) = z0 + (z1−z0)s + 4·h·s(1−s)", font=font("m", 13), fill=ACC, anchor="mm")
            if tc >= 0.80:
                d.line((xb + 22, Z(0.052), xb + 22, Z(0.06)), fill=(255, 200, 80), width=1)
                d.text((xb - 6, Z(0.095)), "⟂", font=font("b", 20), fill=(255, 200, 80), anchor="mm")
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# --------------------------------------------------------------------- taught
def make_taught():
    snippet = ("pose_read = robot.arm.get_pose()\n"
               "print(f'{round(pose_read.x,4), round(pose_read.y,4), round(pose_read.z,4), round(pose_read.roll,4), round(pose_read.pitch,4), round(pose_read.yaw,4)} ')\n"
               "\n"
               "# learning mode: guide the arm by hand over each cell, read the pose, store it\n"
               "robot.arm.set_learning_mode(True)\n"
               "position_coordinates = get_pickup_coordinates(position)   # table lookup\n"
               "robot.arm.move_pose(position_coordinates)                 # replay the taught pose")
    import importlib.util
    spec = importlib.util.spec_from_file_location("rp", ROOT + "v1_fixed_layout/recorded_positions.py")
    rpm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rpm)
    rec = "pickup_coordinates = {\n" + "\n".join(f"    '{k}': {v}," for k, v in list(rpm.pickup_coordinates.items())[:9]) + "\n    ...\n}"
    panel = CodePanel(30, 524, 1020, 442, [dict(title="movement.ipynb  (teach by hand)", first=1, text=snippet, t0=0.3, t1=5.0),
                                         dict(title="v1_fixed_layout/recorded_positions.py", first=6, text=rec, t0=5.4, t1=10.6)])
    bg = std_scene("08", "TAUGHT POSITIONS · LOW LIGHT", [(30, 108, 532, 400)], [(574, 108, 476, 400, "When detection becomes unreliable", ACC2)])

    def draw(canvas, tl, pre):
        paste(canvas, CLIPS["s7_v11"].frame(tl), 30, 108, 532, 400)
        title, ln, n = panel.draw(canvas, tl, None)

        def ov(d):
            tag(d, 46, 124, "LOW-LIGHT TEST WITH TWO ROBOTS", ACC2)
            code_head(d, (30, 524), title, ln, n)
            bullets(d, 592, 156, 440, ["Dim room: colour contrast collapses and detection errors rise",
                                       "Each pick/place pose is taught once in learning mode and read with get_pose()",
                                       "The table (x, y, z, roll, pitch, yaw) replaces vision for positions",
                                       "Vision still verifies the board; the table only supplies where to move"], tl, 0.8, step=2.4, size=17, gap=10)
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# --------------------------------------------------------------------- standby
STATES = [("REST", "folded pose: the other arm cannot hit it"), ("WATCH", "camera sees only the red-token region"),
          ("OBSERVE", "fixed observation point, same view every time"), ("PERCEIVE", "colour + background subtraction, diff check"),
          ("DECIDE", "alpha-beta minimax picks the move"), ("ACT", "polar arc, vertical descent, pick and place")]


def make_standby():
    g1 = block(ROOT + "common/game_runner.py", r"    def _standby_until_token")
    g2 = block(ROOT + "common/game_runner.py", r"    def my_move")
    panel = CodePanel(30, 524, 1020, 442, [dict(title="common/game_runner.py", first=g1[0], text=g1[1] + "\n\n" + g2[1], t0=0.4, t1=12.0)])
    bg = std_scene("09", "STANDBY · OBSERVE · ACT", [(30, 108, 532, 400)], [(574, 108, 476, 400, "Collision-free turn protocol", ACC)])

    def draw(canvas, tl, pre):
        paste(canvas, CLIPS["s8_v11"].frame(tl), 30, 108, 532, 400)
        title, ln, n = panel.draw(canvas, tl, None)
        active = int(tl / 1.9) % 6

        def ov(d):
            tag(d, 46, 124, "ROBOT VS ROBOT", ACC2)
            code_head(d, (30, 524), title, ln, n)
            for i, (nm, desc) in enumerate(STATES):
                y = 166 + i * 54
                on = i == active
                col = ACC if on else (70, 82, 112)
                d.rectangle((592, y, 700, y + 40), fill=(0, 70, 95) if on else PANEL2, outline=col, width=2)
                d.text((646, y + 20), nm, font=font("b", 15), fill=TXT if on else DIM, anchor="mm")
                d.text((712, y + 20), desc, font=font("r", 13), fill=TXT if on else DIM, anchor="lm")
                if i < 5:
                    d.polygon([(646, y + 44), (638, y + 52), (654, y + 52)], fill=col)
            if active == 1 and int(tl * 3) % 2 == 0:
                d.ellipse((1000, 140, 1024, 164), fill=(235, 70, 60))
            d.text((1012, 176), "token", font=font("r", 11), fill=DIM, anchor="mm")
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# ---------------------------------------------------------------------- brain
def minimax_demo():
    cnt, orig = {"n": 0}, gl._minimax

    def wrapped(*a, **k):
        cnt["n"] += 1
        return orig(*a, **k)
    gl._minimax = wrapped
    g, ag, states, turn = gl.TicTacToe(), gl.MinimaxAgent("O"), [], "X"
    human = [0, 8, 5, 3, 7, 1, 2, 6, 4]
    while not g.is_over():
        if turn == "X":
            mv = next(m for m in human if m in g.available_moves())
            g.make_move(mv, "X")
            states.append((mv, "X", 0))
        else:
            cnt["n"] = 0
            mv = ag.get_move(g)
            g.make_move(mv, "O")
            states.append((mv, "O", cnt["n"]))
        turn = "O" if turn == "X" else "X"
    gl._minimax = orig
    return states


def make_brain():
    states = minimax_demo()
    m1 = block(ROOT + "common/game_logic.py", r"^def _minimax")
    m2 = block(ROOT + "common/game_logic.py", r"^def diff_boards")
    panel = CodePanel(30, 524, 1020, 442, [dict(title="common/game_logic.py", first=m1[0], text=m1[1], t0=0.3, t1=4.8), dict(title="common/game_logic.py", first=m2[0], text=m2[1], t0=5.2, t1=9.0)])
    bg = std_scene("10", "DECISION · MINIMAX", [(30, 108, 711, 400)], [(753, 108, 297, 400)])

    def draw(canvas, tl, pre):
        paste(canvas, CLIPS["s6_fin"].frame(tl), 30, 108, 711, 400)
        title, ln, n = panel.draw(canvas, tl, None)

        def ov(d):
            tag(d, 46, 124, "ROBOT VS ROBOT · BEFORE THE EXPO", ACC2)
            code_head(d, (30, 524), title, ln, n)
            d.text((901, 134), "alpha-beta minimax", font=font("b", 19), fill=ACC, anchor="mm")
            ox, oy, cs = 783, 160, 78
            for i in range(1, 3):
                d.line((ox + i * cs, oy + 6, ox + i * cs, oy + 3 * cs - 6), fill=LINE, width=4)
                d.line((ox + 6, oy + i * cs, ox + 3 * cs - 6, oy + i * cs), fill=LINE, width=4)
            k = min(len(states), int((tl - 0.6) / 1.25) + 1) if tl > 0.6 else 0
            for j in range(k):
                mv, who, nodes = states[j]
                age = ease_out((tl - (0.6 + j * 1.25)) / 0.4)
                r_, c_ = divmod(mv, 3)
                cx, cy = ox + c_ * cs + cs / 2, oy + r_ * cs + cs / 2
                col = (70, 130, 255) if who == "X" else (70, 210, 110)
                d.ellipse((cx - 26 * age, cy - 26 * age, cx + 26 * age, cy + 26 * age), fill=col)
            if k > 0:
                mv, who, nodes = states[k - 1]
                d.text((901, 424), "opponent move verified" if who == "X" else f"robot searched {nodes:,} nodes", font=font("b", 16), fill=TXT, anchor="mm")
            if k >= len(states):
                d.line((ox + 1 * cs + cs / 2, oy + 12, ox + 1 * cs + cs / 2, oy + 3 * cs - 12), fill=(255, 220, 80), width=7)
                d.text((901, 456), "ROBOT WINS", font=font("b", 22), fill=(255, 220, 80), anchor="mm")
            else:
                d.text((901, 456), "never loses · prefers fast wins", font=font("r", 14), fill=DIM, anchor="mm")
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# --------------------------------------------------------------------- design
def make_design():
    bg = std_scene("11", "DESIGN · SIMULATION · BUILD", [(30, 108, 711, 400), (753, 108, 297, 400), (30, 524, 600, 442), (642, 524, 408, 442)])

    def draw(canvas, tl, pre):
        paste(canvas, CLIPS["s7_sim"].frame(tl), 30, 108, 711, 400)
        z = 1.0 + 0.05 * (tl / 8.0)
        h10, w10 = IMG10.shape[:2]
        ch, cw = int(h10 / z), int(w10 / z)
        paste(canvas, IMG10[(h10 - ch) // 2:(h10 - ch) // 2 + ch, (w10 - cw) // 2:(w10 - cw) // 2 + cw], 753, 108, 297, 400)
        h9, w9 = IMG09.shape[:2]
        zz = 1.0 + 0.18 * ease(tl / 8.0)
        ch, cw = int(h9 / zz), int(w9 / zz)
        paste(canvas, IMG09[int((h9 - ch) * 0.35):int((h9 - ch) * 0.35) + ch, int((w9 - cw) * (0.35 + 0.3 * ease(tl / 8.0))):int((w9 - cw) * (0.35 + 0.3 * ease(tl / 8.0))) + cw], 30, 524, 600, 442)
        paste(canvas, CLIPS["s7_cad"].frame(tl), 642, 524, 408, 442)

        def ov(d):
            tag(d, 46, 124, "SOLIDWORKS MOTION STUDY", ACC3)
            tag(d, 769, 124, "BUILT & PAINTED", ACC2)
            tag(d, 46, 540, "DRAWING 02 · ABS · 398.5 × 181.6 mm", GOOD)
            tag(d, 658, 540, "CAD DESIGN")
        pre.append(ov)
    return dict(bg=bg, draw=draw)


# ------------------------------------------------------------------------ end
def make_end():
    im, d = static_page("", "", [(30, 30, 1020, 300)])
    bg = np.asarray(im).copy()
    face = cv2.resize(FACE, (190, 190), interpolation=cv2.INTER_AREA)

    def draw(canvas, tl, pre):
        canvas[30:330, 30:1050] = CLIPS["end_bar"].frame(tl)
        a = ease_out((tl - 0.2) / 0.8)
        canvas[400:590, 30:220] = (face.astype(np.float32) * a + canvas[400:590, 30:220].astype(np.float32) * (1 - a)).astype(np.uint8)

        def ov(d):
            d.rectangle((28, 398, 222, 592), outline=BORDER, width=2)
            b = ease_out((tl - 0.9) / 0.7)
            d.text((250, 440), "Hadi Sarhangi Fard", font=font("b", 42), fill=(DARK[0], DARK[1], DARK[2], int(255 * b)), anchor="lm")
            b = ease_out((tl - 1.4) / 0.7)
            d.text((250, 500), "AI Engineer / Mechanical Engineer", font=font("r", 27), fill=(ACCD[0], ACCD[1], ACCD[2], int(255 * b)), anchor="lm")
            d.text((250, 546), "Computer vision & Robotics", font=font("r", 23), fill=(GRAY[0], GRAY[1], GRAY[2], int(255 * b)), anchor="lm")
            rows = [("LinkedIn", "https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/", (70, 140, 255)), ("GitHub", "https://github.com/Hadifard", (30, 40, 70))]
            for k, (lab, url, col) in enumerate(rows):
                c = ease_out((tl - (2.2 + k * 0.6)) / 0.6)
                y = 640 + k * 100
                if c <= 0:
                    continue
                ac = int(255 * c)
                d.rectangle((30, y, 1050, y + 80), fill=(PANEL[0], PANEL[1], PANEL[2], ac), outline=(col[0], col[1], col[2], ac), width=2)
                d.rectangle((46, y + 14, 196, y + 66), fill=(col[0], col[1], col[2], ac))
                d.text((121, y + 40), lab, font=font("b", 24), fill=(255, 255, 255, ac), anchor="mm")
                d.text((216, y + 40), url, font=font("m", 21 if k == 0 else 28), fill=(TXT[0], TXT[1], TXT[2], ac), anchor="lm")
            c = ease_out((tl - 3.6) / 0.8)
            d.text((540, 870), "Two Niryo robots playing Tic-Tac-Toe · three vision variants · tested Python code", font=font("r", 20), fill=(GRAY[0], GRAY[1], GRAY[2], int(255 * c)), anchor="mm")
        pre.append(ov)
    return dict(bg=bg, draw=draw)


BUILD = dict(hero=make_hero, challenge=make_challenge, color=make_color, bgsub=make_bgsub, calib=make_calib, seg=make_seg,
             motion=make_motion, taught=make_taught, standby=make_standby, brain=make_brain, design=make_design, end=make_end)


def main(out_path, start, end, only=None):
    scenes = {}
    total = int(TOTAL * FPS)
    frames = only if only is not None else range(start, end)
    ff = None if only is not None else subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", out_path], stdin=subprocess.PIPE)
    t_start = time.time()
    for n in frames:
        t = n / FPS
        key, a, b, num, title = next(s for s in SCENES if s[1] <= t < s[2])
        if key not in scenes:
            scenes[key] = BUILD[key]()
        sc = scenes[key]
        tl = t - a
        canvas = sc["bg"].copy()
        pre = []
        sc["draw"](canvas, tl, pre)
        canvas[0:5, :int(W * t / TOTAL)] = ACC
        img = Image.fromarray(canvas)
        d = ImageDraw.Draw(img, "RGBA")
        for fn in pre:
            fn(d)
        out = np.asarray(img)
        fade = min(ease(tl / 0.30), ease((b - a - tl) / 0.20))
        if fade < 1:
            out = (out.astype(np.float32) * fade).astype(np.uint8)
        if only is not None:
            Image.fromarray(out).save(f"/home/claude/film/t2_{n:05d}.png")
        else:
            ff.stdin.write(out.tobytes())
            if n % 120 == 0:
                print(f"frame {n}/{total} {time.time() - t_start:.0f}s", flush=True)
    if ff:
        ff.stdin.close()
        ff.wait()


if __name__ == "__main__":
    if sys.argv[1] == "test":
        main(None, 0, 0, only=[int(float(x) * FPS) for x in sys.argv[2:]])
    else:
        main(sys.argv[4], int(sys.argv[2]), int(sys.argv[3]))
