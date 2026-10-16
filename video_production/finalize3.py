"""
File: video_production/finalize3.py
Description: See the code below; this file has no module description yet.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""

import sys, subprocess
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

FD = "/usr/share/fonts/truetype/dejavu/"
W = H = 1080
BAND_Y = 972
FPS = 30

CUES = [
    (0.5, 2.6, "Computer vision & Robotics."),
    (2.8, 5.8, "A Tic-Tac-Toe system built under real-world constraints."),
    (6.4, 9.8, "Actuation error: gear backlash and limited repeatability create millimetre-scale pose errors that vision must correct."),
    (10.0, 13.2, "Illumination changes with daylight and lamps, so fixed colour thresholds are not photometrically invariant."),
    (13.4, 16.6, "The wrist camera is low grade: noisy, compressed and lens-distorted."),
    (16.8, 21.6, "Three-year-old deep detectors and segmenters were heavy, and they lowered the control rate on a laptop CPU."),
    (22.4, 26.4, "Colour is classified in HSV, where brightness is separated from hue and saturation."),
    (26.6, 30.6, "Each cell is a circular region: every pixel votes, which is more stable than averaging."),
    (30.8, 35.0, "White-patch balancing and CLAHE reduce the effect of changing illumination."),
    (35.2, 39.6, "The hue ranges were measured on the real chips under laboratory lighting."),
    (40.4, 46.2, "To detect chips in dim, over-exposed or specially lit rooms, where camera colours are unreliable, we chose the image subtraction method."),
    (46.4, 52.4, "Twice per second the robot photographs the board and subtracts consecutive images; what remains is the new or moved chip."),
    (52.6, 58.6, "The centre of that region is added to the list of known chip positions, or it updates the position of a chip that has moved."),
    (58.8, 65.0, "Colours are not detected at all: the robot follows every chip from the start of the game and labels the opponent's chips itself."),
    (65.2, 69.6, "Seat play of about 5 mm makes the difference image noisy, so the largest closed area and its centroid are used."),
    (70.4, 75.4, "Calibration uses fiducial circles on the paper as references."),
    (75.6, 80.6, "Two points define a similarity transform in closed form: rotation, scale and translation."),
    (80.8, 85.4, "The grid, chip trays and token spots follow the paper wherever it lies."),
    (85.6, 89.6, "A hand-eye step maps pixels to robot coordinates for a downward-looking camera."),
    (90.4, 95.6, "Segmentation test on a laptop camera with a COCO-trained deep network."),
    (95.8, 101.0, "It labels chairs and animals but knows nothing about our chips, and needs a GPU for real time."),
    (101.2, 106.2, "The classical alternative: HSV masks, distance transform and watershed give one mask per chip."),
    (106.4, 109.6, "About 60 ms per frame on a single CPU core, even when chips touch."),
    (110.4, 115.4, "Depth estimation: a network such as Depth Anything V2 gives relative depth, shown as a heat map beside the original image."),
    (115.6, 120.6, "For metric distance the chip diameter is the ruler: Z equals focal length times diameter divided by the apparent diameter."),
    (120.8, 125.0, "To avoid hitting chips, the board or the other arm, the robot keeps a safe clearance; it is measured only in the first passes."),
    (125.2, 127.6, "Then the model has learned the distances and moves without re-measuring."),
    (128.4, 134.0, "Paths are learned: the robot tries lift heights on simulated transfers and keeps the shortest collision-free one."),
    (134.2, 139.4, "A regression then predicts the lift from distance and obstacle height for every new transfer."),
    (139.6, 146.0, "In 3-D the arc is parabolic in height and can bend sideways; the tool descends vertically onto the chip."),
    (146.2, 152.0, "It picks a chip from its tray, places it on the target cell and then moves to the red chip."),
    (152.2, 158.5, "If an obstacle suddenly blocks the path, a lateral detour is searched and the move continues."),
    (158.7, 163.6, "The same idea applies to grasp-success prediction, anomaly detection for moved chips and per-lamp colour models."),
    (164.4, 169.4, "In dim light detection becomes unreliable, so positions were taught by hand."),
    (169.6, 175.6, "The arm is guided in learning mode, each pose is read and stored, and the code replays the table."),
    (176.4, 181.8, "To avoid collisions, the waiting robot folds into a rest pose that the other arm cannot reach."),
    (182.0, 186.6, "Its camera watches only the red token region and starts when the token appears."),
    (186.8, 189.6, "A fixed observation point gives every board reading the same viewpoint."),
    (190.4, 194.8, "Decision: alpha-beta minimax never loses and prefers faster wins."),
    (195.0, 199.6, "A move is accepted only after the vision output is verified against the expected board."),
    (200.4, 204.0, "Everything was first validated in a SolidWorks motion study."),
    (204.2, 207.6, "Then the board was designed, built from wood and painted."),
    (208.4, 212.0, "Hadi Sarhangi Fard, AI Engineer / Mechanical Engineer."),
    (212.2, 215.6, "Computer vision & Robotics. Code and contact on GitHub and LinkedIn."),
]


def wrap(text, font, width, rtl=False):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.getlength(test, direction="rtl" if rtl else None, language="fa" if rtl else None) <= width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def draw_cue(band_img, text, lang, alpha):
    d = ImageDraw.Draw(band_img, "RGBA")
    rtl = lang == "fa"
    font = ImageFont.truetype(FD + "DejaVuSans-Bold.ttf" if not rtl else FD + "DejaVuSans.ttf", 33 if not rtl else 34)
    lines = wrap(text, font, 980, rtl)
    lh = 44
    y0 = (108 - lh * len(lines)) // 2 + lh // 2 + 2
    for i, ln in enumerate(lines):
        y = y0 + i * lh
        kw = dict(direction="rtl", language="fa") if rtl else {}
        d.text((W // 2 + 2, y + 2), ln, font=font, fill=(0, 0, 0, int(200 * alpha)), anchor="mm", **kw)
        d.text((W // 2, y), ln, font=font, fill=(255, 255, 255, int(255 * alpha)), anchor="mm", **kw)


def srt_time(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"


def write_srt(lang, path):
    with open(path, "w", encoding="utf-8") as fh:
        for i, (a, b, en) in enumerate(CUES, 1):
            fh.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{en}\n\n")


def build(lang, out):
    cap = cv2.VideoCapture("/home/claude/film/base3.mp4")
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-i", "/home/claude/film/music.wav", "-map", "0:v", "-map", "1:a",
                           "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / FPS
        for a, b, en in CUES:
            if a - 0.01 <= t <= b:
                al = min(1.0, (t - a) / 0.2 + 0.001, (b - t) / 0.2 + 0.001)
                band = Image.fromarray(cv2.cvtColor(frame[BAND_Y:], cv2.COLOR_BGR2RGB))
                draw_cue(band, en, 'en', max(0.0, min(1.0, al)))
                frame[BAND_Y:] = cv2.cvtColor(np.asarray(band), cv2.COLOR_RGB2BGR)
                break
        ff.stdin.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
        n += 1
        if n % 300 == 0:
            print(lang, n, flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    write_srt("en", "/home/claude/film/subtitles_EN.srt")
    build("en", "/home/claude/film/final3.mp4")
    print("done")
