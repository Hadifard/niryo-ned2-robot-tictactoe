"""
File: video_production/lib.py
Description: See the code below; this file has no module description yet.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""

import re, math
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

W = H = 1080
FPS = 30
FD = "/usr/share/fonts/truetype/dejavu/"
_fc = {}


def font(name, size):
    key = (name, size)
    if key not in _fc:
        path = {"r": "DejaVuSans.ttf", "b": "DejaVuSans-Bold.ttf", "m": "DejaVuSansMono.ttf",
                "mb": "DejaVuSansMono-Bold.ttf"}[name]
        _fc[key] = ImageFont.truetype(FD + path, size)
    return _fc[key]


# palette (RGB)
BG1, BG2 = (9, 13, 24), (20, 30, 56)
PANEL = (19, 25, 41)
PANEL2 = (26, 33, 52)
LINE = (52, 66, 98)
ACC = (0, 205, 255)       # cyan
ACC2 = (255, 120, 80)     # warm orange
ACC3 = (150, 120, 255)    # violet
GOOD = (80, 220, 140)
TXT = (232, 238, 250)
DIM = (140, 152, 180)
CODE_BG = (30, 34, 44)

CODE_COL = {
    "kw": (198, 120, 221), "str": (152, 195, 121), "com": (110, 120, 138), "num": (209, 154, 102),
    "fn": (97, 175, 239), "cls": (229, 192, 123), "bi": (86, 182, 194), "op": (86, 182, 194),
    "txt": (205, 210, 222),
}


def lerp(a, b, t):
    return a + (b - a) * t


def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp01(t)
    return 1 - (1 - t) ** 3


# ------------------------------------------------------------------ gradient bg
def gradient_bg(c1=BG1, c2=BG2, glow=None):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = (yy * 0.7 + xx * 0.3) / (H * 0.7 + W * 0.3)
    img = np.zeros((H, W, 3), np.float32)
    for k in range(3):
        img[..., k] = c1[k] + (c2[k] - c1[k]) * t
    if glow:
        cx, cy, col, rad = glow
        d = np.hypot(xx - cx, yy - cy) / rad
        g = np.clip(1 - d, 0, 1) ** 2 * 0.35
        for k in range(3):
            img[..., k] += g * col[k]
    # faint dot grid
    img[::36, ::36] += 9
    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------- clips
class Clip:
    def __init__(self, path, loop=False):
        self.cap = cv2.VideoCapture(path)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.loop, self.last, self.cache = loop, -2, None

    def frame(self, t):
        i = int(t * FPS)
        i = i % self.n if self.loop else min(i, self.n - 1)
        if i == self.last:
            return self.cache
        if i != self.last + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = self.cap.read()
        if not ok:
            return self.cache
        self.last, self.cache = i, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        return self.cache


_masks = {}


def rmask(w, h, r):
    key = (w, h, r)
    if key not in _masks:
        m = np.zeros((h * 2, w * 2), np.uint8)
        cv2.rectangle(m, (r * 2, 0), (w * 2 - r * 2, h * 2), 255, -1)
        cv2.rectangle(m, (0, r * 2), (w * 2, h * 2 - r * 2), 255, -1)
        for cx, cy in ((r * 2, r * 2), (w * 2 - r * 2, r * 2), (r * 2, h * 2 - r * 2), (w * 2 - r * 2, h * 2 - r * 2)):
            cv2.circle(m, (cx, cy), r * 2, 255, -1)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
        _masks[key] = (m > 127)[..., None]
    return _masks[key]


def fit_cover(img, w, h, fx=0.5, fy=0.5):
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x0, y0 = int((nw - w) * fx), int((nh - h) * fy)
    return r[y0:y0 + h, x0:x0 + w]


def paste(canvas, img, x, y, w, h, radius=0, fx=0.5, fy=0.5):
    if img.shape[1] != w or img.shape[0] != h:
        img = fit_cover(img, w, h, fx, fy)
    roi = canvas[y:y + h, x:x + w]
    if radius:
        np.copyto(roi, img, where=rmask(w, h, radius))
    else:
        roi[:] = img


def dim(canvas, amount):
    """amount 0..1 darkens in place."""
    if amount > 0:
        canvas[:] = (canvas.astype(np.float32) * (1 - amount)).astype(np.uint8)


# ------------------------------------------------------------------ text utils
def wrap(text, fnt, width):
    out = []
    for para in text.split("\n"):
        words, cur = para.split(), ""
        for w_ in words:
            test = (cur + " " + w_).strip()
            if fnt.getlength(test) <= width:
                cur = test
            else:
                out.append(cur)
                cur = w_
        out.append(cur)
    return out


def text_c(d, xy, s, fnt, fill, anchor="ma"):
    d.text(xy, s, font=fnt, fill=fill, anchor=anchor)


def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rectangle(box, fill=fill, outline=outline, width=width)      # square corners (formal look)


def pill(d, x, y, s, fnt, fill, fg=(10, 14, 26), pad=14, h=34):
    w_ = int(fnt.getlength(s)) + pad * 2
    rrect(d, (x, y, x + w_, y + h), h // 2, fill=fill)
    d.text((x + pad, y + h // 2), s, font=fnt, fill=fg, anchor="lm")
    return w_


# -------------------------------------------------------------- source excerpts
def block(path, start_pat, extra_after=0):
    """Return (first_line_number, text) of the def/class starting at start_pat."""
    lines = open(path).read().split("\n")
    for i, ln in enumerate(lines):
        if re.search(start_pat, ln):
            indent = len(ln) - len(ln.lstrip())
            j = i + 1
            while j < len(lines):
                s = lines[j]
                if s.strip() and (len(s) - len(s.lstrip())) <= indent and not s.lstrip().startswith(("#", ")", "]", "}")):
                    break
                j += 1
            while j > i + 1 and not lines[j - 1].strip():
                j -= 1
            return i + 1, "\n".join(lines[i:j + extra_after])
    raise KeyError(start_pat)


def span(path, first_pat, last_pat):
    lines = open(path).read().split("\n")
    a = next(i for i, l in enumerate(lines) if re.search(first_pat, l))
    b = next(i for i in range(a, len(lines)) if re.search(last_pat, lines[i]))
    return a + 1, "\n".join(lines[a:b + 1])


# ----------------------------------------------------------------- code panel
def _kind(tt):
    if tt in Token.Keyword or tt in Token.Operator.Word:
        return "kw"
    if tt in Token.Name.Function or tt in Token.Name.Function.Magic:
        return "fn"
    if tt in Token.Name.Class:
        return "cls"
    if tt in Token.Name.Builtin or tt in Token.Name.Builtin.Pseudo:
        return "bi"
    if tt in Token.String:
        return "str"
    if tt in Token.Comment:
        return "com"
    if tt in Token.Number:
        return "num"
    if tt in Token.Operator:
        return "op"
    return "txt"


class CodePanel:
    """Syntax-highlighted code that is typed live and auto-scrolls."""
    FS, LH, GUT = 16, 23, 52

    def __init__(self, x, y, w, h, tabs):
        # tabs: list of dict(title, first, text, t0, t1)  (times relative to scene start)
        self.x, self.y, self.w, self.h = x, y, w, h
        self.fnt = font("m", self.FS)
        self.cw = self.fnt.getlength("M")
        self.top = 40
        self.vis = (h - self.top - 10) // self.LH
        self.tabs = []
        for tab in tabs:
            code = tab["text"].replace("\t", "    ")
            toks = list(lex(code, PythonLexer()))
            lines, cur = [], []
            for tt, val in toks:
                parts = val.split("\n")
                for k, p in enumerate(parts):
                    if k > 0:
                        lines.append(cur)
                        cur = []
                    if p:
                        cur.append((p, CODE_COL[_kind(tt)]))
            if cur:
                lines.append(cur)
            while lines and not lines[-1]:
                lines.pop()
            imgs, lens = [], []
            for k, segs in enumerate(lines):
                im = Image.new("RGB", (w - 24, self.LH), CODE_BG)
                d = ImageDraw.Draw(im)
                d.text((self.GUT - 12, self.LH // 2), str(tab["first"] + k), font=font("m", 13), fill=(88, 96, 114), anchor="rm")
                px, n = self.GUT, 0
                for s, col in segs:
                    d.text((px, self.LH // 2), s, font=self.fnt, fill=col, anchor="lm")
                    px += self.cw * len(s)
                    n += len(s)
                imgs.append(np.asarray(im))
                lens.append(n)
            tab = dict(tab, imgs=imgs, lens=lens, total=max(1, sum(lens) + len(lens)))
            self.tabs.append(tab)
        self.base = self._chrome()

    def _chrome(self):
        im = Image.new("RGB", (self.w, self.h), CODE_BG)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, self.w, 34), fill=(22, 26, 36))
        for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
            d.ellipse((14 + i * 22, 11, 26 + i * 22, 23), fill=c)
        return np.asarray(im).copy()

    def draw(self, canvas, t, d_title):
        """t = seconds since scene start. d_title(draw_surface) handled by caller via return of title."""
        tab = self.tabs[0]
        for tb in self.tabs:
            if t >= tb["t0"]:
                tab = tb
        panel = self.base.copy()
        prog = clamp01((t - tab["t0"]) / max(0.01, tab["t1"] - tab["t0"]))
        typed = prog * tab["total"]
        # current line / partial
        acc, cur_line, cur_col = 0, len(tab["lens"]) - 1, tab["lens"][-1]
        for i, n in enumerate(tab["lens"]):
            if typed < acc + n + 1:
                cur_line, cur_col = i, int(typed - acc)
                break
            acc += n + 1
        else:
            cur_line, cur_col = len(tab["lens"]) - 1, tab["lens"][-1]
        scroll = max(0.0, cur_line - (self.vis - 3))
        s0 = int(scroll)
        frac = int((scroll - s0) * self.LH)
        y = self.top - frac
        for i in range(s0, min(len(tab["imgs"]), s0 + self.vis + 2)):
            if i > cur_line:
                break
            img = tab["imgs"][i]
            if i == cur_line:
                img = img.copy()
                cut = int(self.GUT + cur_col * self.cw)
                img[:, cut:] = CODE_BG
                if int(t * 2.5) % 2 == 0:
                    img[3:self.LH - 3, cut:cut + 2] = (240, 240, 240)
            y0, y1 = y + (i - s0) * self.LH, y + (i - s0 + 1) * self.LH
            a, b = max(y0, 34), min(y1, self.h)
            if b > a:
                panel[a:b, 12:self.w - 12] = img[a - y0:b - y0]
        # fade of top edge under chrome handled by chrome drawn last
        panel[0:34] = self.base[0:34]
        paste(canvas, panel, self.x, self.y, self.w, self.h)
        return tab["title"], cur_line + 1, len(tab["lens"])
