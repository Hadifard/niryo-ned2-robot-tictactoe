"""
File: video_production/music.py
Description: Original soundtrack, 120 s, 120 BPM (one bar = exactly 2 s so scene cuts land on bars).
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import wavfile

SR = 44100
BPM = 120
BEAT = 60 / BPM            # 0.5 s
BAR = 4 * BEAT             # 2.0 s
DUR = 216.0
N = int(SR * DUR)
rng = np.random.default_rng(7)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def lp(x, fc, order=2):
    b, a = butter(order, fc / (SR / 2), "low")
    return lfilter(b, a, x)


def hp(x, fc, order=2):
    b, a = butter(order, fc / (SR / 2), "high")
    return lfilter(b, a, x)


def place(buf, sig, t):
    i = int(t * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[:j - i]


def saw(f, n, detune=(0.0,)):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for d in detune:
        ph = (t * f * (1 + d)) % 1.0
        out += 2 * ph - 1
    return out / len(detune)


def kick():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 120 * np.exp(-t * 22) + 42
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 9) * 1.0


def hat(open_=False):
    n = int((0.22 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    return hp(rng.normal(0, 1, n), 7000) * np.exp(-t * (18 if open_ else 70)) * 0.25


def clap():
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    env = np.exp(-t * 28) + 0.6 * np.exp(-((t - 0.012) ** 2) * 40000)
    return hp(lp(rng.normal(0, 1, n), 5500), 900) * env * 0.4


def pluck(f, length=0.35):
    n = int(length * SR)
    t = np.arange(n) / SR
    sig = saw(f, n, (-0.003, 0.0, 0.004))
    sig = lp(sig, 3800) * np.exp(-t * 7.5)
    return sig * 0.22


def bass(f, length):
    n = int(length * SR)
    t = np.arange(n) / SR
    env = np.minimum(1, t * 60) * np.exp(-t * 1.6)
    return (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t)) * env * 0.55


def pad(notes, length):
    n = int(length * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for m in notes:
        sig += saw(midi(m), n, (-0.006, 0.0, 0.007))
    sig = lp(sig / len(notes), 1400 + 900 * np.sin(2 * np.pi * 0.08 * t).mean())
    env = np.minimum(1, t / 0.9) * np.minimum(1, (length - t) / 0.9)
    return sig * env * 0.30


# chord progression, one chord per 2 bars (4 s): Am - F - C - G  (root, third, fifth in MIDI)
CHORDS = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]
ARP_PATTERN = [0, 1, 2, 1, 2, 1, 0, 2]            # 8th notes over the chord tones, +octave variants


def build():
    L = np.zeros(N)
    R = np.zeros(N)
    total_bars = int(DUR / BAR)                    # 60

    def section_gain(bar):
        # energy curve: soft intro, build, full groove, breakdown, final lift, tail
        if bar < 4:   return dict(pad=1, arp=0.0, drums=0.0, bass=0.0)
        if bar < 8:   return dict(pad=1, arp=0.7, drums=0.0, bass=0.6)
        if bar < 12:  return dict(pad=1, arp=0.9, drums=0.6, bass=1.0)
        if bar < 100:  return dict(pad=0.9, arp=1.0, drums=1.0, bass=1.0)
        if bar < 104:  return dict(pad=1.0, arp=0.8, drums=0.0, bass=0.5)       # breakdown before the credits
        if bar < 106:  return dict(pad=1.0, arp=1.0, drums=1.0, bass=1.0)       # final lift
        return dict(pad=1.0, arp=0.0, drums=0.0, bass=0.0)

    for bar in range(total_bars):
        g = section_gain(bar)
        chord = CHORDS[(bar // 2) % 4]
        t0 = bar * BAR
        if bar % 2 == 0:
            place(L, pad(chord, 2 * BAR + 0.4) * g["pad"], t0)
            place(R, pad([c + 12 for c in chord[:2]] + [chord[2]], 2 * BAR + 0.4) * g["pad"], t0)
        for step in range(8):                       # 8th notes
            t = t0 + step * BEAT / 2
            if g["arp"] > 0:
                tone = chord[ARP_PATTERN[step] % 3] + (24 if step % 4 == 3 else 12)
                sig = pluck(midi(tone)) * g["arp"] * (1.0 if step % 2 == 0 else 0.7)
                pan = 0.35 + 0.3 * ((step % 4) / 3)
                place(L, sig * (1 - pan), t)
                place(R, sig * pan, t)
            if g["drums"] > 0:
                h = hat(open_=(step % 4 == 3)) * g["drums"]
                place(L, h * 0.6, t)
                place(R, h * 0.9, t)
        for beat in range(4):
            t = t0 + beat * BEAT
            if g["drums"] > 0:
                k = kick() * g["drums"] * 0.9
                place(L, k, t)
                place(R, k, t)
                if beat in (1, 3):
                    c = clap() * g["drums"]
                    place(L, c * 0.8, t)
                    place(R, c * 0.8, t)
        if g["bass"] > 0:
            root = midi(chord[0] - 24)
            for off in (0.0, 1.5 * BEAT, 2.0 * BEAT, 3.0 * BEAT):
                b = bass(root if off != 3.0 * BEAT else root * 1.5, 0.45) * g["bass"]
                place(L, b, t0 + off)
                place(R, b, t0 + off)

    # risers into the scene changes that matter (bars 4, 12, 50, 54)
    for bar in (4, 12, 100, 104):
        n = int(2 * BAR * SR / 2)
        t = np.arange(n) / SR
        noise = hp(rng.normal(0, 1, n), 2500) * (t / t[-1]) ** 2 * 0.22
        place(L, noise, bar * BAR - n / SR)
        place(R, noise, bar * BAR - n / SR)

    # final chord tail + gentle master processing
    mix = np.stack([L, R], axis=1)
    mix = np.tanh(mix * 1.1)
    fade_in = np.minimum(1, np.arange(N) / (SR * 1.5))
    fade_out = np.minimum(1, (N - np.arange(N)) / (SR * 3.0))
    mix *= (fade_in * fade_out)[:, None]
    mix /= np.abs(mix).max() / 0.89
    return (mix * 32767).astype(np.int16)


if __name__ == "__main__":
    wavfile.write("/home/claude/film/music.wav", SR, build())
    print("music.wav written")
