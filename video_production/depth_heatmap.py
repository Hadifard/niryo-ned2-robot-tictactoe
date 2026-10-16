#!/usr/bin/env python3
"""
File: video_production/depth_heatmap.py
Description: depth_heatmap.py — Depth-heatmap video from a normal video using Depth Anything V2.

Pipeline (designed for clean, stable, non-bleeding colours):
  1. Depth Anything V2 (ONNX, auto-downloaded from GitHub releases) -> relative depth (disparity)
  2. Temporal stabilisation: per-frame robust normalisation (1st-99th percentile), the
     normalisation range is smoothed over time, plus a light temporal Gaussian on depth
  3. Edge-aware upsampling: cubic resize + guided filter driven by the real RGB frame,
     so depth edges follow the object edges (no colour spreading across boundaries)
  4. Colormap (Turbo by default: red = near, blue = far) + colour bar legend
  5. High-quality H.264 export (CRF 14), original audio is kept

Outputs:  <name>_heatmap.mp4        heatmap only
          <name>_side_by_side.mp4   original | heatmap

Install:  pip install numpy scipy opencv-contrib-python onnxruntime   (onnxruntime-gpu if you have CUDA)
Run:      python depth_heatmap.py depth_detection.mp4
Examples: python depth_heatmap.py in.mp4 --model vitb --stride 1       # best quality, every frame
          python depth_heatmap.py in.mp4 --model vitl --stride 1       # strongest model (needs a GPU)
          python depth_heatmap.py in.mp4 --colormap inferno --chroma444
Note: the output depth is RELATIVE (not metric) depth - colours show near/far ordering.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d

BASE = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/"
MODELS = {  # Depth Anything V2, static 518x518 ONNX export
    "vits": "depth_anything_v2_vits.onnx",   # fastest   (~99 MB)
    "vitb": "depth_anything_v2_vitb.onnx",   # balanced  (~389 MB)
    "vitl": "depth_anything_v2_vitl.onnx",   # strongest (~1.3 GB)
}
SIZE = 518
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CMAPS = {"turbo": cv2.COLORMAP_TURBO, "inferno": cv2.COLORMAP_INFERNO, "jet": cv2.COLORMAP_JET,
         "magma": cv2.COLORMAP_MAGMA, "viridis": cv2.COLORMAP_VIRIDIS, "plasma": cv2.COLORMAP_PLASMA}


# ----------------------------------------------------------------------------- model
def get_session(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, MODELS[model])
    if not os.path.exists(path):
        print(f"Downloading {MODELS[model]} ...", flush=True)
        urllib.request.urlretrieve(BASE + MODELS[model], path + ".part")
        os.replace(path + ".part", path)
    avail = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider")
                 if p in avail] + ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=providers)
    print("Providers:", sess.get_providers(), flush=True)
    return sess


def predict(sess, frame_bgr, tta=False):
    """Returns relative disparity (higher = closer) at 518x518, float32."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].copy()
    name = sess.get_inputs()[0].name
    d = sess.run(None, {name: x})[0][0]
    if tta:  # horizontal-flip test-time augmentation (2x slower, slightly cleaner)
        d2 = sess.run(None, {name: x[..., ::-1].copy()})[0][0][:, ::-1]
        a, b = d.mean(), d2.mean()
        d = 0.5 * (d + (d2 - b) * (d.std() / (d2.std() + 1e-6)) + a)
    return d.astype(np.float32)


# ----------------------------------------------------------------------------- video io
def open_writer(path, w, h, fps, audio_src, chroma444, use_audio):
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps}", "-i", "-"]
    if use_audio:
        cmd += ["-i", audio_src, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    pix = "yuv444p" if chroma444 else "yuv420p"
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", pix,
            "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def colorbar(h, cmap, width=26):
    """Vertical colour bar (top = near, bottom = far) with labels, as a BGR patch."""
    bar_h = int(h * 0.55)
    grad = np.linspace(1.0, 0.0, bar_h, dtype=np.float32)[:, None].repeat(width, 1)
    bar = cv2.applyColorMap((grad * 255).astype(np.uint8), cmap)
    pad = 14
    patch = np.zeros((bar_h + 2 * pad, width + 2 * pad, 3), np.uint8)
    patch[pad:pad + bar_h, pad:pad + width] = bar
    cv2.rectangle(patch, (pad - 1, pad - 1), (pad + width, pad + bar_h), (255, 255, 255), 1)
    return patch, bar_h


def draw_legend(img, patch, bar_h, scale):
    h, w = img.shape[:2]
    ph, pw = patch.shape[:2]
    y0, x0 = (h - ph) // 2, w - pw - 18
    roi = img[y0:y0 + ph, x0:x0 + pw]
    cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, roi)  # dark translucent panel
    mask = patch.sum(2) > 0
    roi[mask] = patch[mask]
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "NEAR", (x0 - 62, y0 + 28), f, 0.5 * scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "FAR", (x0 - 48, y0 + ph - 14), f, 0.5 * scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-o", "--outdir", default="output")
    ap.add_argument("--model", choices=MODELS, default="vitb")
    ap.add_argument("--stride", type=int, default=1,
                    help="run the network every N frames and interpolate in between (speed vs. quality)")
    ap.add_argument("--colormap", choices=CMAPS, default="turbo")
    ap.add_argument("--tta", action="store_true", help="flip test-time augmentation (2x slower)")
    ap.add_argument("--temporal", type=float, default=1.0, help="temporal smoothing sigma in keyframes (0=off)")
    ap.add_argument("--gamma", type=float, default=1.0, help="<1 spreads far range, >1 spreads near range")
    ap.add_argument("--chroma444", action="store_true", help="yuv444p output: sharpest colour edges")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N frames (testing)")
    ap.add_argument("--cache", default="depth_cache")
    ap.add_argument("--model-dir", default="models")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if a.limit:
        N = min(N, a.limit)
    keys = list(range(0, N, a.stride))
    if keys[-1] != N - 1:
        keys.append(N - 1)
    print(f"{a.video}: {W}x{H} @ {fps:.3f} fps, {N} frames, {len(keys)} keyframes, model={a.model}", flush=True)

    # ---------------- pass 1: depth inference (cached & resumable)
    cdir = os.path.join(a.cache, f"{a.model}{'_tta' if a.tta else ''}")
    os.makedirs(cdir, exist_ok=True)
    todo = [k for k in keys if not os.path.exists(os.path.join(cdir, f"{k:06d}.npy"))]
    if todo:
        sess = get_session(a.model, a.model_dir)
        keyset, t0, done = set(todo), time.time(), 0
        for i in range(N):
            ok, frame = cap.read()
            if not ok:
                break
            if i in keyset:
                np.save(os.path.join(cdir, f"{i:06d}.npy"), predict(sess, frame, a.tta).astype(np.float16))
                done += 1
                el = time.time() - t0
                print(f"  depth {done}/{len(todo)}  elapsed {el:5.0f}s  eta {el / done * (len(todo) - done):5.0f}s",
                      flush=True)
    cap.release()

    # ---------------- temporal stabilisation (on 518x518 keyframe depths)
    D = np.stack([np.load(os.path.join(cdir, f"{k:06d}.npy")).astype(np.float32) for k in keys])
    lo = np.percentile(D.reshape(len(keys), -1), 1, axis=1)
    hi = np.percentile(D.reshape(len(keys), -1), 99, axis=1)
    if len(keys) > 3:  # smooth the normalisation window over ~0.5 s -> no colour pumping
        sg = max(1.0, 0.25 * fps / a.stride)
        lo, hi = gaussian_filter1d(lo, sg, mode="nearest"), gaussian_filter1d(hi, sg, mode="nearest")
    D = np.clip((D - lo[:, None, None]) / (hi - lo + 1e-6)[:, None, None], 0, 1)
    if a.temporal > 0 and len(keys) > 2:
        D = gaussian_filter1d(D, a.temporal, axis=0, mode="nearest")
    keys_arr = np.array(keys, np.float32)

    # ---------------- pass 2: render
    os.makedirs(a.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(a.video))[0]
    p_heat = os.path.join(a.outdir, f"{base}_heatmap.mp4")
    p_side = os.path.join(a.outdir, f"{base}_side_by_side.mp4")
    audio = not a.no_audio
    w_heat = open_writer(p_heat, W, H, fps, a.video, a.chroma444, audio)
    w_side = open_writer(p_side, 2 * W, H, fps, a.video, a.chroma444, audio)
    cmap = CMAPS[a.colormap]
    patch, bar_h = colorbar(H, cmap)
    scale = H / 854.0
    radius, eps = max(4, int(0.012 * W)), 250.0
    has_gf = hasattr(cv2, "ximgproc")
    if not has_gf:
        print("WARNING: opencv-contrib-python not found -> no guided filter (edges will be softer)")

    cap = cv2.VideoCapture(a.video)
    t0 = time.time()
    for t in range(N):
        ok, frame = cap.read()
        if not ok:
            break
        j = int(np.clip(np.searchsorted(keys_arr, t, side="right") - 1, 0, len(keys) - 2))
        span = keys_arr[j + 1] - keys_arr[j]
        al = float(np.clip((t - keys_arr[j]) / span, 0, 1))
        d = (1 - al) * D[j] + al * D[j + 1]
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
        if has_gf:  # edge-aware refinement guided by the real frame
            d = cv2.ximgproc.guidedFilter(frame, d.astype(np.float32), radius, eps)
        d = np.clip(d, 0, 1) ** a.gamma
        heat = cv2.applyColorMap((d * 255 + 0.5).astype(np.uint8), cmap)
        heat = draw_legend(heat, patch, bar_h, scale)
        w_heat.stdin.write(heat.tobytes())
        w_side.stdin.write(np.hstack([frame, heat]).tobytes())
        if t % 30 == 0:
            print(f"  render {t}/{N}  {time.time() - t0:4.0f}s", flush=True)
    cap.release()
    for w in (w_heat, w_side):
        w.stdin.close()
        w.wait()
    print("Done:\n ", p_heat, "\n ", p_side)


if __name__ == "__main__":
    sys.exit(main())
