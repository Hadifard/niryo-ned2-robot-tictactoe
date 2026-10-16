"""
File: video_production/prep.py
Description: See the code below; this file has no module description yet.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""

import subprocess, os
S1="/home/claude/s1/selected for making movie_01/"; S2="/home/claude/s2/selected for making movie_02/"; S3="/home/claude/s3/selected for making movie_03/"
SRC = {
 "v01": S1+"01_working on codes in lab.mp4",
 "v02": S1+"02_working on color detection.mp4",
 "v03": S1+"03_working on segmentation codes_compressed (1).mp4",
 "v04a": S1+"05_test of  segmentation codes on robot in lab_compressed.mp4",
 "cad": S1+"08_Drawing and design playing ground by CAD softwares.mp4",
 "v04b": S2+"04_test color detection of codes on robot in lab.mp4",
 "sim": S2+"12_simulation in vitual environment_01.mp4",
 "final": S2+"Final video of playing robots with each other before expo.mp4",
 "v06": S3+"06_working on codes in lab _01_origin_compressed.mp4",
 "v07": S3+"07_.mp4",
 "v11": S3+"11_VID-20241024-WA0001 -original - Trim.mp4",
}
CROP = {"v02": "crop=640:480:0:0", "v03": "crop=1024:768:0:0"}
# name: (src, ss, out_duration, speed, W, H, fx, fy)
JOBS = {
 "hero01":  ("v01", 0.0, 7.8, 0.45, 1080, 1080, 0.95, 0.5),
 "s1_lab":  ("v04b", 2.0, 14.5, 1.0, 711, 400, 0.5, 0.5),
 "s1_rob":  ("v04a", 4.0, 14.5, 1.0, 297, 400, 0.5, 0.55),
 "s2_v02":  ("v02", 0.3, 18.5, 1.0, 533, 400, 0.5, 0.5),
 "s3_06":   ("v06", 0.0, 19.1, 0.45, 360, 640, 0.5, 0.5),
 "s4_lab":  ("v04b", 17.0, 14.5, 1.0, 711, 400, 0.5, 0.5),
 "s4_07":   ("v07", 40.0, 14.5, 1.0, 297, 400, 0.5, 0.5),
 "s5_v03":  ("v03", 62.0, 16.5, 1.0, 533, 400, 0.5, 0.5),
 "s6_fin":  ("final", 2.0, 12.5, 1.0, 711, 400, 0.5, 0.5),
 "s7_sim":  ("sim", 16.0, 8.5, 1.0, 711, 400, 0.5, 0.5),
 "s7_cad":  ("cad", 2.0, 8.5, 1.0, 408, 442, 0.5, 0.5),
 "end_bg":  ("v11", 8.0, 8.5, 1.0, 1080, 1080, 0.5, 0.5),
 "s0_face": ("v02", 4.0, 3.0, 1.0, 640, 480, 0.5, 0.5),   # for the portrait on the end card
}
def run(name):
    src, ss, dur, speed, W, H, fx, fy = JOBS[name]
    out = f"/home/claude/film/clips/{name}.mp4"
    if os.path.exists(out): return
    vf = []
    if src in CROP: vf.append(CROP[src])
    vf += [f"scale={W}:{H}:force_original_aspect_ratio=increase", f"crop={W}:{H}:(iw-{W})*{fx}:(ih-{H})*{fy}",
           f"setpts=PTS/{speed}", "fps=30"]
    cmd = ["ffmpeg","-v","error","-y","-ss",str(ss),"-t",str(dur*speed),"-i",SRC[src],"-vf",",".join(vf),
           "-an","-c:v","libx264","-crf","14","-preset","veryfast","-pix_fmt","yuv420p",out]
    subprocess.run(cmd, check=True)
    print("ok", name)
for n in JOBS: run(n)
