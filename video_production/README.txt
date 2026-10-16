Film render code (Python)

Pipeline (run in this order):
  1. prep.py        cut/crop the source clips with ffmpeg into clips/ (extra clips were prepared inline, see notes)
  2. music.py       synthesise the soundtrack (music.wav, 216 s, numpy/scipy)
  3. scenes3.py     render frames: python scenes3.py chunk START END out.mp4   (run3.sh renders 12 chunks of 540 frames)
                    scenes3.py imports render2.py (scenes, layout) and lib.py (fonts, clip reader, code typing panel)
  4. concatenate    ffmpeg -f concat -safe 0 -i list3.txt -c copy base3.mp4
  5. finalize3.py   burn English subtitles, mux music.wav, write subtitles_EN.srt

Requirements: python3, numpy, scipy, opencv-python, pillow, pygments, ffmpeg, DejaVu fonts.
Notes: paths are hard-coded to /home/claude/... and /mnt/user-data/uploads; edit them for your machine.
render2.py/scenes3.py import the project code (niryo_tictactoe/) and overlay_real.py (real-footage overlay).
The depth_strip clip and the hero/segmentation clips were made with ffmpeg commands shown in the chat session.
