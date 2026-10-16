"""
File: niryo_tictactoe/segmentation/yolo_seg_demo.py
Description: Deep-learning segmentation experiment (YOLOv8-seg) - the heavy alternative.

This is the kind of model tested in the lab recordings: a COCO-pretrained
network labels persons, chairs, monitors ... but does NOT know our chips, is
sensitive to lighting and needs a GPU for real time.  Kept as a reference /
benchmark against segmentation/classical_segmentation.py.

    pip install ultralytics
    python -m segmentation.yolo_seg_demo --source 0            # webcam
    python -m segmentation.yolo_seg_demo --source video.mp4 --out out.mp4
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import argparse
import time

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO                    # heavy import, only needed here
    model = YOLO(args.model)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    writer = None
    frames, start = 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, conf=args.conf, verbose=False)[0]
        annotated = result.plot()                   # masks + boxes + labels
        frames += 1
        fps = frames / (time.time() - start)
        cv2.putText(annotated, f"YOLOv8-seg  {fps:4.1f} FPS  {len(result.boxes)} objects",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if args.out:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(annotated)
        cv2.imshow("segmentation", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
