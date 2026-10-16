"""
File: niryo_tictactoe/common/camera.py
Description: Camera sources with one interface: `read() -> BGR ndarray` (640x480).
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import cv2
import numpy as np

FRAME_SIZE = (640, 480)


class NiryoCamera:
    """Wrist camera of the Niryo Ned/Ned2 (via the ROS bridge, as in the original project)."""

    def __init__(self, ip: str = "169.254.200.200"):
        import pyniryo                                  # imported lazily: not needed for tests
        from pyniryo2 import NiryoRos, Vision
        self._pyniryo = pyniryo
        self._ros = NiryoRos(ip)
        self._vision = Vision(self._ros)
        info = self._vision.get_camera_intrinsics()     # read once, not every frame
        self._k, self._d = info.intrinsics, info.distortion

    def read(self) -> np.ndarray:
        raw = self._pyniryo.uncompress_image(self._vision.get_img_compressed())
        img = self._pyniryo.undistort_image(raw, self._k, self._d)
        return cv2.resize(img, FRAME_SIZE)

    def close(self):
        self._ros.end()


class VideoCamera:
    """Replay a video file (offline development without the robot)."""

    def __init__(self, path: str, loop: bool = True):
        self.cap, self.loop = cv2.VideoCapture(path), loop

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("no more frames")
        return cv2.resize(frame, FRAME_SIZE)

    def close(self):
        self.cap.release()


class ImageCamera:
    """Always returns the same still image (unit tests)."""

    def __init__(self, image: np.ndarray):
        self.image = image

    def read(self) -> np.ndarray:
        return self.image.copy()

    def close(self):
        pass
