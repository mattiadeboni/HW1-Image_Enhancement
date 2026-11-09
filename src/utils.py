# utility functions for reading and writing images

from pathlib import Path

import cv2
import numpy as np


def read_bgr(path: Path):
    # np.fromfile + imdecode avoids cv2.imread problems with non-ASCII paths on Windows
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def write_png(path, img):
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))
    