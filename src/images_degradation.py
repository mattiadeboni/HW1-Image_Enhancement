"""
This script degrades the original images with the gain-bias pixel transformation + Gaussian noise: 

f = GAIN * f + BIAS + N(0, SIGMA)
"""

from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path("data/original")
DST_DIR = Path("data/degraded")

GAIN = 0.5    # reduces contrast
BIAS = 30.0   # adds brightness
SIGMA = 8.0   # amount of Gaussian noise
SEED = 42     # base seed

DST_DIR.mkdir(parents=True, exist_ok=True) # create the output directory if it doesn't exist


def read_bgr(path: Path):
    # np.fromfile + imdecode avoids cv2.imread problems with non-ASCII paths on Windows
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def degrade(img, rng):
    f = img.astype(np.float32)                 # convert to float32 for the pixel transformation
    f = GAIN * f + BIAS                        # we apply the pixel transformation (gain-contrast and bias-brightness)
    f += rng.normal(0.0, SIGMA, f.shape)       # noise, drawn from the seeded generator
    return np.clip(f, 0, 255).astype(np.uint8) # clip once, at the end (important to avoid artifacts) and convert back to uint8

# get all image paths
paths = sorted(p for p in SRC_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

for i, path in enumerate(paths):
    img = read_bgr(path)                        # read the image in BGR format
    if img is None:
        print(f"skip {path.name}: cannot read")
        continue
    rng = np.random.default_rng(SEED + i)                # create a seeded random generator for reproducibility
    out = degrade(img, rng)                              # degrade the image
    cv2.imwrite(str(DST_DIR / f"{path.stem}.png"), out)  # PNG: no extra compression
