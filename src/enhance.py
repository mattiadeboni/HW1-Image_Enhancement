"""
Apply the enhancement methods to the degraded images.

Methods:
  baseline    : degraded image, unchanged
  stretch     : linear contrast stretching, per channel (min-max to 0-255)
  equalize    : histogram equalization of the L channel in CIE-Lab
  denoise_stretch : bilateral filter followed by per-channel stretching
"""

from pathlib import Path

import cv2
import numpy as np

from utils import read_bgr, write_png

DEG_DIR = Path("data/degraded")
OUT_DIR = Path("results")


def baseline(img):
    # returns the degraded image unchanged
    return img.copy()


def stretch(img):
    # each channel is rescaled independently, so the color balance can shift
    out = np.empty(img.shape, dtype=np.float32)                 # uninitialized float32 array with the same shape as img
    for c in range(3):                                          # iterate over the three color channels (B, G, R)
        ch = img[..., c].astype(np.float32)                     # convert the channel to float32 for processing
        lo, hi = ch.min(), ch.max()                             # find the minimum and maximum pixel values in the channel
        out[..., c] = (ch - lo) * 255.0 / max(hi - lo, 1e-6)    # rescale the channel values to the range [0, 255], avoiding division by zero with a small epsilon
    return np.clip(out, 0, 255).astype(np.uint8)                # clip the rescaled values to the range [0, 255] and convert back to uint8


def equalize(img):
    # only the lightness channel is equalized, so hue is kept
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)     # convert the image from BGR color space to CIE-Lab color space
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])    # apply histogram equalization to the L channel (lightness) of the Lab image
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)    # convert the Lab image back to BGR color space and return it


def denoise_stretch(img):
    # bilateral filter smooths the Gaussian noise while keeping edges
    d = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=9)  # apply a bilateral filter to the image to reduce noise while preserving edges
    return stretch(d)   # apply the stretch function to the denoised image and return the result


METHODS = {
    "baseline": baseline,
    "stretch": stretch,
    "equalize": equalize,
    "denoise_stretch": denoise_stretch,
}

paths = sorted(DEG_DIR.glob("*.png"))

for path in paths:
    img = read_bgr(path)
    if img is None:
        print(f"skip {path.name}: cannot read")
        continue
    for name, fn in METHODS.items():
        out_dir = OUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        write_png(out_dir / path.name, fn(img))
    print(f"{path.name}: {len(METHODS)} methods written")
    