"""
Build side-by-side comparison figures: original, degraded, and each method.
"""

from pathlib import Path

import cv2
import numpy as np

from utils import read_bgr, write_png

ORIG_DIR = Path("data/original")
DEG_DIR = Path("data/degraded")
RES_DIR = Path("results")
FIG_DIR = Path("figures")

# column order in the figure: reference, input, then the methods
PANELS = [
    ("original", ORIG_DIR),
    ("degraded", DEG_DIR),
    ("stretch", RES_DIR / "stretch"),
    ("equalize", RES_DIR / "equalize"),
    ("denoise_stretch", RES_DIR / "denoise_stretch"),
]

TILE = 384        # each panel is resized to TILE x TILE so all panels share one scale
LABEL_H = 32      # height of the title strip above each panel
FIG_DIR.mkdir(exist_ok=True)


def make_tile(img, label):                          # returns a TILE x TILE image with a title strip above it
    h, w = img.shape[:2]                            # get the height and width of the input image
    scale = TILE / max(h, w)                        # fit the longer side to TILE
    nw, nh = round(w * scale), round(h * scale)     # compute the new width and height while maintaining the aspect ratio
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)   # resize the image to the new dimensions using area interpolation for downscaling

    # white square canvas; the photo is centered, so the padding is even on both sides
    canvas = np.full((TILE, TILE, 3), 255, dtype=np.uint8) # create a white canvas of size TILE x TILE with 3 color channels (BGR)
    y0, x0 = (TILE - nh) // 2, (TILE - nw) // 2            # compute the top-left corner coordinates to center the resized image on the canvas
    canvas[y0:y0 + nh, x0:x0 + nw] = resized               # place the resized image onto the canvas at the computed coordinates

    bar = np.full((LABEL_H, TILE, 3), 255, dtype=np.uint8) # create a white title strip of height LABEL_H and width TILE
    cv2.putText(bar, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA) # draw the label text on the title strip using a black color, specified font, size, and thickness
    return np.vstack([bar, canvas])                        # stack the title strip above the canvas to create the final tile image


for ref_path in sorted(ORIG_DIR.glob("*.png")):
    tiles = []
    for label, folder in PANELS:
        img = read_bgr(folder / ref_path.name)
        if img is None:
            raise FileNotFoundError(f"missing {folder / ref_path.name}")
        tiles.append(make_tile(img, label))

    row = np.hstack(tiles) # horizontally stack the individual tile images to create a single row image for the figure
    write_png(FIG_DIR / f"comparison_{ref_path.stem}.png", row)
    print(f"saved figures/comparison_{ref_path.stem}.png  {row.shape[1]}x{row.shape[0]}")
