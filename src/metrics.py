"""
Compute quantitative metrics for the degraded input and every enhancement method.

Each result is compared with the original clean image:
  PSNR : peak signal-to-noise ratio in dB (higher is better)
  SSIM : structural similarity, 0 to 1 (higher is better)
  MAE  : mean absolute error on the 0-255 scale (lower is better)
  MSE  : mean squared error on the 0-255 scale (lower is better)
"""

import csv
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

from utils import read_bgr

ORIG_DIR = Path("data/original")
DEG_DIR = Path("data/degraded")
RES_DIR = Path("results")
OUT_CSV = RES_DIR / "metrics.csv"

METHODS = ["baseline", "stretch", "equalize", "denoise_stretch"]

# "degraded" is the unprocessed input, so every method can be compared with it
SOURCES = {"degraded": DEG_DIR}
SOURCES.update({name: RES_DIR / name for name in METHODS})


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def psnr(mse_value):
    if mse_value == 0:
        return float("inf")  # identical images
    return 10.0 * np.log10(255.0 ** 2 / mse_value)

# empty list to store the metrics for each image and method
rows = []

for ref_path in sorted(ORIG_DIR.glob("*.png")):
    ref = read_bgr(ref_path).astype(np.float64) # load the original clean image

    for method, folder in SOURCES.items():
        out_path = folder / ref_path.name
        if not out_path.exists():
            print(f"missing {out_path}, skipped")
            continue

        img = read_bgr(out_path).astype(np.float64) # load the enhanced image
        if img.shape != ref.shape:
            raise ValueError(f"{out_path} has shape {img.shape}, expected {ref.shape}")

        m = mse(ref, img) # compute the mean squared error between the original and enhanced images

        rows.append({
            "image": ref_path.stem,
            "method": method,
            "psnr_db": round(psnr(m), 3),
            "ssim": round(structural_similarity(ref, img, channel_axis=2, data_range=255), 4),
            "mae": round(float(np.mean(np.abs(ref - img))), 3),
            "mse": round(m, 3),
        })

        print(f"{ref_path.stem:<12} {method:<16} PSNR={rows[-1]['psnr_db']:6.2f} "
              f"| SSIM={rows[-1]['ssim']:.4f}")

RES_DIR.mkdir(exist_ok=True)

# write the metrics to a CSV file
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "method", "psnr_db", "ssim", "mae", "mse"])
    writer.writeheader()
    writer.writerows(rows)

# average over images, one line per method
print("\nMean over images:")
for method in SOURCES:
    subset = [r for r in rows if r["method"] == method]
    if not subset:
        continue
    print(f"{method:<16} PSNR={np.mean([r['psnr_db'] for r in subset]):6.2f}  "
          f"SSIM={np.mean([r['ssim'] for r in subset]):.4f}  "
          f"MAE={np.mean([r['mae'] for r in subset]):6.2f}")

print(f"\nSaved {len(rows)} rows to {OUT_CSV}")
