"""
analyze_noise.py
-----------------------
Characterizes the low-light noise model using the public 460 pairs.
Run from the repo root (where competition_data/public/ exists):

    python analyze_noise.py --gt_dir competition_data/public/ground_truth \
                             --noisy_dir competition_data/public/noisy \
                             --n_samples 40
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True, type=Path)
    ap.add_argument("--noisy_dir", required=True, type=Path)
    ap.add_argument("--n_samples", type=int, default=40)
    args = ap.parse_args()

    gt_paths = sorted(args.gt_dir.glob("*.png"))[: args.n_samples]

    residual_stds, residual_means = [], []
    per_channel_std = []
    brightness_vs_noise = []

    for gt_path in gt_paths:
        img_id = gt_path.stem
        noisy_path = args.noisy_dir / f"{img_id}_noise.png"
        if not noisy_path.exists():
            continue

        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        noisy = np.asarray(Image.open(noisy_path).convert("RGB"), dtype=np.float32) / 255.0
        residual = noisy - gt

        residual_means.append(residual.mean())
        residual_stds.append(residual.std())
        per_channel_std.append(residual.std(axis=(0, 1)))  # R,G,B

        # bucket pixels by GT brightness, check if noise std varies (signal-dependent noise check)
        gt_gray = gt.mean(axis=2)
        res_gray = np.abs(residual).mean(axis=2)
        for lo, hi in [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]:
            mask = (gt_gray >= lo) & (gt_gray < hi)
            if mask.sum() > 100:
                brightness_vs_noise.append((lo, hi, res_gray[mask].std()))

        # outlier fraction (defect-pixel-like check)
        outlier_frac = float((np.abs(residual) > 0.25).mean())

    residual_means = np.array(residual_means)
    residual_stds = np.array(residual_stds)
    per_channel_std = np.array(per_channel_std)

    print(f"Analyzed {len(residual_stds)} image pairs\n")
    print(f"Residual mean (bias check, should be ~0 if unbiased): {residual_means.mean():.5f}")
    print(f"Residual std (overall noise level): {residual_stds.mean():.5f} (+/- {residual_stds.std():.5f} across images)")
    print(f"Per-channel std (R,G,B): {per_channel_std.mean(axis=0)}")
    print(f"  -> ratio max/min channel std: {per_channel_std.mean(axis=0).max() / per_channel_std.mean(axis=0).min():.3f}")

    print("\nNoise std by GT-brightness bucket (checks signal-dependence, e.g. Poisson-like):")
    buckets = {}
    for lo, hi, std in brightness_vs_noise:
        buckets.setdefault((lo, hi), []).append(std)
    for (lo, hi), stds in sorted(buckets.items()):
        print(f"  brightness [{lo:.2f}, {hi:.2f}): mean noise std = {np.mean(stds):.5f}  (n={len(stds)} images)")

    print(f"\nOutlier fraction (|residual| > 0.25, i.e. >64/255 off): {outlier_frac:.5f} of pixels in last image checked")
    print("(if this is small but nonzero, that supports the baseline's defect-pixel-correction step)")

if __name__ == "__main__":
    main()