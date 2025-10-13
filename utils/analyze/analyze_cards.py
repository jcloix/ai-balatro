# utils/analyze/analyze_cards.py
from PIL import Image
import numpy as np
import os

FOLDER = "data/Negative"
SAMPLE_PIXELS = 5000
TOLERANCE = 15

color_bins = ["white", "red", "green", "blue", "yellow"]

def process_pair(orig_path, neg_path):
    orig = np.array(Image.open(orig_path).convert("RGB"))
    aug  = np.array(Image.open(neg_path).convert("RGB"))
    
    orig_flat = orig.reshape(-1,3)
    aug_flat  = aug.reshape(-1,3)
    
    if SAMPLE_PIXELS < orig_flat.shape[0]:
        idx = np.random.choice(orig_flat.shape[0], SAMPLE_PIXELS, replace=False)
        orig_flat = orig_flat[idx]
        aug_flat  = aug_flat[idx]
    
    # Define color bins
    masks = {
        "white": (orig_flat[:,0]>200) & (orig_flat[:,1]>200) & (orig_flat[:,2]>200),
        "red":   (orig_flat[:,0]>200) & (orig_flat[:,1]<100) & (orig_flat[:,2]<100),
        "green": (orig_flat[:,0]<100) & (orig_flat[:,1]>200) & (orig_flat[:,2]<100),
        "blue":  (orig_flat[:,0]<100) & (orig_flat[:,1]<100) & (orig_flat[:,2]>200),
        "yellow":(orig_flat[:,0]>200) & (orig_flat[:,1]>200) & (orig_flat[:,2]<100)
    }
    
    summary = {}
    for color, mask in masks.items():
        n_pixels = mask.sum()
        if n_pixels == 0:
            continue
        orig_bin = orig_flat[mask]
        aug_bin  = aug_flat[mask]
        inv_bin = 255 - orig_bin
        delta = np.abs(inv_bin - aug_bin)
        fully_inv_pct = (delta <= TOLERANCE).sum() / n_pixels * 100
        mean_aug = aug_bin.mean(axis=0).astype(int)
        summary[color] = {"pixels": n_pixels, "fully_inverted_pct": fully_inv_pct, "mean_aug_rgb": mean_aug}
    return summary

# Build pairs
all_files = os.listdir(FOLDER)
original_files = [f for f in all_files if "_negative" not in f]

# Process each pair and print per image
for f in original_files:
    orig_path = os.path.join(FOLDER, f)
    name, ext = os.path.splitext(f)
    neg_name = f"{name}_negative{ext}"
    neg_path = os.path.join(FOLDER, neg_name)
    if os.path.exists(neg_path):
        summary = process_pair(orig_path, neg_path)
        print(f"=== {f} vs {neg_name} ===")
        for color, stats in summary.items():
            print(f"{color:7s}: pixels={stats['pixels']:6d}, fully inverted ≈ {stats['fully_inverted_pct']:.2f}%")
            print(f"        mean augmented RGB: {stats['mean_aug_rgb']}")
        print()
    else:
        print(f"[WARN] Missing negative for {f}")
