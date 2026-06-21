"""Vytvoří malý norm_stats.npz jen s normalizačními statistikami.

App nepotřebuje celých 185 MB processed.npz — jen statistiky (pár čísel).
Spusť jednou:  python scripts/extract_stats.py
"""

import numpy as np
import os

PROCESSED = "data/processed.npz"
OUTPUT = "models/norm_stats.npz"

print(f"Načítám statistiky z {PROCESSED}...")
data = np.load(PROCESSED)

os.makedirs("models", exist_ok=True)
np.savez(
    OUTPUT,
    X_mean=data["X_mean"],
    X_std=data["X_std"],
    Y_mean=data["Y_mean"],
    Y_std=data["Y_std"],
)

size_kb = os.path.getsize(OUTPUT) / 1024
print(f"Uloženo: {OUTPUT} ({size_kb:.1f} KB)")
print("\nStatistiky:")
for key in ["X_mean", "X_std", "Y_mean", "Y_std"]:
    print(f"  {key}: shape {data[key].shape}")
