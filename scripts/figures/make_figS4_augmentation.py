"""
make_figS4_augmentation.py -- Fig. S4: training-time augmentations and loss
variants of the compact U-Net.

No file is read: the bar heights are the best validation SNR_miss (30 %
random masking), taken from the training logs and written in the code:
    reference (L1 + SSIM + f-k)        6.91   CKPT_ROOT/v6_grid25m
    no FiLM                            6.47   RESULTS_ROOT/no_film/summary.json
    + Charbonnier                      6.91   RESULTS_ROOT/v7b_baseline/summary.json
    + Charbonnier + curriculum         6.79   RESULTS_ROOT/v7b_full/summary.json
    + noise, fixed SNR (25 dB)         6.90   RESULTS_ROOT/v8_baseline_snr25/summary.json
    + noise, variable SNR (15-30 dB)   6.97   RESULTS_ROOT/v8_full_snr15_30/summary.json

Output: RESULTS_ROOT/figures/Fig_S4.jpg

Usage:
  python scripts/figures/make_figS4_augmentation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import RESULTS_ROOT

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(str(RESULTS_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_DEEP   = "#065A82"
C_TEAL   = "#1C7293"
C_RED    = "#C44536"
C_GREEN  = "#5E8C61"
C_ORANGE = "#F2A359"
C_LIGHT  = "#9EB3C2"
C_NAVY   = "#0F2A47"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.grid":       True,
    "grid.alpha":      0.3,
})


def figS4_augmentation():
    print("Fig. S4: augmentation...")
    # All values are the BEST VALIDATION SNR (not the test SNR),
    # taken from the training logs and summary files
    data = [
        ("v6 baseline\n(L1 + SSIM + f-k)",       6.91),
        ("v6 no FiLM",                            6.47),
        ("+ Charbonnier",                          6.91),
        ("+ Charbonnier\n+ curriculum",           6.79),
        ("+ Noise (25 dB)",                        6.90),
        ("+ Noise (15-30 dB)",                     6.97),
    ]
    labels = [d[0] for d in data]
    vals   = [d[1] for d in data]

    fig, ax = plt.subplots(1, 1, figsize=(11, 5))

    baseline = 6.91
    colors_b = []
    for v in vals:
        if v > baseline + 0.05:
            colors_b.append(C_GREEN)
        elif v < baseline - 0.05:
            colors_b.append(C_RED)
        else:
            colors_b.append(C_TEAL)

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, vals, color=colors_b, edgecolor=C_NAVY, linewidth=1)

    for bar, v in zip(bars, vals):
        delta = v - baseline
        delta_str = f"\nΔ={delta:+.2f}" if abs(delta) > 0.01 else ""
        ax.text(bar.get_x() + bar.get_width()/2,
                v + 0.03, f"{v:.2f}{delta_str}",
                ha="center", fontsize=10, color=C_NAVY)

    ax.axhline(y=baseline, color=C_NAVY, linestyle="--", linewidth=1.2,
               label=f"baseline (v6) = {baseline:.2f} dB")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Best validation SNR (dB)")
    ax.set_title(
        "Ablation: training-time augmentations and loss variants",
        fontweight="bold",
    )
    ax.set_ylim(5.5, 7.4)
    ax.legend(loc="lower left", fontsize=9)

    # Interpretation note
    ax.text(0.02, 0.97,
            "All variants within ±0.5 dB of baseline:\n"
            "training-time augmentations do not improve\n"
            "performance on the available real data",
            transform=ax.transAxes, fontsize=9.5, color=C_NAVY,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor=C_LIGHT, alpha=0.85))

    plt.tight_layout()
    out = OUT_DIR / "Fig_S4.jpg"
    plt.savefig(out)
    plt.close()
    print(f"  saved {out}")


if __name__ == "__main__":
    figS4_augmentation()
