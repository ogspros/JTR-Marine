"""
make_fig5_generalization.py -- Fig. 5: generalization of the compact U-Net,
held-out folds (left) and cross-survey test on the three held-out lines
(right).

Left panel: the bars are the per-fold values stored in
cross_geometry_LOO_for_paper.json (key snr_db.per_fold). 

The horizontal reference lines use values written in the code: 6.91 dB is
the best validation SNR_miss (30 % random) of the paper checkpoint; 10.16 /
14.20 / 14.87 dB are its internal-test SNR_full at 30 % random for the 1500,
3000 and 4500 m geometries. The caption quotes 6.91 and
10.2 / 14.2 / 14.9 dB.

Inputs:  RESULTS_ROOT/v6_grid25m/cross_geometry_LOO_for_paper.json
         RESULTS_ROOT/cross_survey_v6/cross_survey_v6_final.json
Output:  RESULTS_ROOT/figures/Fig_5.jpg

Usage:
  python scripts/figures/make_fig5_generalization.py
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


def fig5_generalization():
    print("Fig. 5: generalization...")
    loo = json.loads(
        Path(str(RESULTS_ROOT / "v6_grid25m/cross_geometry_LOO_for_paper.json")).read_text()
    )
    cs = json.loads(
        Path(str(RESULTS_ROOT / "cross_survey_v6/cross_survey_v6_final.json")).read_text()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ────── Left: per-fold values of the JSON (train.py --fold k) ──────
    per_fold = loo["snr_db"]["per_fold"]
    mean_loo = loo["snr_db"]["mean"]
    std_loo  = loo["snr_db"]["std"]

    # Tick labels of the three bars, kept as in the published figure: one
    # geometry per bar (1500, 3000, 4500 m). Each fold of train.py holds out
    # one line, not one geometry (see the module docstring).
    geom_labels = ["1500 m\n(SISOC-MEPA + CG-1)",
                   "3000 m\n(IT90-65 + IT89-33)",
                   "4500 m\n(CROP-M31 + CROP-M32)"]

    x_pos = np.arange(len(per_fold))
    bars = ax1.bar(x_pos, per_fold, color=C_TEAL, edgecolor=C_NAVY,
                   linewidth=1)
    for bar, val in zip(bars, per_fold):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 val + 0.15, f"{val:+.2f} dB",
                 ha="center", fontsize=10, color=C_NAVY)
    ax1.axhline(y=mean_loo, color=C_RED, linestyle="--", linewidth=1.5,
                label=f"LOO mean = {mean_loo:+.2f} dB")
    # 6.91 dB = best validation SNR_miss (30 % random masking) of the paper
    # checkpoint (CKPT_ROOT/v6_grid25m/within_geology), quoted in the caption.
    ax1.axhline(y=6.91, color=C_GREEN, linestyle=":", linewidth=1.5,
                label="in-distribution = 6.91 dB")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(geom_labels, fontsize=10)
    ax1.set_xlabel("Held-out geometry (training on other two)")
    ax1.set_ylabel("Test SNR (dB)")
    ax1.set_title("Leave-one-geometry-out", fontweight="bold")
    ax1.legend(loc="upper right", bbox_to_anchor=(1,1.01), fontsize=9)
    ax1.set_ylim(0, 8)

    # ────── Right: cross-survey ──────
    ds_order  = ["CG03", "IT245", "CROP_M18"]
    ds_order2  = ["CG-3", "IT97-245", "CROP-M18"]
    geoms     = ["1500m", "3000m", "4500m"]
    ratios_cs = ["30pct", "40pct", "50pct"]
    ratio_labels = ["30% missing", "40% missing", "50% missing"]
    colors_r  = [C_DEEP, C_TEAL, C_LIGHT]

    x_pos = np.arange(len(ds_order))
    bar_w = 0.27

    for i, (rk, rl, col) in enumerate(zip(ratios_cs, ratio_labels, colors_r)):
        means = [cs[ds]["by_ratio"][rk]["snr_full"]["mean"] for ds in ds_order]
        stds  = [cs[ds]["by_ratio"][rk]["snr_full"]["std"]  for ds in ds_order]
        ax2.bar(x_pos + i*bar_w - bar_w,
                means, bar_w, yerr=stds, capsize=4,
                color=col, label=rl, edgecolor=C_NAVY, linewidth=0.5)

    labels_cs = [f"{geom}\n({ds})" for ds, geom in zip(ds_order2, geoms)]
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels_cs)
    ax2.set_xlabel("Cross-survey dataset (never seen in training)")
    ax2.set_ylabel("SNR (dB) — with trace replacement")
    ax2.set_title("Cross-survey: 3 unseen datasets", fontweight="bold")
    # In-distribution reference per geometry (1500, 3000, 4500 m): internal-test
    # SNR_full at 30 % random masking of the paper checkpoint, with trace
    # replacement (caption: 10.2 / 14.2 / 14.9 dB).
    indist_per_geom = [10.16, 14.20, 14.87]   # 1500m, 3000m, 4500m
    for i, (xp, yv) in enumerate(zip(x_pos, indist_per_geom)):
        ax2.hlines(yv, xp - 0.45, xp + 0.45,
                   color=C_GREEN, linestyle=":", linewidth=1.8,
                   label="in-distribution\n(same geometry)" if i == 0 else None)

    ax2.legend(loc="upper center", bbox_to_anchor=(0.15, 1.015), fontsize=9)
    ax2.set_ylim(0, 16)

    fig.suptitle("Generalization: held-out geometry & unseen surveys",
                 fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "Fig_5.jpg"
    plt.savefig(out)
    plt.close()
    print(f"  saved {out}")


if __name__ == "__main__":
    fig5_generalization()
