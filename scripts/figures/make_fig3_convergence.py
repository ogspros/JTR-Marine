"""
make_fig3_convergence.py -- Fig. 3: training convergence of the compact U-Net
with and without FiLM (training loss and validation SNR per epoch).

Inputs:  CKPT_ROOT/v6_grid25m/within_geology/history.json
         RESULTS_ROOT/no_film/training_log.json
Output:  RESULTS_ROOT/figures/Fig_3.jpg

Usage:
  python scripts/figures/make_fig3_convergence.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT, RESULTS_ROOT

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT_DIR = Path(str(RESULTS_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette shared by the figure scripts
C_DEEP   = "#065A82"   # primary blue (DL FiLM)
C_TEAL   = "#1C7293"   # secondary blue
C_RED    = "#C44536"   # POCS / negative
C_GREEN  = "#5E8C61"   # positive
C_ORANGE = "#F2A359"   # accent
C_LIGHT  = "#9EB3C2"   # light gray-blue
C_NAVY   = "#0F2A47"   # dark

# Matplotlib style
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
})


def fig3_convergence():
    print("Fig. 3: convergence...")
    # v6 FiLM history (CKPT_ROOT/v6_grid25m/within_geology/history.json)
    film_history = json.loads(
        Path(str(CKPT_ROOT / "v6_grid25m/within_geology/history.json")).read_text()
    )
    # no-FiLM history (RESULTS_ROOT/no_film/training_log.json)
    nofilm_history = json.loads(
        Path(str(RESULTS_ROOT / "no_film/training_log.json")).read_text()
    )

    # Extract epoch, train_loss, val_snr
    def extract(history):
        epochs    = [h["epoch"]      for h in history]
        train_l   = [h["train_loss"] for h in history]
        # the validation SNR may be stored under different keys
        val_snr   = []
        for h in history:
            v = h.get("val_snr_db_mean",
                       h.get("val_snr",
                              h.get("val_snr_db", 0.0)))
            val_snr.append(v)
        return np.array(epochs), np.array(train_l), np.array(val_snr)

    e1, l1, s1 = extract(film_history)
    e2, l2, s2 = extract(nofilm_history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Train loss
    ax1.plot(e1, l1, color=C_DEEP, linewidth=2,    label="U-Net + FiLM (v6)")
    ax1.plot(e2, l2, color=C_RED,  linewidth=2,
             linestyle="--", label="U-Net (no FiLM)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Composite training loss")
    ax1.set_title("Training loss")
    ax1.legend(loc="upper right")
    ax1.set_yscale("log")

    # Val SNR
    ax2.plot(e1, s1, color=C_DEEP, linewidth=2,    label="U-Net + FiLM (v6)")
    ax2.plot(e2, s2, color=C_RED,  linewidth=2,
             linestyle="--", label="U-Net (no FiLM)")
    # Annotate best
    best_i1 = int(np.argmax(s1))
    ax2.scatter([e1[best_i1]], [s1[best_i1]],
                color=C_DEEP, s=80, marker="*", zorder=5)
    ax2.annotate(f"best = {s1[best_i1]:.2f} dB",
                 xy=(e1[best_i1], s1[best_i1]),
                 xytext=(e1[best_i1]-50, s1[best_i1]-1.0),
                 fontsize=10, color=C_DEEP)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation SNR (dB)")
    ax2.set_title("Validation SNR")
    ax2.legend(loc="lower right")

    fig.suptitle("Training convergence — multi-geometry within-survey setup",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "Fig_3.jpg"
    plt.savefig(out)
    plt.close()
    print(f"  saved {out}")


if __name__ == "__main__":
    fig3_convergence()
