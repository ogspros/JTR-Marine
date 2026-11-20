"""
make_fig2_architecture.py -- Fig. 2: diagram of the U-Net + FiLM architecture.

The drawing is schematic (no input data): encoder and decoder blocks, the
bottleneck, the FiLM modulation box and the skip connections. The axis limits
leave a margin on both sides, so that the INPUT and OUTPUT boxes are not
cropped.

Usage:
  python scripts/figures/make_fig2_architecture.py
Output:
  RESULTS_ROOT/figures/Fig_2.jpg
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import RESULTS_ROOT

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path


OUT_DIR = Path(str(RESULTS_ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
C_DARK    = "#0F2A47"   # navy (boxes input/output, bottleneck)
C_PRIMARY = "#065A82"   # deep blue
C_TEAL    = "#1C7293"
C_LIGHT1  = "#3A8FB0"
C_LIGHT2  = "#7BB6CC"
C_FILM    = "#F2A359"   # orange for FiLM
C_SKIP    = "#F2A359"   # same orange for skip
C_TEXT    = "#0F2A47"


def main():
    # Larger figure with adequate margins; let matplotlib's bbox_inches='tight'
    # snap to content but ensure we have enough padding
    fig, ax = plt.subplots(figsize=(17, 7.5))

    # Coordinate system: extended further to give MORE margin on both sides
    ax.set_xlim(-1.0, 25.0)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    # --- Title ---
    ax.text(12, 9.6, "U-Net + FiLM \u2014 geometry-conditioned seismic reconstruction",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=C_TEXT)

    # --- FiLM modulation box (top center) ---
    film_box = FancyBboxPatch(
        (9.5, 7.5), 5.0, 1.1,
        boxstyle="round,pad=0.05",
        facecolor=C_FILM, edgecolor=C_FILM, linewidth=1.5,
    )
    ax.add_patch(film_box)
    ax.text(12.0, 8.25, "FiLM modulation",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="white")
    ax.text(12.0, 7.85,
            r"$\gamma(L), \beta(L)$ from streamer length $L$",
            ha="center", va="center", fontsize=12, color="white",
            style="italic")

    # --- Skip connections legend (top right) ---
    ax.text(23.0, 8.1, "$- -$ Skip connections",
            ha="center", va="bottom", fontsize=14,
            color=C_FILM, fontweight="bold")

    # --- Encoder boxes (left side, moved right to leave INPUT space) ---
    # Each block: (x_center, height, channels_label)
    encoder_blocks = [
        (3.8,  5.0, "32"),
        (5.0,  4.0, "64"),
        (6.2,  3.0, "128"),
        (7.4,  2.0, "256"),
    ]
    encoder_colors = [C_PRIMARY, C_TEAL, C_LIGHT1, C_LIGHT2]

    encoder_centers = []
    encoder_tops    = []
    for (xc, h, label), color in zip(encoder_blocks, encoder_colors):
        y_bot = 3.5
        y_top = y_bot + h
        w = 0.7
        rect = Rectangle((xc - w/2, y_bot), w, h,
                          facecolor=color, edgecolor=color, linewidth=0)
        ax.add_patch(rect)
        ax.text(xc, y_bot - 0.3, label,
                ha="center", va="top", fontsize=11, color=C_TEXT,
                fontweight="bold")
        encoder_centers.append((xc, (y_bot + y_top)/2))
        encoder_tops.append((xc, y_top))

    # --- Bottleneck (center) ---
    bn_x = 9.5
    bn_y_bot = 4.5
    bn_h = 1.5
    bn_w = 1.0
    rect = Rectangle((bn_x - bn_w/2, bn_y_bot), bn_w, bn_h,
                      facecolor=C_DARK, edgecolor=C_DARK, linewidth=0)
    ax.add_patch(rect)
    ax.text(bn_x, bn_y_bot + bn_h + 0.3, "Bottleneck",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            color=C_TEXT)
    ax.text(bn_x, bn_y_bot - 0.3, "512",
            ha="center", va="top", fontsize=14, fontweight="bold",
            color=C_TEXT)

    # --- Arrow from FiLM to Bottleneck ---
    ax.annotate(
        "",
        xy=(bn_x, bn_y_bot + bn_h),
        xytext=(bn_x, 7.5),
        arrowprops=dict(arrowstyle="->", color=C_FILM, lw=2.2),
    )

    # --- Decoder boxes (right side, mirror of encoder) ---
    decoder_blocks = [
        (11.6, 2.0, "256"),
        (12.8, 3.0, "128"),
        (14.0, 4.0, "64"),
        (15.2, 5.0, "32"),
    ]
    decoder_colors = [C_LIGHT2, C_LIGHT1, C_TEAL, C_PRIMARY]

    decoder_centers = []
    decoder_tops    = []
    for (xc, h, label), color in zip(decoder_blocks, decoder_colors):
        y_bot = 3.5
        y_top = y_bot + h
        w = 0.7
        rect = Rectangle((xc - w/2, y_bot), w, h,
                          facecolor=color, edgecolor=color, linewidth=0)
        ax.add_patch(rect)
        ax.text(xc, y_bot - 0.3, label,
                ha="center", va="top", fontsize=14, color=C_TEXT,
                fontweight="bold")
        decoder_centers.append((xc, (y_bot + y_top)/2))
        decoder_tops.append((xc, y_top))

    # --- INPUT box (left of encoder, with proper margin) ---
    in_box = FancyBboxPatch(
        (0.2, 4.5), 1.8, 1.5,
        boxstyle="round,pad=0.08",
        facecolor="white", edgecolor=C_TEXT, linewidth=1.5,
    )
    ax.add_patch(in_box)
    ax.text(1.1, 5.6, "INPUT",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_TEXT)
    ax.text(1.1, 5.05, "masked\ngather",
            ha="center", va="center", fontsize=12, color=C_TEXT)
    # Arrow from INPUT to first encoder block (now with gap)
    ax.annotate(
        "",
        xy=(encoder_blocks[0][0] - 0.4, encoder_centers[0][1]),
        xytext=(2.05, encoder_centers[0][1]),
        arrowprops=dict(arrowstyle="->", color=C_TEXT, lw=1.5),
    )
    # Shape label (2, T, X) above first encoder block
    ax.text(encoder_blocks[0][0], encoder_tops[0][1] + 0.25,
            "(2, T, X)", ha="center", va="bottom", fontsize=9,
            color=C_TEXT, style="italic")

    # --- OUTPUT box (right of decoder, with proper margin) ---
    out_box = FancyBboxPatch(
        (16.4, 4.5), 1.9, 1.5,
        boxstyle="round,pad=0.08",
        facecolor="white", edgecolor=C_TEXT, linewidth=1.5,
    )
    ax.add_patch(out_box)
    ax.text(17.35, 5.6, "OUTPUT",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_TEXT)
    ax.text(17.35, 5.05, "reconstructed\ngather",
            ha="center", va="center", fontsize=12, color=C_TEXT)
    # Arrow from last decoder block to OUTPUT
    ax.annotate(
        "",
        xy=(16.35, decoder_centers[-1][1]),
        xytext=(decoder_blocks[-1][0] + 0.4, decoder_centers[-1][1]),
        arrowprops=dict(arrowstyle="->", color=C_TEXT, lw=1.5),
    )
    # Shape label (1, T, X) above last decoder block
    ax.text(decoder_blocks[-1][0], decoder_tops[-1][1] + 0.25,
            "(1, T, X)", ha="center", va="bottom", fontsize=9,
            color=C_TEXT, style="italic")

    # --- Skip connections (routed below the FiLM box) ---
    # skip lines go up from the top of the encoder block to a fixed height,
    # across, then down to the top of the decoder block
    skip_pairs = [
        (encoder_tops[0], decoder_tops[3]),
        (encoder_tops[1], decoder_tops[2]),
        (encoder_tops[2], decoder_tops[1]),
        (encoder_tops[3], decoder_tops[0]),
    ]
    # One height per skip connection. The first assignment is overwritten by
    # the second one, which is the one used: all heights are below the base of
    # the FiLM box (x 9.5-14.5, y 7.5-8.6), so the lines do not cross it.
    skip_heights = [8.85, 8.50, 8.15, 7.80]
    skip_heights = [6.95, 6.7, 6.45, 6.2]  # all below FiLM box

    for (enc_top, dec_top), h in zip(skip_pairs, skip_heights):
        x_e, y_e = enc_top
        x_d, y_d = dec_top
        # Vertical from encoder top up to height h, then horizontal to decoder x, then down
        ax.plot([x_e, x_e], [y_e, h], "--", color=C_SKIP, lw=1.3, alpha=0.85)
        ax.plot([x_e, x_d], [h, h],   "--", color=C_SKIP, lw=1.3, alpha=0.85)
        ax.plot([x_d, x_d], [h, y_d], "--", color=C_SKIP, lw=1.3, alpha=0.85)

    # --- Encoder / Decoder labels (below) ---
    ax.text(5.6, 2.6, "Encoder (4 levels)",
            ha="center", va="center", fontsize=14, color=C_TEXT,
            style="italic", fontweight="bold")
    ax.text(13.4, 2.6, "Decoder (4 levels)",
            ha="center", va="center", fontsize=14, color=C_TEXT,
            style="italic", fontweight="bold")

    # --- Bottom footer ---
    ax.text(12, 1.4,
            "8.36M parameters  |  Encoder/Decoder: 4 levels (32 → 256 channels)  |  "
            "InstanceNorm  |  Anisotropic kernel (3, 5) at level 1  |  "
            "Composite loss: L1 + 0.3$\\cdot$SSIM + 0.1$\\cdot$f-k",
            ha="center", va="center", fontsize=12, color=C_TEXT,
            style="italic")

    # Save with generous bbox padding to prevent any cropping
    out_path = OUT_DIR / "Fig_2.jpg"
    plt.savefig(out_path, dpi=300, bbox_inches="tight",
                  pad_inches=0.5, facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
