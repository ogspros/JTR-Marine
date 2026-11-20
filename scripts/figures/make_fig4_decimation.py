"""
make_fig4_decimation.py -- Fig. 4: SNR versus decimation ratio for DL, MSSA
and POCS (random masking, internal test set).

Layout: two panels side by side (SNR_full on the left, SNR_miss on the
right), each with three curves (DL, MSSA, POCS) against the decimation ratio;
the 95 % bootstrap confidence intervals are drawn as shaded bands.

All numbers are read from statistics_3methods.json (MSSA with rank=10,
n_iter=3; POCS with n_iter=100, threshold from 99 to 1 %, exponential decay).
A numeric summary of SNR_miss is also printed.

Input:   RESULTS_ROOT/statistics/statistics_3methods.json
Output:  RESULTS_ROOT/figures/Fig_4.png

Usage:
  python scripts/figures/make_fig4_decimation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import RESULTS_ROOT

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STATS_PATH = str(RESULTS_ROOT / "statistics/statistics_3methods.json")
OUT_PATH = str(RESULTS_ROOT / "figures/Fig_4.png")

# Same method colours as in the gather figure (Fig. 7)
COLOR_DL   = "#065A82"
COLOR_MSSA = "#3E8E5A"
COLOR_POCS = "#C44536"

RATIOS_PCT = [20, 30, 40, 50, 60, 70, 80]


def main():
    s = json.load(open(STATS_PATH))
    data3way = s["3way"]["random"]

    # Extract the series (mean and 95 % CI per ratio) of one metric
    def get_series(metric_key):
        means_dl, ci_dl_low, ci_dl_hi = [], [], []
        means_mssa, ci_mssa_low, ci_mssa_hi = [], [], []
        means_pocs, ci_pocs_low, ci_pocs_hi = [], [], []
        for r in RATIOS_PCT:
            rkey = f"{r}pct"
            d = data3way[rkey]
            for mlist, lo_list, hi_list, key in [
                (means_dl, ci_dl_low, ci_dl_hi, f"{metric_key}_dl"),
                (means_mssa, ci_mssa_low, ci_mssa_hi, f"{metric_key}_mssa"),
                (means_pocs, ci_pocs_low, ci_pocs_hi, f"{metric_key}_pocs"),
            ]:
                e = d[key]
                mlist.append(e["mean"])
                ci = e["ci95_bootstrap"]
                lo_list.append(ci[0])
                hi_list.append(ci[1])
        return dict(
            dl=(np.array(means_dl), np.array(ci_dl_low), np.array(ci_dl_hi)),
            mssa=(np.array(means_mssa), np.array(ci_mssa_low), np.array(ci_mssa_hi)),
            pocs=(np.array(means_pocs), np.array(ci_pocs_low), np.array(ci_pocs_hi)),
        )

    full = get_series("snr_full")
    miss = get_series("snr_miss")

    # Render figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                              gridspec_kw=dict(wspace=0.22))

    titles = ["SNR$_{full}$ (full gather, trace-replaced)",
              "SNR$_{miss}$ (missing positions only)"]
    series = [full, miss]
    x = np.array(RATIOS_PCT)

    for ax, t, ser in zip(axes, titles, series):
        # MSSA fill + line
        m, lo, hi = ser["mssa"]
        ax.fill_between(x, lo, hi, color=COLOR_MSSA, alpha=0.22)
        ax.plot(x, m, "-o", color=COLOR_MSSA, label="MSSA (rank=10, $n_{iter}$=3)",
                markersize=7, linewidth=2.0)
        # POCS fill + line
        m, lo, hi = ser["pocs"]
        ax.fill_between(x, lo, hi, color=COLOR_POCS, alpha=0.22)
        ax.plot(x, m, "-s", color=COLOR_POCS, label="POCS ($n_{iter}$=100, $t$=99→1)",
                markersize=7, linewidth=2.0)
        # DL fill + line (drawn last, so it is on top)
        m, lo, hi = ser["dl"]
        ax.fill_between(x, lo, hi, color=COLOR_DL, alpha=0.22)
        ax.plot(x, m, "-D", color=COLOR_DL, label="Deep learning",
                markersize=7, linewidth=2.5)

        # Zero line (drawn in both panels)
        ax.axhline(0, color="k", linestyle=":", linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Decimation ratio (%)", fontsize=11)
        ax.set_ylabel("SNR (dB)", fontsize=11)
        ax.set_title(t, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.suptitle(
        "DL vs MSSA vs POCS — random masking, internal test set "
        "(n=276 gathers, 95% bootstrap CI)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2,
                  facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")

    # Print numeric summary
    print("\n" + "=" * 72)
    print("Numeric summary (from statistics_3methods.json)")
    print("=" * 72)
    print(f"{'Ratio':<6} {'DL_miss':>14} {'MSSA_miss':>14} {'POCS_miss':>14}")
    for i, r in enumerate(RATIOS_PCT):
        dl = miss["dl"][0][i]
        ms = miss["mssa"][0][i]
        po = miss["pocs"][0][i]
        print(f"{r}%    {dl:>+11.2f} dB  {ms:>+11.2f} dB  {po:>+11.2f} dB")


if __name__ == "__main__":
    main()
