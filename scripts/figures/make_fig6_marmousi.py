#!/usr/bin/env python3
"""
make_fig6_marmousi.py -- Fig. 6: zoomed panels and a single-trace comparison
on the Marmousi synthetic (exact reference).

Layout
------
Row 1   ground truth, masked input, DL, MSSA, POCS (common clip, recorded window);
        the shallow and deep zoom windows are outlined on the ground truth.
Row 2   the shallow window, enlarged, for the same five panels.
Row 3   the deep window, enlarged.
Row 4   one missing trace, wiggle overlay of truth and the three methods, in
        the shallow (left) and deep (right) windows.

Choice of the displayed trace
-----------------------------
To avoid showing a favourable case, the missing trace is chosen automatically
as the one whose DL correlation with the truth is closest to the median over
the missing traces of the shot. The script also prints SNR_miss of every saved
shot, so that a representative shot can be chosen and declared in the caption.

Input:  the .npz saved by scripts/evaluate/eval_marmousi.py --save-examples
        RESULTS_ROOT/marmousi/examples/MARMOUSI_1500_random_<R>pct_shot<SSS>.npz
Output: RESULTS_ROOT/figures/Fig_6.{png,pdf}

Usage
-----
    python scripts/figures/make_fig6_marmousi.py --list        # SNR of saved shots
    python scripts/figures/make_fig6_marmousi.py --shot 120 --ratio 50
    python scripts/figures/make_fig6_marmousi.py --shot 120 --ratio 50 \\
        --shallow 0.6 1.3 --deep 1.7 2.5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import RESULTS_ROOT

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

T_REC_S = 2.9
DX = 25.0
METHODS = [("dl", "DL", "tab:red"),
           ("mssa", "MSSA", "tab:blue"),
           ("pocs", "POCS", "tab:green")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9})


def snr_miss(pred, tgt, miss):
    e = ((pred[:, miss] - tgt[:, miss]) ** 2).sum()
    s = (tgt[:, miss] ** 2).sum()
    return 10 * np.log10(s / max(e, 1e-30))


def load(path):
    d = np.load(path)
    dt = float(d["dt"])
    n = int(round(T_REC_S / dt))
    out = {k: d[k][:n] for k in ("target", "masked", "dl", "mssa", "pocs")}
    out["mask"], out["valid"], out["dt"] = d["trace_mask"], d["validity"], dt
    out["miss"] = np.flatnonzero((1 - out["mask"]) * out["valid"] > 0)
    out["cols"] = np.flatnonzero(out["valid"] > 0)
    return out


def list_shots(ex_dir, ratio):
    files = sorted(Path(ex_dir).glob(f"*_{ratio}pct_shot*.npz"))
    print(f"Saved shots at {ratio}% (SNR_miss on the recorded window):")
    for f in files:
        g = load(f)
        vals = "  ".join(f"{lab} {snr_miss(g[m], g['target'], g['miss']):5.2f}"
                         for m, lab, _ in METHODS)
        print(f"  {f.stem:40s} {vals}")


def pick_trace(g):
    tgt, dl, miss = g["target"], g["dl"], g["miss"]
    corr = np.array([np.corrcoef(dl[:, j], tgt[:, j])[0, 1] for j in miss])
    return miss[np.argmin(np.abs(corr - np.median(corr)))], np.median(corr)


def panel(ax, img, cols, t, clip, title=None, tlim=None, xlabel=True):
    ext = [cols[0] * DX, cols[-1] * DX, t[-1], t[0]]
    ax.imshow(img[:, cols[0]:cols[-1] + 1], aspect="auto", cmap="gray_r",
              vmin=-clip, vmax=clip, extent=ext, interpolation="bilinear")
    if tlim:
        ax.set_ylim(tlim[1], tlim[0])
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel("offset (m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=str(RESULTS_ROOT / "marmousi/examples"))
    ap.add_argument("--dataset", default="MARMOUSI_1500")
    ap.add_argument("--shot", type=int, default=120)
    ap.add_argument("--ratio", type=int, default=50)
    ap.add_argument("--shallow", nargs=2, type=float, default=[0.6, 1.3])
    ap.add_argument("--deep", nargs=2, type=float, default=[1.7, 2.5])
    ap.add_argument("--trace", type=int, default=None,
                    help="grid column of the trace to show (default: automatic)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_ROOT / "figures/Fig_6"))
    args = ap.parse_args()

    if args.list:
        list_shots(args.examples, args.ratio)
        return

    path = Path(args.examples) / f"{args.dataset}_random_{args.ratio}pct_shot{args.shot:03d}.npz"
    g = load(path)
    t = np.arange(g["target"].shape[0]) * g["dt"]
    cols = g["cols"]
    clip = np.percentile(np.abs(g["target"][:, cols]), 99)

    j, med_corr = (args.trace, None) if args.trace is not None else pick_trace(g)
    imgs = [("Ground truth", g["target"]), ("Masked input", g["masked"])] + \
           [(lab, g[m]) for m, lab, _ in METHODS]

    fig = plt.figure(figsize=(12, 13.5))
    gs = fig.add_gridspec(4, 5, height_ratios=[1.35, 0.8, 0.8, 0.9],
                          hspace=0.42, wspace=0.12)

    # rows 1-3: full panels and zooms
    for k, (title, img) in enumerate(imgs):
        for row, tlim, ttl in ((0, None, title),
                               (1, args.shallow, None),
                               (2, args.deep, None)):
            ax = fig.add_subplot(gs[row, k])
            panel(ax, img, cols, t, clip, ttl, tlim, xlabel=True)
            if k == 0:
                ax.set_ylabel("time (s)")
            else:
                ax.set_yticklabels([])
            if row == 0 and k == 0:
                for (a, b), lab in ((args.shallow, "A"), (args.deep, "B")):
                    ax.add_patch(Rectangle((cols[0] * DX, a),
                                           (cols[-1] - cols[0]) * DX, b - a,
                                           fill=False, ec="tab:orange", lw=1.2))
                    ax.text(cols[-1] * DX, a, f" {lab}", color="tab:orange",
                            va="top", ha="left", fontsize=10, fontweight="bold")
            if row > 0 and k == 0:
                ax.text(-0.42, 0.5, "A" if row == 1 else "B",
                        transform=ax.transAxes, color="tab:orange",
                        fontsize=12, fontweight="bold", va="center")

    # row 4: single-trace overlays in the two windows
    tgt = g["target"][:, j]
    sub = gs[3, :].subgridspec(1, 2, wspace=0.12)
    for c, (a, b), lab in ((0, args.shallow, "A"), (1, args.deep, "B")):
        ax = fig.add_subplot(sub[0, c])
        sel = (t >= a) & (t <= b)
        ax.plot(t[sel], tgt[sel], color="black", lw=1.8, label="ground truth")
        for m, labm, col in METHODS:
            rec = g[m][:, j]
            r = np.corrcoef(rec, tgt)[0, 1]
            rr = np.sqrt((rec ** 2).mean() / max((tgt ** 2).mean(), 1e-30))
            ax.plot(t[sel], rec[sel], color=col, lw=1.0,
                    label=f"{labm}  (r = {r:.2f}, RMS ratio = {rr:.2f})")
        ax.set_xlim(a, b)
        ax.set_xlabel("time (s)")
        ax.set_title(f"Window {lab}: missing trace at offset {j * DX:.0f} m")
        if c == 0:
            ax.set_ylabel("normalised amplitude")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.075))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"shot {args.shot}, {args.ratio}% nominal decimation, "
          f"{len(g['miss'])} missing traces")
    for m, lab, _ in METHODS:
        print(f"  {lab:5s} SNR_miss = {snr_miss(g[m], g['target'], g['miss']):5.2f} dB")
    print(f"  displayed trace: grid column {j} (offset {j * DX:.0f} m)"
          + (f", DL correlation closest to the shot median ({med_corr:.2f})"
             if med_corr is not None else ""))
    print(f"  saved {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
