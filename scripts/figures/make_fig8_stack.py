#!/usr/bin/env python3
"""
make_fig8_stack.py -- Fig. 8: stacked sections of CROP-M18 (ground truth,
masked input, DL, MSSA, POCS) and stack SNR table.

1. Enlarged windows. Row 1 shows the five sections (ground truth, masked input,
   DL, MSSA, POCS); boxes A and B on the ground truth mark two windows that are
   enlarged in rows 2 and 3, for all five panels (same layout as the Marmousi
   figure).

2. Output format. Saved as PDF and PNG (300 dpi) rather than JPEG, because
   JPEG compression blurs thin seismic events. In the PDF the sections are
   embedded at their native resolution. No suptitle (the caption does that).

3. SNR variants. The paper's stack SNR is computed over the whole section,
   after a 1-s AGC whose gain is derived from EACH section separately. Because
   an AGC derived from each section partly compensates the attenuated
   amplitudes of the classical methods, SNR vs ground truth is also computed
   with:
     - "own AGC"   : each section with its own gain (the values in the paper)
     - "GT gain"   : the gain of the ground-truth section applied to all
                     sections (identical gain function for every panel)
     - "no AGC"    : raw stacked amplitudes
   All over the whole section, or over a time window with --snr-tmin/--snr-tmax.

4. Extra reconstructions (benchmark Chai et al. 2020). --extra LABEL=PATH adds
   one stacked section per option, shown after DL and included in the SNR table,
   e.g. --extra "Chai+loss=output_for_stack_chai_lossonly/CROP_M18_STK_50_CHAI_LOSS.sgy".
   Each extra section must have the same size and sampling interval as the
   ground truth.

Inputs:  RESULTS_ROOT/output_for_stack/<prefix><ratio>_{GT,MASK,DL,MSSA,POCS}_STK.sgy
         (--stack-dir, --prefix, --ratio; default CROP_M18_A50_*_STK.sgy)
Output:  RESULTS_ROOT/figures/Fig_8.{pdf,png} (--out)

Usage
-----
    python scripts/figures/make_fig8_stack.py                     # auto windows
    python scripts/figures/make_fig8_stack.py --zoomA 100 300 0.8 1.6 --zoomB 400 600 2.0 3.0
    python scripts/figures/make_fig8_stack.py --snr-only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import RESULTS_ROOT

import argparse
from pathlib import Path

import numpy as np
import segyio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator

STACK_DIR = Path(str(RESULTS_ROOT / "output_for_stack"))
METHODS = [("GT", "Ground truth"), ("MASK", "Masked input"),
           ("DL", "Compact U-Net"), ("MSSA", "MSSA"), ("POCS", "POCS")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9})


def load_stack_sgy(path):
    with segyio.open(path, "r", ignore_geometry=True) as f:
        data = np.stack([f.trace[i] for i in range(f.tracecount)], axis=1)
        dt_ms = f.bin[segyio.BinField.Interval] / 1000.0
    return data.astype(np.float32), dt_ms


def agc_rms(section, dt_ms, window_s=1.0):
    """Moving-window RMS along time (centred), cumulative sum in float64."""
    n_t, n_x = section.shape
    half = int(round(window_s * 1000.0 / dt_ms)) // 2
    csum = np.concatenate([np.zeros((1, n_x), np.float64),
                           np.cumsum(section.astype(np.float64) ** 2, axis=0)])
    idx = np.arange(n_t)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n_t, idx + half + 1)
    return np.sqrt((csum[hi] - csum[lo]) / (hi - lo)[:, None]).astype(np.float32)


def agc_rms_float32_original(section, dt_ms, window_s=1.0):
    """Same moving-window RMS with the cumulative sum in float32 (the original
    implementation); used only by --agc-check."""
    n_t, n_x = section.shape
    half = int(round(window_s * 1000.0 / dt_ms)) // 2
    csum = np.cumsum(section.astype(np.float32) ** 2, axis=0)
    csum = np.concatenate([np.zeros((1, n_x), np.float32), csum], axis=0)
    idx = np.arange(n_t)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n_t, idx + half + 1)
    return np.sqrt((csum[hi] - csum[lo]) / (hi - lo)[:, None].astype(np.float32))


def agc_rms_direct(section, dt_ms, window_s=1.0):
    """Independent check: direct convolution in float64 (no cumulative sum)."""
    n_t, n_x = section.shape
    win = 2 * (int(round(window_s * 1000.0 / dt_ms)) // 2) + 1
    k = np.ones(win)
    count = np.convolve(np.ones(n_t), k, mode="same")
    sq = section.astype(np.float64) ** 2
    out = np.empty_like(sq)
    for j in range(n_x):
        out[:, j] = np.convolve(sq[:, j], k, mode="same")
    return np.sqrt(out / count[:, None]).astype(np.float32)


def apply_gain(section, rms, eps=1e-10):
    out = np.zeros_like(section)
    ok = rms > eps
    out[ok] = section[ok] / rms[ok]
    return out


def apply_gain_display(section, rms):
    """Display only: where the local RMS is negligible compared with the
    typical RMS of the section (no-coverage zones, tiny constant values) the
    output is set to zero instead of being normalised to +-1."""
    nz = rms[rms > 0]
    floor = 1e-3 * np.median(nz) if nz.size else 0.0
    out = np.zeros_like(section)
    ok = rms > floor
    out[ok] = section[ok] / rms[ok]
    return out


def snr_db(ref, est):
    return 10 * np.log10((ref ** 2).sum() / max(((ref - est) ** 2).sum(), 1e-30))


def snr_table(raw, dt_ms, agc_s, tsel, xsel=None):
    rms = {k: agc_rms(v, dt_ms, agc_s) for k, v in raw.items()}
    # floor for the ground-truth gain: where the reference section is almost
    # silent (water column, before the first arrivals) 1/RMS would explode and
    # dominate the comparison; the gain is capped at 1 / (1% of the median RMS)
    nz = rms["GT"][rms["GT"] > 0]
    rms_gt_floored = np.maximum(rms["GT"], 0.01 * np.median(nz))
    gt_raw = raw["GT"]
    if xsel is None:
        xsel = np.ones(gt_raw.shape[1], bool)
    sel = np.ix_(tsel, xsel)
    rows = {}
    for key in [k for k in raw if k not in ("GT", "MASK")]:
        own = snr_db(apply_gain(gt_raw, rms["GT"])[sel],
                     apply_gain(raw[key], rms[key])[sel])
        gtg = snr_db(apply_gain(gt_raw, rms_gt_floored)[sel],
                     apply_gain(raw[key], rms_gt_floored)[sel])
        noagc = snr_db(gt_raw[sel], raw[key][sel])
        rows[key] = (own, gtg, noagc)
    return rows


def auto_windows(n_x, t_max):
    """Default windows: a shallow and a deep box over the central CMPs."""
    x0, x1 = int(0.30 * n_x), int(0.60 * n_x)
    return ((x0, x1, 0.20 * t_max, 0.45 * t_max),
            (x0, x1, 0.55 * t_max, 0.85 * t_max))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=int, default=50, choices=[30, 50])
    ap.add_argument("--prefix", default="CROP_M18_A")
    ap.add_argument("--stack-dir", default=str(STACK_DIR))
    ap.add_argument("--agc", type=float, default=1.0)
    ap.add_argument("--vmax-pct", type=float, default=98.0)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--cmap", default="seismic")
    ap.add_argument("--zoomA", nargs=4, type=float, default=None,
                    metavar=("CMP0", "CMP1", "T0", "T1"))
    ap.add_argument("--zoomB", nargs=4, type=float, default=None,
                    metavar=("CMP0", "CMP1", "T0", "T1"))
    ap.add_argument("--snr-tmin", type=float, default=None)
    ap.add_argument("--snr-tmax", type=float, default=None)
    ap.add_argument("--snr-xmin", type=int, default=None,
                    help="first CMP included in the SNR (e.g. to exclude the edge)")
    ap.add_argument("--snr-xmax", type=int, default=None)
    ap.add_argument("--edge-check", nargs=2, type=int, default=None,
                    metavar=("CMP0", "CMP1"),
                    help="print raw statistics of each section in these CMPs")
    ap.add_argument("--snr-only", action="store_true")
    ap.add_argument("--agc-check", action="store_true",
                    help="compare three AGC implementations (own-AGC SNR)")
    ap.add_argument("--out", default=str(RESULTS_ROOT / "figures/Fig_8"))
    ap.add_argument("--extra", action="append", default=[], metavar="LABEL=PATH",
                    help="extra stacked section (repeatable), shown after DL")
    args = ap.parse_args()

    # ---------------- load ----------------
    raw, dt_ms = {}, None
    for key, _ in METHODS:
        p = Path(args.stack_dir) / f"{args.prefix}{args.ratio}_{key}_STK.sgy"
        raw[key], dt_ms = load_stack_sgy(str(p))
    n_t, n_x = raw["GT"].shape
    methods = list(METHODS)
    extras = []
    for item in args.extra:
        label, path = item.split("=", 1)
        data, dt_e = load_stack_sgy(path)
        if data.shape != raw["GT"].shape or dt_e != dt_ms:
            raise SystemExit(f"{label}: shape {data.shape}, dt {dt_e} ms "
                             f"differ from GT {raw['GT'].shape}, dt {dt_ms} ms")
        raw[label] = data
        extras.append((label, label))
        print(f"extra section: {label} <- {path}")
    i_dl = [k for k, _ in methods].index("DL")
    methods = methods[:i_dl + 1] + extras + methods[i_dl + 1:]
    # rename the proposed model in titles when extras are shown
    if extras:
        methods = [(k, "DL" if k == "DL" else lab) for k, lab in methods]
    raw = {k: raw[k] for k, _ in methods}          # table and panels in display order
    t = np.arange(n_t) * dt_ms / 1000.0
    print(f"{n_t} samples x {n_x} CMPs, dt = {dt_ms} ms")

    # ---------------- SNR variants ----------------
    tsel = np.ones(n_t, bool)
    if args.snr_tmin is not None:
        tsel &= t >= args.snr_tmin
    if args.snr_tmax is not None:
        tsel &= t <= args.snr_tmax
    xsel = np.ones(n_x, bool)
    if args.snr_xmin is not None:
        xsel[:args.snr_xmin] = False
    if args.snr_xmax is not None:
        xsel[args.snr_xmax + 1:] = False
    if args.edge_check:
        a, b = args.edge_check
        print(f"\nRaw values in CMP {a}-{b} (all times):")
        for key, _ in methods:
            z = raw[key][:, a:b + 1]
            nz = z[z != 0]
            print(f"  {key:5s} zero samples {100*(z == 0).mean():5.1f}%   "
                  f"non-zero: median {np.median(nz) if nz.size else 0:+.3e}  "
                  f"std {nz.std() if nz.size else 0:.3e}")
    rows = snr_table(raw, dt_ms, args.agc, tsel, xsel)
    win = ("whole section" if tsel.all() and xsel.all()
           else f"t {t[tsel][0]:.2f}-{t[tsel][-1]:.2f} s, "
                f"CMP {np.flatnonzero(xsel)[0]}-{np.flatnonzero(xsel)[-1]}")
    print(f"\nStack SNR vs ground truth (dB), {args.ratio}% missing, {win}")
    w = max(6, max(len(k) for k in rows))
    print(f"{'method':{w}s} {'own AGC':>9} {'GT gain':>9} {'no AGC':>9}")
    for key, (a, b, c) in rows.items():
        print(f"{key:{w}s} {a:>+9.2f} {b:>+9.2f} {c:>+9.2f}")
    print("  own AGC = each section with its own 1-s AGC (values in the paper)")
    print("  GT gain = ground-truth AGC gain applied to every section "
          "(gain capped at 1/(1% median RMS))")
    print("  no AGC  = raw stacked amplitudes")
    if args.agc_check:
        print("\nAGC implementation check (own AGC, whole section):")
        impl = {"float32 cumsum (original)": agc_rms_float32_original,
                "float64 cumsum (this script)": agc_rms,
                "float64 direct convolution": agc_rms_direct}
        for name, fn in impl.items():
            g = apply_gain(raw["GT"], fn(raw["GT"], dt_ms, args.agc))
            vals = "  ".join(
                f"{k} {snr_db(g, apply_gain(raw[k], fn(raw[k], dt_ms, args.agc))):+.2f}"
                for k in ("DL", "MSSA", "POCS"))
            print(f"  {name:30s} {vals}")
        # own-AGC SNR excluding the zone without data (raw GT negligible)
        amp = np.abs(raw["GT"])
        data_zone = agc_rms(raw["GT"], dt_ms, args.agc) > 1e-3 * np.median(
            agc_rms(raw["GT"], dt_ms, args.agc)[amp > 0])
        g = apply_gain(raw["GT"], agc_rms(raw["GT"], dt_ms, args.agc))
        vals = "  ".join(
            f"{k} {snr_db(g[data_zone], apply_gain(raw[k], agc_rms(raw[k], dt_ms, args.agc))[data_zone]):+.2f}"
            for k in ("DL", "MSSA", "POCS"))
        print(f"  {'float64, excluding no-data zone':30s} {vals}"
              f"   ({100*(~data_zone).mean():.1f}% of samples excluded)")
        r32 = agc_rms_float32_original(raw["GT"], dt_ms, args.agc)
        r64 = agc_rms(raw["GT"], dt_ms, args.agc)
        rel = np.abs(r32 - r64) / np.maximum(r64, 1e-30)
        for a, b in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, t[-1] + 1)):
            sel = (t >= a) & (t < b)
            if sel.any():
                print(f"  GT RMS relative error float32 vs float64, {a:.0f}-{min(b, t[-1]):.1f} s: "
                      f"median {np.median(rel[sel]):.1e}, max {rel[sel].max():.1e}")
    if args.snr_only or args.agc_check:
        return

    # ---------------- figure ----------------
    shown = {k: apply_gain_display(v, agc_rms(v, dt_ms, args.agc)) if args.agc > 0 else v
             for k, v in raw.items()}
    tlim = args.tmax or t[-1]
    vmax = float(np.percentile(np.abs(shown["GT"][t <= tlim]), args.vmax_pct))
    wA, wB = (tuple(args.zoomA) if args.zoomA else None,
              tuple(args.zoomB) if args.zoomB else None)
    if wA is None or wB is None:
        aA, aB = auto_windows(n_x, tlim)
        wA, wB = wA or aA, wB or aB
    print(f"\nwindow A: CMP {wA[0]:.0f}-{wA[1]:.0f}, {wA[2]:.2f}-{wA[3]:.2f} s")
    print(f"window B: CMP {wB[0]:.0f}-{wB[1]:.0f}, {wB[2]:.2f}-{wB[3]:.2f} s")

    ncol = len(methods)
    fig = plt.figure(figsize=(12 * ncol / 5, 11))
    gs = fig.add_gridspec(3, ncol, height_ratios=[1.3, 0.85, 0.85],
                          hspace=0.3, wspace=0.1)
    for k, (key, label) in enumerate(methods):
        for row, w in ((0, None), (1, wA), (2, wB)):
            ax = fig.add_subplot(gs[row, k])
            ax.imshow(shown[key], aspect="auto", cmap=args.cmap,
                      vmin=-vmax, vmax=vmax, interpolation="nearest",
                      extent=[0, n_x - 1, t[-1], 0])
            if w is None:
                ax.set_ylim(tlim, 0)
                ax.set_title(label)
                if key in rows:
                    ax.text(0.97, 0.03, f"{rows[key][0]:+.2f} dB",
                            transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=8.5, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                      ec="black", alpha=0.9, lw=0.8))
                if k == 0:
                    for (x0, x1, t0, t1), lab in ((wA, "A"), (wB, "B")):
                        ax.add_patch(Rectangle((x0, t0), x1 - x0, t1 - t0,
                                               fill=False, ec="black", lw=1.3))
                        ax.text(x1, t0, f" {lab}", va="top", ha="left",
                                fontsize=10, fontweight="bold")
            else:
                ax.set_xlim(w[0], w[1])
                ax.set_ylim(w[3], w[2])
                # keep tick labels away from the panel edges, so that the last
                # label of one panel does not overlap the first of the next
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
                if k == 0:
                    ax.text(-0.45, 0.5, "A" if row == 1 else "B",
                            transform=ax.transAxes, fontsize=12,
                            fontweight="bold", va="center")
            if k == 0:
                ax.set_ylabel("time (s)")
            else:
                ax.set_yticklabels([])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
            if row == 2 or row == 0:
                ax.set_xlabel("CMP")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
