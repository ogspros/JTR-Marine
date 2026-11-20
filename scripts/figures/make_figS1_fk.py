#!/usr/bin/env python3
"""
make_figS1_fk.py -- Fig. S1: f-k spectra of a training line and a cross-survey
line.

Average f-k power spectrum over all gathers of two 4500 m lines:
    CROP-M31  (deep water, training pool)
    CROP-M18  (shallow water, cross-survey, coherent noise)
with the two lines that bound the "signal cone" of the f-k energy-ratio
metric (jtr_marine/metrics.py: v_min = 1400 m/s, v_max = 5000 m/s), i.e. the band
    f / v_max <= |k| <= f / v_min.
Same preprocessing as the model input (project loader). The spectrum is
computed on the recorded offsets only (padding excluded), with a Hann taper
in offset and time to limit leakage.

It prints, for each line, the fraction of the spectral energy that falls
inside the cone, to support the interpretation in the text.

Inputs:  DATA_ROOT/M31_SP250_550.sgy, DATA_ROOT/CROP_M18_SP_500_800.sgy
Output:  RESULTS_ROOT/figures/Fig_S1.{pdf,png} (--out)

Usage
-----
    python scripts/figures/make_figS1_fk.py
    python scripts/figures/make_figS1_fk.py --fmax 80 --n-gathers 100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import DATA_ROOT, RESULTS_ROOT

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config
from jtr_marine.data.dataset import SeismicGatherDataset

RAW = str(DATA_ROOT)
LINES = [("CROP-M31 (4500 m, deep water, training)", f"{RAW}/M31_SP250_550.sgy"),
         ("CROP-M18 (4500 m, shallow water, unseen)", f"{RAW}/CROP_M18_SP_500_800.sgy")]
V_MIN, V_MAX = 1400.0, 5000.0

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9})


def load(cfg, path):
    dc = cfg.data
    return SeismicGatherDataset(
        data_path=path, streamer_length=4500.0,
        max_streamer_length=dc.max_streamer_length,
        dt_target_ms=dc.dt_target_ms, t_max_s=dc.t_max_s,
        x_max_m=dc.x_max_m, dx_target_m=dc.dx_target_m,
        masking_ratios=[0.3], masking_strategy="random",
        block_size=dc.block_size, clip_sigma=dc.clip_sigma,
        normalize_per_trace=dc.normalize_per_trace,
        apply_filter=dc.apply_filter, filter_f_cut=dc.filter_f_cut,
        val_fraction=0.0, random_seed=42, split="train").gathers


def mean_fk_power(gathers, dt, dx, n_max=None):
    """Average |FFT2|^2 over gathers, recorded offsets only, Hann-tapered."""
    valid = np.flatnonzero((np.asarray(gathers[0]) ** 2).sum(axis=0) > 1e-10)
    i0, i1 = valid[0], valid[-1] + 1
    n_t, n_x = gathers[0].shape[0], i1 - i0
    taper = np.outer(np.hanning(n_t), np.hanning(n_x))
    nf, nk = n_t, 2 * n_x                        # zero-pad in offset for smoother k
    acc = np.zeros((nf, nk))
    sel = gathers if n_max is None else gathers[:: max(1, len(gathers) // n_max)]
    for g in sel:
        spec = np.fft.fft2(np.asarray(g)[:, i0:i1] * taper, s=(nf, nk))
        acc += np.abs(spec) ** 2
    acc /= len(sel)
    f = np.fft.fftfreq(nf, dt)
    k = np.fft.fftfreq(nk, dx)
    return np.fft.fftshift(f), np.fft.fftshift(k), np.fft.fftshift(acc), len(sel)


def cone_fraction(f, k, P, fmin=0.0):
    F, K = np.meshgrid(f, k, indexing="ij")
    Fa, Ka = np.abs(F), np.abs(K)
    inside = (Ka <= Fa / V_MIN) & (Ka >= Fa / V_MAX) & (Fa >= fmin)
    total = Fa >= fmin
    return P[inside].sum() / P[total].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmax", type=float, default=80.0)
    ap.add_argument("--n-gathers", type=int, default=None,
                    help="subsample gathers (default: all)")
    ap.add_argument("--db-range", type=float, default=40.0)
    ap.add_argument("--out", default=str(RESULTS_ROOT / "figures/Fig_S1"))
    args = ap.parse_args()

    cfg = get_config()
    dt = cfg.data.dt_target_ms / 1000.0
    dx = cfg.data.dx_target_m

    fig, axes = plt.subplots(1, len(LINES), figsize=(11, 4.6), sharey=True)
    for ax, (label, path) in zip(axes, LINES):
        f, k, P, n = mean_fk_power(load(cfg, path), dt, dx, args.n_gathers)
        frac = cone_fraction(f, k, P)
        keep = (f >= 0) & (f <= args.fmax)
        Pd = 10 * np.log10(P[keep] / P[keep].max() + 1e-30)
        ax.imshow(Pd, aspect="auto", origin="lower", cmap="magma",
                  vmin=-args.db_range, vmax=0,
                  extent=[k[0], k[-1], f[keep][0], f[keep][-1]])
        ff = np.linspace(0, args.fmax, 50)
        for v, ls in ((V_MIN, "--"), (V_MAX, ":")):
            for sgn in (1, -1):
                ax.plot(sgn * ff / v, ff, color="cyan", ls=ls, lw=1.1,
                        label=f"{v:.0f} m/s" if sgn == 1 else None)
        ax.set_xlim(k[0], k[-1]); ax.set_ylim(0, args.fmax)
        ax.set_title(f"{label}\nenergy inside the cone: {100*frac:.1f}%  "
                     f"({n} gathers)")
        ax.set_xlabel("wavenumber (1/m)")
        print(f"{label:45s} gathers={n:4d}  energy inside cone "
              f"(1400-5000 m/s): {100*frac:.1f}%")
    axes[0].set_ylabel("frequency (Hz)")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
