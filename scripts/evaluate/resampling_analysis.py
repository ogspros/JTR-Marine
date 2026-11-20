#!/usr/bin/env python3
"""
resampling_analysis.py — Fig. S2: how does the resampling of the 1500 m lines
from 12.5 m to 25 m affect the reconstruction task?

Works on the SEG-Y files at their NATIVE trace spacing (not through the project
loader, which maps the traces onto the 25 m grid by linear interpolation; since
the 25 m positions coincide with every other native trace, that mapping keeps
one trace in two, without spatial anti-alias filtering).

Same time preprocessing as the network input: 2 ms sampling, 4.5 s window,
8 Hz zero-phase high-pass, per-trace clip at 3 sigma and normalisation to unit
peak.

For each survey (a subset of gathers evenly spaced along the line):
  1. Spectral: fraction of the f-k energy at |k| above the Nyquist wavenumber
     of the 25 m grid (0.02 1/m), i.e. the energy that the resampling cannot
     represent and folds back (only for lines recorded at 12.5 m), also split
     by frequency band.
  2. Interpolation difficulty, independent of the network:
     - median correlation between neighbouring traces, and
     - SNR of a plain linear interpolation of each trace from its two
       neighbours, trace(i) ~ [trace(i-s) + trace(i+s)] / 2,
     at 12.5 m (s = 1) and at 25 m (s = 2) for the 12.5 m lines, and at 25 m
     (s = 1) for the lines recorded at 25 m, for comparison.

Optionally (--figure) the average f-k spectrum of the 12.5 m lines with the
25 m Nyquist wavenumber marked.

Usage
-----
    python scripts/evaluate/resampling_analysis.py
    python scripts/evaluate/resampling_analysis.py --n-gathers 40 --figure
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import DATA_ROOT, RESULTS_ROOT

import argparse
import json
from pathlib import Path

import numpy as np
import segyio
from scipy.signal import resample_poly, butter, sosfiltfilt

RAW = str(DATA_ROOT)
SURVEYS = [  # label, file
    ("SISOC-MEPA (1500 m)", "MEPA_SP_200_500.sgy"),
    ("CG-1 (1500 m)",       "GC01_SP1100_1400.sgy"),
    ("CG-3 (1500 m)",       "CG03_SP150_450.sgy"),
    ("IT90-65 (3000 m)",    "IT065_SP1000_1300.sgy"),
    ("IT89-33 (3000 m)",    "IT89_33_SP600_900.sgy"),
    ("CROP-M31 (4500 m)",   "M31_SP250_550.sgy"),
    ("CROP-M32 (4500 m)",   "M32_SP500_800.sgy"),
]
DT_MS, T_MAX_S, F_CUT, CLIP = 2.0, 4.5, 8.0, 3.0
K_NYQ_25 = 1.0 / (2 * 25.0)
BANDS = [(8, 30), (30, 60), (60, 125)]


def read_gathers(path, n_gathers):
    """Shot gathers at native spacing, sorted by |offset|; (gathers, dx_m)."""
    with segyio.open(path, "r", ignore_geometry=True) as f:
        ffid = f.attributes(segyio.TraceField.FieldRecord)[:]
        off = np.abs(f.attributes(segyio.TraceField.offset)[:]).astype(float)
        dt_native = segyio.tools.dt(f) / 1000.0
        shots, counts = np.unique(ffid, return_counts=True)
        full = shots[counts == np.bincount(counts).argmax()]      # complete shots
        pick = full[np.linspace(0.05 * len(full), 0.95 * len(full) - 1,
                                n_gathers).astype(int)]
        gathers = []
        for s in pick:
            idx = np.flatnonzero(ffid == s)
            idx = idx[np.argsort(off[idx])]
            g = np.stack([f.trace[int(i)] for i in idx], axis=1).astype(np.float64)
            gathers.append(g)
        # offsets are stored as integers (50, 63, 75 m...): use the mean spacing
        # over the whole spread rather than the median of rounded differences
        o = np.sort(off[np.flatnonzero(ffid == pick[0])])
        dx = float((o[-1] - o[0]) / (len(o) - 1))
    return gathers, dx, dt_native


def preprocess(g, dt_native):
    if abs(dt_native - DT_MS) > 1e-6:
        from fractions import Fraction
        fr = Fraction(dt_native / DT_MS).limit_denominator(8)
        g = resample_poly(g, fr.numerator, fr.denominator, axis=0)
    n = int(round(T_MAX_S * 1000 / DT_MS))
    g = g[:n] if g.shape[0] >= n else np.pad(g, ((0, n - g.shape[0]), (0, 0)))
    sos = butter(4, F_CUT, btype="highpass", fs=1000.0 / DT_MS, output="sos")
    g = sosfiltfilt(sos, g, axis=0)
    for j in range(g.shape[1]):
        tr = g[:, j]
        s = tr.std()
        if s > 0:
            tr = np.clip(tr, -CLIP * s, CLIP * s)
            g[:, j] = tr / (np.abs(tr).max() + 1e-12)
    return g


def fk_power(g, dx):
    nt, nx = g.shape
    taper = np.outer(np.hanning(nt), np.hanning(nx))
    P = np.abs(np.fft.fft2(g * taper, s=(nt, 2 * nx))) ** 2
    f = np.fft.fftfreq(nt, DT_MS / 1000.0)
    k = np.fft.fftfreq(2 * nx, dx)
    return f, k, P


def beyond_nyquist(f, k, P):
    F, K = np.meshgrid(np.abs(f), np.abs(k), indexing="ij")
    out = {"all": P[(K > K_NYQ_25) & (F >= F_CUT)].sum() / P[F >= F_CUT].sum()}
    for a, b in BANDS:
        band = (F >= a) & (F < b)
        out[f"{a}-{b} Hz"] = P[band & (K > K_NYQ_25)].sum() / P[band].sum()
    return out


def interp_difficulty(g, s):
    """Correlation of traces s apart, and SNR of linear interpolation from
    the neighbours at distance s."""
    nx = g.shape[1]
    corr = [np.corrcoef(g[:, j], g[:, j + s])[0, 1] for j in range(nx - s)
            if g[:, j].std() > 0 and g[:, j + s].std() > 0]
    tgt = g[:, s:nx - s]
    est = 0.5 * (g[:, :nx - 2 * s] + g[:, 2 * s:])
    snr = 10 * np.log10((tgt ** 2).sum() / max(((tgt - est) ** 2).sum(), 1e-30))
    return float(np.median(corr)), float(snr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gathers", type=int, default=30)
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_ROOT / "resampling"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    results, spectra = {}, {}
    for label, fname in SURVEYS:
        path = Path(args.raw) / fname
        if not path.exists():
            print(f"  (missing {path})"); continue
        raw, dx, dtn = read_gathers(str(path), args.n_gathers)
        gs = [preprocess(g, dtn) for g in raw]
        r = {"dx_native": dx, "n_gathers": len(gs), "n_traces": gs[0].shape[1]}
        if dx < 20:                                   # recorded at 12.5 m
            fr = [beyond_nyquist(*fk_power(g, dx)) for g in gs]
            r["energy_beyond_25m_nyquist"] = {key: float(np.mean([x[key] for x in fr]))
                                              for key in fr[0]}
            for s, lab in ((1, "12.5 m"), (2, "25 m")):
                v = [interp_difficulty(g, s) for g in gs]
                r[f"corr_{lab}"] = float(np.median([a for a, _ in v]))
                r[f"linear_interp_snr_{lab}"] = float(np.mean([b for _, b in v]))
            if args.figure:
                acc = None
                for g in gs:
                    f, k, P = fk_power(g, dx)
                    acc = P if acc is None else acc + P
                spectra[label] = (np.fft.fftshift(f), np.fft.fftshift(k),
                                  np.fft.fftshift(acc / len(gs)))
        else:                                          # recorded at 25 m
            v = [interp_difficulty(g, 1) for g in gs]
            r["corr_25 m"] = float(np.median([a for a, _ in v]))
            r["linear_interp_snr_25 m"] = float(np.mean([b for _, b in v]))
        results[label] = r

    print(f"\n{'survey':22s} {'dx':>5} {'E>kN25':>7} {'8-30':>6} {'30-60':>6} {'60-125':>7}"
          f" | {'corr12.5':>8} {'corr25':>7} | {'SNRlin12.5':>10} {'SNRlin25':>9}")
    for label, r in results.items():
        e = r.get("energy_beyond_25m_nyquist", {})
        fmt = lambda x, p=1: "" if x is None else f"{100*x:.{p}f}%"
        print(f"{label:22s} {r['dx_native']:>5.1f} {fmt(e.get('all')):>7} "
              f"{fmt(e.get('8-30 Hz')):>6} {fmt(e.get('30-60 Hz')):>6} {fmt(e.get('60-125 Hz')):>7}"
              f" | {r.get('corr_12.5 m', float('nan')):>8.3f} {r.get('corr_25 m', float('nan')):>7.3f}"
              f" | {r.get('linear_interp_snr_12.5 m', float('nan')):>10.2f}"
              f" {r.get('linear_interp_snr_25 m', float('nan')):>9.2f}")
    json.dump(results, open(out / "resampling_analysis.json", "w"), indent=2)
    print(f"\nSaved {out / 'resampling_analysis.json'}")
    print("E>kN25: f-k energy above the 25 m Nyquist wavenumber (lost/aliased by "
          "the resampling), overall and per frequency band.\ncorr: median correlation "
          "of neighbouring traces. SNRlin: SNR of linear interpolation from the two "
          "neighbours.")

    if args.figure and spectra:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(spectra), figsize=(5.2 * len(spectra), 4.4),
                                 sharey=True, squeeze=False)
        for ax, (label, (f, k, P)) in zip(axes[0], spectra.items()):
            keep = (f >= 0) & (f <= 125)
            Pd = 10 * np.log10(P[keep] / P[keep].max() + 1e-30)
            ax.imshow(Pd, aspect="auto", origin="lower", cmap="magma", vmin=-40, vmax=0,
                      extent=[k[0], k[-1], 0, 125])
            for sgn in (-1, 1):
                ax.axvline(sgn * K_NYQ_25, color="cyan", ls="--", lw=1.1)
            ax.set_title(f"{label}\nenergy beyond 25 m Nyquist: "
                         f"{100*results[label]['energy_beyond_25m_nyquist']['all']:.1f}%")
            ax.set_xlabel("wavenumber (1/m)")
        axes[0, 0].set_ylabel("frequency (Hz)")
        for ext in ("pdf", "png"):
            fig.savefig(out / f"Fig_S2.{ext}", dpi=300, bbox_inches="tight")
        print(f"saved {out / 'Fig_S2.pdf'}")


if __name__ == "__main__":
    main()
