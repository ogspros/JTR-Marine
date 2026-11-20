#!/usr/bin/env python3
"""
make_fig7_crosssurvey.py -- Fig. 7: qualitative reconstructions on the three
cross-survey lines (CG-3, IT97-245, CROP-M18): ground truth, masked input, DL,
MSSA, POCS.

Notes:

1. Conditioning. The model receives the normalised code
   length / max_streamer_length (0.33, 0.67, 1.0), as built by the dataset
   during training, not the streamer length in metres (1500, 3000, 4500).
2. Each line is loaded once.
3. --rank: SNR_miss of the model on every gather of each line, with the masks of
   this figure, to check where the displayed gathers fall in the distribution
   and to suggest the gather closest to the median.
4. Default layout: one enlarged window per line (a box on the ground truth),
   shown in an extra column next to each line for all five rows. --no-zoom
   gives the 5 x 3 layout without enlarged windows. PDF and PNG output, no
   suptitle.

Masks: int(ratio * N_valid) traces removed at random among the recorded
traces, seed = SEED + gather index. Metrics: SNR on the missing traces, native
amplitudes (no AGC), as in the paper figure.

Inputs:  CKPT_ROOT/v6_grid25m/within_geology/best.pt (--checkpoint)
         DATA_ROOT/CG03_SP150_450.sgy, IT245_SP_800_1100.sgy,
         CROP_M18_SP_500_800.sgy
Output:  RESULTS_ROOT/figures/Fig_7.{pdf,png} (--out)

Usage
-----
    python scripts/figures/make_fig7_crosssurvey.py --rank
    python scripts/figures/make_fig7_crosssurvey.py --gathers 150 180 200
    python scripts/figures/make_fig7_crosssurvey.py --gathers 150 180 200 \\
        --zoom CG3 400 1200 0.5 1.5 --zoom IT245 800 2000 1.0 2.0 --zoom M18 1500 3000 0.8 1.8
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from config import get_config
from jtr_marine.data.dataset import SeismicGatherDataset
from jtr_marine.model_loader import load_checkpoint
from jtr_marine.baselines.pocs import pocs_interpolate_validity_aware
from jtr_marine.baselines.mssa import mssa_interpolate_validity_aware

RAW = str(DATA_ROOT)
SURVEYS = [  # key, label, file, streamer length (m)
    ("CG3",   "CG-3 (1500 m)",     f"{RAW}/CG03_SP150_450.sgy",      1500.0),
    ("IT245", "IT97-245 (3000 m)", f"{RAW}/IT245_SP_800_1100.sgy",   3000.0),
    ("M18",   "CROP-M18 (4500 m)", f"{RAW}/CROP_M18_SP_500_800.sgy", 4500.0),
]
POCS_KW = dict(n_iter=100, threshold_max_pct=99.0, threshold_min_pct=1.0,
               decay="exponential")
MSSA_KW = dict(rank=10, n_iter=3, damping=0.5)
SEED = 42
ROWS = [("gt", "Ground truth", "black"), ("masked", "Masked input", "black"),
        ("dl", "DL", "#065A82"), ("mssa", "MSSA", "#3E8E5A"),
        ("pocs", "POCS", "#C44536")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 8})


def load_line(cfg, path, length):
    dc = cfg.data
    ds = SeismicGatherDataset(
        data_path=path, streamer_length=length,
        max_streamer_length=dc.max_streamer_length,
        dt_target_ms=dc.dt_target_ms, t_max_s=dc.t_max_s,
        x_max_m=dc.x_max_m, dx_target_m=dc.dx_target_m,
        masking_ratios=[0.3], masking_strategy="random",
        block_size=dc.block_size, clip_sigma=dc.clip_sigma,
        normalize_per_trace=dc.normalize_per_trace,
        apply_filter=dc.apply_filter, filter_f_cut=dc.filter_f_cut,
        val_fraction=0.0, random_seed=SEED, split="train")
    return ds.gathers


def make_mask(g, gather_idx, ratio):
    validity = ((g ** 2).sum(axis=0) > 1e-10).astype(np.float32)
    valid_idx = np.flatnonzero(validity > 0.5)
    rng = np.random.default_rng(SEED + gather_idx)
    mask = np.ones(g.shape[1], np.float32)
    mask[rng.choice(valid_idx, int(ratio * len(valid_idx)), replace=False)] = 0
    return mask, validity


def run_dl(model, device, g, mask, geom_code):
    masked = g * mask[None, :]
    x = np.stack([masked, np.tile(mask[None, :], (g.shape[0], 1))])[None]
    with torch.no_grad():
        pred = model(torch.from_numpy(x).float().to(device),
                     torch.tensor([geom_code], dtype=torch.float32,
                                  device=device)).cpu().numpy()[0, 0]
    return mask[None, :] * g + (1 - mask[None, :]) * pred


def snr_miss(gt, rec, mask, validity):
    mv = (mask == 0) & (validity > 0.5)
    return 10 * np.log10((gt[:, mv] ** 2).sum() /
                         max(((gt[:, mv] - rec[:, mv]) ** 2).sum(), 1e-30))


def rank(model, device, cfg, ratio, chosen):
    print(f"DL SNR_miss on every gather ({int(ratio*100)}% random, masks of this figure)")
    for (key, label, path, L), gi in zip(SURVEYS, chosen):
        gathers = load_line(cfg, path, L)
        code = L / cfg.data.max_streamer_length
        vals = {}
        for i, g in enumerate(gathers):
            g = g.astype(np.float32)
            mask, validity = make_mask(g, i, ratio)
            if validity.sum() < 20:
                continue
            vals[i] = snr_miss(g, run_dl(model, device, g, mask, code), mask, validity)
        v = np.array(list(vals.values()))
        med = np.median(v)
        best = min(vals, key=lambda i: abs(vals[i] - med))
        pct = (v < vals[gi]).mean() * 100 if gi in vals else float("nan")
        print(f"  {label:18s} n={len(v)}  median {med:5.2f} dB  "
              f"[p25 {np.percentile(v,25):5.2f}, p75 {np.percentile(v,75):5.2f}]  "
              f"chosen gather {gi}: {vals.get(gi, float('nan')):5.2f} dB "
              f"= {pct:4.0f}th percentile   closest to median: gather {best} "
              f"({vals[best]:.2f} dB)")


def display_agc(g, dt, window_s):
    """Display-only AGC: moving-window RMS by direct convolution in float64."""
    win = 2 * (int(round(window_s / dt)) // 2) + 1
    k = np.ones(win)
    count = np.convolve(np.ones(g.shape[0]), k, mode="same")
    out = np.zeros_like(g, dtype=np.float64)
    for j in range(g.shape[1]):
        sq = np.convolve(g[:, j].astype(np.float64) ** 2, k, mode="same") / count
        rms = np.sqrt(sq)
        ok = rms > 1e-12 * max(rms.max(), 1e-30)
        out[ok, j] = g[ok, j] / rms[ok]
    return out.astype(np.float32)


def render_original_layout(data, args, dt, dx, n_t):
    """Layout without enlarged windows (--no-zoom): 5 x 3 panels, colour map
    --cmap (default seismic), clip at the --clip-pct percentile of the ground
    truth (default 97), SNR boxes; no suptitle, PDF + PNG."""
    row_labels = ["Ground truth",
                  f"Masked input\n({int(args.ratio*100)}% missing)",
                  "Deep learning\nreconstruction",
                  "MSSA reconstruction\n(rank=10, $n_{iter}$=3)",
                  "POCS reconstruction\n($n_{iter}$=100, $t$=99\u21921)"]
    fig, axes = plt.subplots(5, 3, figsize=(15, 18),
                             gridspec_kw=dict(wspace=0.12, hspace=0.18))
    t_end = args.tmax or n_t * dt
    n_show = min(n_t, int(round(t_end / dt)))
    for c, d in enumerate(data):
        cols = np.flatnonzero(d["validity"] > 0.5)
        i0, i1 = int(cols[0]), int(cols[-1]) + 1
        show = {m: (display_agc(d[m], dt, args.agc) if args.agc > 0 else d[m])
                for m, _, _ in ROWS}
        vmax = float(np.percentile(np.abs(show["gt"][:n_show, i0:i1]),
                                   args.clip_pct)) or 1.0
        n_miss = int(((d["mask"] == 0) & (d["validity"] > 0.5)).sum())
        for r, (m, _, rcol) in enumerate(ROWS):
            ax = axes[r, c]
            ax.imshow(show[m][:, i0:i1], aspect="auto", cmap=args.cmap,
                      vmin=-vmax, vmax=vmax, interpolation="nearest",
                      extent=[i0 * dx, (i1 - 1) * dx, n_t * dt, 0])
            ax.set_ylim(t_end, 0)
            if r == 0:
                ax.set_title(f"{d['label']}, gather {d['gi']}\n"
                             f"n_missing = {n_miss}/{len(cols)}",
                             fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{row_labels[r]}\n\nTime (s)", fontsize=11,
                              fontweight="bold")
            if r == 4:
                ax.set_xlabel("Offset (m)", fontsize=10)
            if m in ("dl", "mssa", "pocs"):
                ax.text(0.97, 0.95, f"SNR$_{{miss}}$ = {d['snr_' + m]:+.2f} dB",
                        transform=ax.transAxes, fontsize=9, color=rcol,
                        ha="right", va="top", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor=rcol, alpha=0.92, linewidth=1.2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"saved {args.out}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CKPT_ROOT / "v6_grid25m/within_geology/best.pt"))
    ap.add_argument("--ratio", type=float, default=0.30)
    ap.add_argument("--gathers", nargs=3, type=int, default=[150, 180, 200])
    ap.add_argument("--zoom", nargs=5, action="append", default=[],
                    metavar=("KEY", "X0_M", "X1_M", "T0", "T1"),
                    help="enlarged window for one line (KEY: CG3, IT245, M18)")
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--cmap", default="seismic", help="e.g. seismic, gray_r")
    ap.add_argument("--clip-pct", type=float, default=97.0,
                    help="display clip, percentile of |ground truth| (default 97)")
    ap.add_argument("--agc", type=float, default=0.0,
                    help="display-only AGC window in s (0 = none); SNR values "
                         "are always computed on the original data")
    ap.add_argument("--no-zoom", action="store_true",
                    help="original 5x3 layout of the paper figure, no enlarged windows")
    ap.add_argument("--out", default=str(RESULTS_ROOT / "figures/Fig_7"))
    args = ap.parse_args()

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, cfg, device)

    if args.rank:
        rank(model, device, cfg, args.ratio, args.gathers)
        return

    dt = cfg.data.dt_target_ms / 1000.0
    dx = cfg.data.dx_target_m
    zooms = {z[0]: tuple(map(float, z[1:])) for z in args.zoom}

    data = []
    for (key, label, path, L), gi in zip(SURVEYS, args.gathers):
        g = load_line(cfg, path, L)[gi].astype(np.float32)
        mask, validity = make_mask(g, gi, args.ratio)
        masked = g * mask[None, :]
        d = {"key": key, "label": label, "gi": gi, "gt": g, "masked": masked,
             "mask": mask, "validity": validity,
             "dl": run_dl(model, device, g, mask, L / cfg.data.max_streamer_length)}
        for m, fn, kw in (("mssa", mssa_interpolate_validity_aware, MSSA_KW),
                          ("pocs", pocs_interpolate_validity_aware, POCS_KW)):
            p = fn(masked, mask, validity, **kw)
            d[m] = mask[None, :] * g + (1 - mask[None, :]) * p
        for m in ("dl", "mssa", "pocs"):
            d["snr_" + m] = snr_miss(g, d[m], mask, validity)
        cols = np.flatnonzero(validity > 0)
        d["x0"], d["x1"] = cols[0] * dx, cols[-1] * dx
        tmax = args.tmax or g.shape[0] * dt
        d["zoom"] = zooms.get(key, (d["x0"] + 0.2 * (d["x1"] - d["x0"]),
                                    d["x0"] + 0.6 * (d["x1"] - d["x0"]),
                                    0.15 * tmax, 0.45 * tmax))
        data.append(d)
        print(f"{label}: gather {gi}, SNR_miss DL {d['snr_dl']:+.2f}  "
              f"MSSA {d['snr_mssa']:+.2f}  POCS {d['snr_pocs']:+.2f} dB   "
              f"zoom {d['zoom']}")

    n_t = data[0]["gt"].shape[0]
    t_end = args.tmax or n_t * dt

    if args.no_zoom:
        render_original_layout(data, args, dt, dx, n_t)
        return

    fig = plt.figure(figsize=(13, 14))
    gs = fig.add_gridspec(5, 6, width_ratios=[1.25, 1, 1.25, 1, 1.25, 1],
                          hspace=0.12, wspace=0.18)
    for c, d in enumerate(data):
        vmax = np.percentile(np.abs(d["gt"][:, d["validity"] > 0]), 98)
        zx0, zx1, zt0, zt1 = d["zoom"]
        for r, (m, rlab, rcol) in enumerate(ROWS):
            for sub in (0, 1):
                ax = fig.add_subplot(gs[r, 2 * c + sub])
                ax.imshow(d[m], aspect="auto", cmap="gray_r", vmin=-vmax, vmax=vmax,
                          interpolation="bilinear",
                          extent=[0, (d[m].shape[1] - 1) * dx, n_t * dt, 0])
                if sub == 0:
                    ax.set_xlim(d["x0"], d["x1"]); ax.set_ylim(t_end, 0)
                    if r == 0:
                        ax.set_title(f"{d['label']}\ngather {d['gi']}")
                        ax.add_patch(Rectangle((zx0, zt0), zx1 - zx0, zt1 - zt0,
                                               fill=False, ec="tab:orange", lw=1.2))
                    if m in ("dl", "mssa", "pocs"):
                        ax.text(0.97, 0.03, f"{d['snr_' + m]:+.2f} dB",
                                transform=ax.transAxes, ha="right", va="bottom",
                                fontsize=7.5, color=rcol, fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec=rcol, alpha=0.9, lw=0.8))
                else:
                    ax.set_xlim(zx0, zx1); ax.set_ylim(zt1, zt0)
                    for sp in ax.spines.values():
                        sp.set_edgecolor("tab:orange"); sp.set_linewidth(1.2)
                    if r == 0:
                        ax.set_title("\nenlarged", color="tab:orange")
                if c == 0 and sub == 0:
                    ax.set_ylabel(f"{rlab}\ntime (s)")
                if r < 4:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("offset (m)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
