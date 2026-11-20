#!/usr/bin/env python3
"""
eval_marmousi.py — DL vs POCS vs MSSA on the Marmousi synthetic data:
quantitative comparison on data with known ground truth (source of the
examples plotted in Fig. 6).

Data
----
Original Marmousi (Brougois et al., 1990), converted by marmousi_to_segy.py:
    MARMOUSI_1500.sgy : 60 channels, offsets 200-1675 m -> 1500 m conditioning
                        (a geometry seen in training: the primary test)
    MARMOUSI_full.sgy : 96 channels, offsets 200-2575 m -> streamer length not
                        seen in training (secondary test)
All 240 shots are used: none of them was seen in training.

Protocol (identical to the real-data evaluation)
------------------------------------------------
- Gathers loaded and preprocessed by the project loader (same grid, filter,
  normalisation), exactly as the cross-survey lines.
- Masks from the project masking functions, one fixed seed; for each gather the
  SAME mask is used by DL, POCS and MSSA (paired comparison).
- Trace replacement for all methods (known traces preserved exactly).
- Baselines with the parameters used for the paper's Table 5:
      POCS  n_iter = 100 (threshold schedule from baselines/pocs.py defaults)
      MSSA  rank = 10, n_iter = 3 (damping from baselines/mssa.py defaults)
- Metrics: SNR on missing traces, SNR on the full gather, SSIM, f-k energy ratio.
- Metrics are computed on the recorded window only (0-2.9 s): the data are
  zero-padded to the 4.5 s loader window, and the padded part is not data.
  As a diagnostic, the fraction of each method's energy on the missing traces
  that falls in the padded zone is stored ("*_pad_energy").
- Also stored: the fraction of recorded traces actually removed. With a near
  offset of 200 m the mask window is shifted by 8 grid columns, so the
  effective fraction is below the nominal ratio (about 26% for a nominal 30%);
  it must be reported alongside the results.

Optionally (--save-examples) the reconstructions of a few gathers are saved to
.npz, for the single-trace amplitude comparison and the zoomed panels of
Fig. 6 (make_fig6_marmousi.py).

Usage
-----
    python scripts/evaluate/eval_marmousi.py                 # 1500 m, random, 7 ratios
    python scripts/evaluate/eval_marmousi.py --datasets MARMOUSI_1500 MARMOUSI_full
    python scripts/evaluate/eval_marmousi.py --ratios 0.3 0.5 --save-examples
    python scripts/evaluate/eval_marmousi.py --n-gathers 20  # quick test

Timing: POCS ~5 s and MSSA ~3-5 s per gather -> about 3-4 h for 240 gathers x
7 ratios on one dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import get_config
from jtr_marine.data.dataset import SeismicGatherDataset, SharedGatherDataset
from jtr_marine.model_loader import load_checkpoint
from jtr_marine.baselines.pocs import pocs_interpolate_validity_aware
from jtr_marine.baselines.mssa import mssa_interpolate_validity_aware
from jtr_marine.metrics import compute_snr, compute_ssim_metric, compute_fk_energy_ratio

DATA_DIR = str(DATA_ROOT)
DATASETS = {
    # t_rec_s: length of the actual recording; beyond it the loader window
    # (4.5 s) is zero padding, which is excluded from the metrics
    "MARMOUSI_1500": {"file": "MARMOUSI_1500.sgy", "length": 1500.0, "t_rec_s": 2.9},
    # physical streamer length, not seen in training (conditioning 2575/4500)
    "MARMOUSI_full": {"file": "MARMOUSI_full.sgy", "length": 2575.0, "t_rec_s": 2.9},
}
MASK_SEED = 12345          # fixed, as in the cross-survey evaluation


def patch_normalisation_to_recorded_window(t_rec_s):
    """
    The project normalisation (data.dataset.clip_and_normalize) computes the
    per-trace standard deviation over the whole 4.5 s window. On Marmousi the
    last 1.6 s are zero padding, which lowers sigma and makes the 3-sigma clip
    harsher than on the real data (whose window is always live). This patch
    computes sigma on the recorded window only; clipping and normalisation are
    otherwise unchanged.
    """
    import jtr_marine.data.dataset as dsmod

    def clip_and_normalize_recorded(gather, clip_sigma=3.0, apply_filter=False,
                                    filter_f_cut=8.0, dt_ms=2.0):
        g = gather.copy().astype(np.float32)
        if apply_filter:
            g = dsmod.highpass_filter(g, filter_f_cut, dt_ms)
        n_live = int(round(t_rec_s * 1000.0 / dt_ms))
        for i in range(g.shape[1]):
            tr = g[:, i]
            if np.all(tr == 0):
                continue
            sigma = np.std(tr[:n_live])
            if sigma < 1e-10:
                continue
            tr = np.clip(tr, -clip_sigma * sigma, clip_sigma * sigma)
            g[:, i] = tr / (np.max(np.abs(tr)) + 1e-10)
        return g

    dsmod.clip_and_normalize = clip_and_normalize_recorded


def build_dataset(config, sgy_path, streamer_length, strategy, ratio):
    """All gathers of one file, loaded and masked as the cross-survey lines."""
    dc = config.data
    n_t = int(round(dc.t_max_s * 1000.0 / dc.dt_target_ms))
    n_x = int(round(dc.x_max_m / dc.dx_target_m)) + 1
    full_ds = SeismicGatherDataset(
        data_path=sgy_path, streamer_length=streamer_length,
        max_streamer_length=dc.max_streamer_length,
        dt_target_ms=dc.dt_target_ms, t_max_s=dc.t_max_s,
        x_max_m=dc.x_max_m, dx_target_m=dc.dx_target_m,
        masking_ratios=[ratio], masking_strategy=strategy,
        block_size=dc.block_size, clip_sigma=dc.clip_sigma,
        normalize_per_trace=dc.normalize_per_trace,
        apply_filter=dc.apply_filter, filter_f_cut=dc.filter_f_cut,
        val_fraction=dc.val_fraction, random_seed=dc.random_seed,
        split="train",
    )
    gathers = full_ds.gathers
    return SharedGatherDataset(
        gathers=gathers, indices=np.arange(len(gathers), dtype=np.int64),
        geometry_code=streamer_length / dc.max_streamer_length,
        n_time=n_t, n_offsets=n_x,
        masking_ratios=[ratio], masking_strategy=strategy,
        block_size=dc.block_size, random_seed=MASK_SEED,
    )


def metrics(pred, target, mask, valid, dt, dx):
    p = torch.from_numpy(pred[None, None]).float()
    t = torch.from_numpy(target[None, None]).float()
    m = torch.from_numpy(mask[None]).float()
    v = torch.from_numpy(valid[None]).float()
    return {
        "snr_miss": compute_snr(p, t, m, v, missing_only=True).item(),
        "snr_full": compute_snr(p, t, m, v, missing_only=False).item(),
        "ssim": compute_ssim_metric(p, t).item(),
        "fk": compute_fk_energy_ratio(p, t, dt=dt, dx=dx).item(),
    }


def evaluate(model, device, config, name, strategy, ratio, args, ex_dir):
    ds_info = DATASETS[name]
    dataset = build_dataset(config, f"{args.data_dir}/{ds_info['file']}",
                            ds_info["length"], strategy, ratio)
    n = len(dataset) if args.n_gathers is None else min(args.n_gathers, len(dataset))
    dt = config.data.dt_target_ms * 1e-3
    dx = float(config.data.dx_target_m)
    n_rec = int(round(ds_info["t_rec_s"] / dt))          # samples in recorded window

    res = {"n": n, "n_rec_samples": n_rec, "shot_index": [], "eff_ratio": []}
    for meth in ("dl", "pocs", "mssa"):
        for k in ("snr_miss", "snr_full", "ssim", "fk", "pad_energy",
                  "rms_ratio", "corr"):
            res[f"{meth}_{k}"] = []

    model.eval()
    t0 = time.time()
    for idx in range(n):
        s = dataset[idx]
        target = s["target"].squeeze().numpy()
        masked = s["input"][0].numpy()
        mask = s["trace_mask"].numpy()
        valid = s["validity"].numpy()
        keep = mask[np.newaxis, :]

        with torch.no_grad():
            x = s["input"].unsqueeze(0).to(device)
            g = torch.tensor([s["geometry"].item()], device=device)
            dl = model(x, g).cpu().squeeze().numpy()
        pocs = pocs_interpolate_validity_aware(masked, mask, valid,
                                               n_iter=args.pocs_iter)
        mssa = mssa_interpolate_validity_aware(masked, mask, valid,
                                               rank=args.mssa_rank,
                                               n_iter=args.mssa_iter)

        recon = {}
        miss = ((1 - mask) * valid) > 0
        for meth, pred in (("dl", dl), ("pocs", pocs), ("mssa", mssa)):
            recon[meth] = pred * (1 - keep) + masked * keep      # trace replacement
            # metrics on the recorded window only
            for k, v in metrics(recon[meth][:n_rec], target[:n_rec],
                                mask, valid, dt, dx).items():
                res[f"{meth}_{k}"].append(v)
            # amplitude fidelity on missing traces (recorded window):
            # median over traces of RMS(recon)/RMS(truth) and of the correlation
            cols = np.flatnonzero(miss)
            if cols.size:
                rt = np.sqrt((target[:n_rec, cols] ** 2).mean(axis=0))
                rp = np.sqrt((recon[meth][:n_rec, cols] ** 2).mean(axis=0))
                cc = [np.corrcoef(recon[meth][:n_rec, j], target[:n_rec, j])[0, 1]
                      for j in cols]
                res[f"{meth}_rms_ratio"].append(float(np.median(rp / np.maximum(rt, 1e-12))))
                res[f"{meth}_corr"].append(float(np.median(cc)))
            # diagnostic: share of energy on missing traces inside the padding
            e_all = float((pred[:, miss] ** 2).sum())
            e_pad = float((pred[n_rec:, miss] ** 2).sum())
            res[f"{meth}_pad_energy"].append(e_pad / e_all if e_all > 0 else 0.0)

        n_valid = valid.sum()
        res["eff_ratio"].append(float(((1 - mask) * valid).sum() / max(n_valid, 1)))
        res["shot_index"].append(idx)

        if ex_dir is not None and idx in args.example_shots:
            np.savez_compressed(
                ex_dir / f"{name}_{strategy}_{int(ratio*100)}pct_shot{idx:03d}.npz",
                target=target, masked=masked, trace_mask=mask, validity=valid,
                dl=recon["dl"], pocs=recon["pocs"], mssa=recon["mssa"],
                dt=dt, dx=dx)

        if (idx + 1) % 20 == 0:
            el = time.time() - t0
            print(f"    [{idx+1}/{n}] {el:.0f}s, ETA {el/(idx+1)*(n-idx-1):.0f}s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(CKPT_ROOT / "v6_grid25m/within_geology/best.pt"))
    ap.add_argument("--datasets", nargs="+", default=["MARMOUSI_1500"],
                    choices=list(DATASETS))
    ap.add_argument("--strategies", nargs="+", default=["random"])
    ap.add_argument("--ratios", nargs="+", type=float,
                    default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--pocs-iter", type=int, default=100)
    ap.add_argument("--mssa-rank", type=int, default=10)
    ap.add_argument("--mssa-iter", type=int, default=3)
    ap.add_argument("--n-gathers", type=int, default=None, help="quick test")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--output-dir", default=str(RESULTS_ROOT / "marmousi"))
    ap.add_argument("--sigma-recorded", action="store_true",
                    help="compute the normalisation sigma on the recorded window "
                         "only (excludes the zero padding)")
    ap.add_argument("--save-examples", action="store_true")
    ap.add_argument("--example-shots", nargs="+", type=int, default=[60, 120, 180],
                    help="shot indices saved with --save-examples")
    args = ap.parse_args()

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ex_dir = None
    if args.save_examples:
        ex_dir = out / "examples"
        ex_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MARMOUSI EVALUATION — DL vs POCS vs MSSA (paired masks)")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  datasets   : {args.datasets}")
    print(f"  strategies : {args.strategies}   ratios: {args.ratios}")
    print(f"  POCS n_iter={args.pocs_iter} | MSSA rank={args.mssa_rank}, "
          f"n_iter={args.mssa_iter}")
    print("=" * 70)

    if args.sigma_recorded:
        patch_normalisation_to_recorded_window(DATASETS[args.datasets[0]]["t_rec_s"])
        print("  normalisation: sigma on the recorded window (padding excluded)")
    model, _ = load_checkpoint(args.checkpoint, config, device)

    results = {"config": vars(args)}
    for name in args.datasets:
        results[name] = {}
        for strat in args.strategies:
            results[name][strat] = {}
            for r in args.ratios:
                key = f"{int(round(r*100))}pct"
                print(f"\n--- {name} | {strat} | {key} ---")
                res = evaluate(model, device, config, name, strat, r, args, ex_dir)
                results[name][strat][key] = res
                print(f"  effective removed fraction: "
                      f"{100*np.mean(res['eff_ratio']):.1f}% (nominal {100*r:.0f}%)")
                for meth in ("dl", "mssa", "pocs"):
                    print(f"  {meth.upper():5s} SNR_miss={np.mean(res[meth+'_snr_miss']):+6.2f} dB"
                          f"  SNR_full={np.mean(res[meth+'_snr_full']):+6.2f}"
                          f"  SSIM={np.mean(res[meth+'_ssim']):.3f}"
                          f"  R_fk={np.mean(res[meth+'_fk']):.3f}"
                          f"  RMS ratio={np.median(res[meth+'_rms_ratio']):.2f}"
                          f"  corr={np.median(res[meth+'_corr']):.2f}"
                          f"  energy in padding={100*np.mean(res[meth+'_pad_energy']):.1f}%")
                # save after every setting, so a crash loses nothing
                with open(out / "marmousi_per_gather.json", "w") as f:
                    json.dump(results, f)
    print(f"\nSaved: {out / 'marmousi_per_gather.json'}")


if __name__ == "__main__":
    main()
