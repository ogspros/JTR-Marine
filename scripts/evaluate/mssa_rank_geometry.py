#!/usr/bin/env python3
"""
mssa_rank_geometry.py — Is the DL advantage over MSSA inflated by the use of
a single rank (10) for all streamer lengths?

The Hankel matrix of MSSA grows with the number of recorded traces
(about 31x30 at 1500 m, 61x60 at 3000 m, 88x88 at 4500 m), so a fixed rank
keeps a very different fraction of the singular spectrum across geometries.
This script tunes the rank per geometry, with the same discipline used for
all baselines, and re-evaluates MSSA on the test set.

Two modes
---------
--mode tune   On the TRAINING pool only (first 70% of each survey, evenly
              subsampled): for each streamer length, every (rank, n_iter) in
              the grid; mean SNR on missing traces; best configuration per
              geometry saved to <out>/mssa_rank_tuning.json. The test set is
              never touched.

--mode eval   On the TEST set (last 15% of each survey, 276 gathers), in ONE
              process and on IDENTICAL masks (deterministic seed, not derived
              from Python's randomised hash):
                 DL, MSSA with the paper configuration (rank 10, n_iter 3),
                 MSSA with the per-geometry configuration from the tuning.
              Per-gather values saved to <out>/mssa_rank_eval.json, summary
              printed overall and per geometry, with paired statistics.

Usage
-----
    python scripts/evaluate/mssa_rank_geometry.py --mode tune   # ~3 h (8 gathers/geometry, 30%)
    python scripts/evaluate/mssa_rank_geometry.py --mode eval   # ~6 h (92 gathers, 30/50/80%)
MSSA is slow on the larger Hankel matrices (about 60-80 s per gather at
4500 m), hence the reduced tuning set and the test stride.
    python scripts/evaluate/mssa_rank_geometry.py --mode eval --ratios 0.3 0.5 --n-test 20   # quick test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT, RESULTS_ROOT

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import get_config
from jtr_marine.data.dataset import SeismicGatherDataset, SharedGatherDataset
from jtr_marine.model_loader import load_checkpoint
from jtr_marine.baselines.mssa import mssa_interpolate_validity_aware
from jtr_marine.metrics import compute_snr

RANKS = [3, 5, 10, 15, 20, 30]
N_ITERS = [3, 5]
PAPER_CFG = {"rank": 10, "n_iter": 3}
MASK_SEED = 20260501           # fixed: identical masks for every method and run


def load_survey(cfg, path, length):
    dc = cfg.data
    return SeismicGatherDataset(
        data_path=path, streamer_length=length,
        max_streamer_length=dc.max_streamer_length,
        dt_target_ms=dc.dt_target_ms, t_max_s=dc.t_max_s,
        x_max_m=dc.x_max_m, dx_target_m=dc.dx_target_m,
        masking_ratios=[0.3], masking_strategy="random",
        block_size=dc.block_size, clip_sigma=dc.clip_sigma,
        normalize_per_trace=dc.normalize_per_trace,
        apply_filter=dc.apply_filter, filter_f_cut=dc.filter_f_cut,
        val_fraction=dc.val_fraction, random_seed=dc.random_seed,
        split="train").gathers


def masked_set(cfg, gathers, indices, length, ratio, seed):
    dc = cfg.data
    return SharedGatherDataset(
        gathers=gathers, indices=np.asarray(indices, dtype=np.int64),
        geometry_code=length / dc.max_streamer_length,
        n_time=int(round(dc.t_max_s * 1000.0 / dc.dt_target_ms)),
        n_offsets=int(round(dc.x_max_m / dc.dx_target_m)) + 1,
        masking_ratios=[ratio], masking_strategy="random",
        block_size=dc.block_size, random_seed=seed)


def unpack(s):
    return (s["target"].squeeze().numpy(), s["input"][0].numpy(),
            s["trace_mask"].numpy(), s["validity"].numpy())


def snr(pred, tgt, mask, valid, missing_only=True):
    t = lambda a: torch.from_numpy(np.asarray(a, np.float32))
    return compute_snr(t(pred)[None, None], t(tgt)[None, None], t(mask)[None],
                       t(valid)[None], missing_only=missing_only).item()


def run_mssa(masked, mask, valid, rank, n_iter):
    p = mssa_interpolate_validity_aware(masked, mask, valid, rank=rank, n_iter=n_iter)
    return p * (1 - mask)[None, :] + masked * mask[None, :]


def usable(smp, min_valid=20):
    """Exclude degenerate gathers: too few recorded traces, or no energy on
    the missing traces (SNR undefined). Same criterion as the cross-survey
    evaluation (min_n_valid = 20)."""
    tg, _, mk, va = smp
    miss = ((1 - mk) * va) > 0
    return va.sum() >= min_valid and miss.any() and (tg[:, miss] ** 2).sum() > 0


def hankel_size(n_valid):
    return n_valid // 2 + 1


# ───────────────────────────── tune ─────────────────────────────
def tune(cfg, args):
    dc = cfg.data
    by_len = {}
    for path, L in zip(dc.dataset_paths, dc.streamer_lengths):
        by_len.setdefault(float(L), []).append(path)

    results = {}
    t0 = time.time()
    for L, paths in sorted(by_len.items()):
        samples, n_skipped = [], 0
        for path in paths:
            g = load_survey(cfg, path, L)
            n_train = int(len(g) * 0.70)
            # evenly spaced, away from the ends of the line (partial shots)
            idx = np.linspace(0.05 * n_train, 0.95 * n_train - 1, args.n_tune).astype(int)
            for r in args.ratios_tune:
                ds = masked_set(cfg, g, idx, L, r, MASK_SEED + int(r * 1000))
                for i in range(len(ds)):
                    smp = unpack(ds[i])
                    if usable(smp):
                        samples.append(smp)
                    else:
                        n_skipped += 1
        n_valid = int(samples[0][3].sum())
        H = hankel_size(n_valid)
        ranks = [k for k in RANKS if k <= H - 3]
        if n_skipped:
            print(f"  ({n_skipped} degenerate gathers skipped)")
        print(f"\n{L:.0f} m: {len(samples)} training samples, "
              f"{n_valid} recorded traces, Hankel ~{H}x{n_valid - H + 1}, "
              f"ranks tested {ranks}")
        res = {}
        for k in ranks:
            for it in N_ITERS:
                v = [snr(run_mssa(m, mk, va, k, it), tg, mk, va)
                     for tg, m, mk, va in samples]
                res[f"r{k}_i{it}"] = {"rank": k, "n_iter": it, "snr_miss": float(np.mean(v))}
                print(f"  rank {k:>2}, n_iter {it}: SNR_miss {np.mean(v):+.3f} dB   "
                      f"[{(time.time()-t0)/60:.0f} min]")
        best = max(res.values(), key=lambda d: d["snr_miss"])
        paper = res.get(f"r{PAPER_CFG['rank']}_i{PAPER_CFG['n_iter']}")
        results[str(int(L))] = {"hankel": H, "n_valid": n_valid, "grid": res,
                                "best": best, "paper": paper}
        print(f"  -> best: rank {best['rank']}, n_iter {best['n_iter']} "
              f"({best['snr_miss']:+.3f} dB); paper config "
              f"{paper['snr_miss']:+.3f} dB" if paper else "")

    out = Path(args.out) / "mssa_rank_tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


# ───────────────────────────── eval ─────────────────────────────
def evaluate(cfg, args):
    dc = cfg.data
    tuned = json.load(open(Path(args.out) / "mssa_rank_tuning.json"))
    best = {int(k): (v["best"]["rank"], v["best"]["n_iter"]) for k, v in tuned.items()}
    print("per-geometry configuration:", {k: f"rank {r}, n_iter {i}" for k, (r, i) in best.items()})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, cfg, device)
    model.eval()

    surveys = [(p, float(L), load_survey(cfg, p, L))
               for p, L in zip(dc.dataset_paths, dc.streamer_lengths)]
    out = {"config": {"paper": PAPER_CFG, "tuned": {str(k): v for k, v in best.items()},
                      "mask_seed": MASK_SEED}}
    t0 = time.time()
    for r in args.ratios:
        key = f"{int(round(r*100))}pct"
        rec = {m: [] for m in ("dl", "mssa_paper", "mssa_tuned")}
        rec_full = {m: [] for m in rec}
        geom = []
        for path, L, g in surveys:
            n = len(g)
            test_idx = np.arange(int(n * 0.70) + int(n * 0.15), n)
            test_idx = test_idx[::args.stride]
            if args.n_test:
                test_idx = test_idx[:args.n_test]
            ds = masked_set(cfg, g, test_idx, L, r, MASK_SEED + int(r * 1000))
            rk, it = best[int(L)]
            for i in range(len(ds)):
                s = ds[i]
                tg, m, mk, va = unpack(s)
                if not usable((tg, m, mk, va)):
                    continue
                with torch.no_grad():
                    dl = model(s["input"].unsqueeze(0).to(device),
                               torch.tensor([s["geometry"].item()], device=device)
                               ).cpu().squeeze().numpy()
                preds = {"dl": dl * (1 - mk)[None, :] + m * mk[None, :],
                         "mssa_paper": run_mssa(m, mk, va, PAPER_CFG["rank"], PAPER_CFG["n_iter"]),
                         "mssa_tuned": (run_mssa(m, mk, va, rk, it)
                                        if (rk, it) != (PAPER_CFG["rank"], PAPER_CFG["n_iter"])
                                        else None)}
                if preds["mssa_tuned"] is None:
                    preds["mssa_tuned"] = preds["mssa_paper"]
                for meth, p in preds.items():
                    rec[meth].append(snr(p, tg, mk, va, True))
                    rec_full[meth].append(snr(p, tg, mk, va, False))
                geom.append(int(L))
        out[key] = {"geometry": geom,
                    **{f"{m}_snr_miss": v for m, v in rec.items()},
                    **{f"{m}_snr_full": v for m, v in rec_full.items()}}
        json.dump(out, open(Path(args.out) / "mssa_rank_eval.json", "w"))

        G = np.array(geom)
        print(f"\n--- {key} (n={len(G)}, {(time.time()-t0)/60:.0f} min) ---")
        for L in sorted(set(G)):
            sel = G == L
            print(f"  {L:>4} m:  DL {np.mean(np.array(rec['dl'])[sel]):+6.2f}  "
                  f"MSSA r10 {np.mean(np.array(rec['mssa_paper'])[sel]):+6.2f}  "
                  f"MSSA tuned {np.mean(np.array(rec['mssa_tuned'])[sel]):+6.2f}")
        dl, mp, mt = (np.array(rec[m]) for m in ("dl", "mssa_paper", "mssa_tuned"))
        print(f"  ALL :  DL {dl.mean():+6.2f}  MSSA r10 {mp.mean():+6.2f}  "
              f"MSSA tuned {mt.mean():+6.2f}   "
              f"gain of tuning {mt.mean()-mp.mean():+.2f} dB   "
              f"DL - MSSA tuned {dl.mean()-mt.mean():+.2f} dB   "
              f"DL wins {int((dl > mt).sum())}/{len(dl)}")
        try:
            from scipy.stats import wilcoxon
            print(f"         paired Wilcoxon DL > MSSA tuned: "
                  f"p = {wilcoxon(dl, mt, alternative='greater').pvalue:.1e}")
        except Exception:
            pass
    print(f"\nSaved {Path(args.out) / 'mssa_rank_eval.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "eval"], required=True)
    ap.add_argument("--checkpoint", default=str(CKPT_ROOT / "v6_grid25m/within_geology/best.pt"))
    ap.add_argument("--n-tune", type=int, default=4,
                    help="training gathers per survey used for tuning")
    ap.add_argument("--ratios-tune", nargs="+", type=float, default=[0.3])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.3, 0.5, 0.8])
    ap.add_argument("--stride", type=int, default=3,
                    help="use one test gather every STRIDE (default 3: 92 of 276)")
    ap.add_argument("--n-test", type=int, default=None, help="quick test only")
    ap.add_argument("--out", default=str(RESULTS_ROOT / "mssa_rank_geometry"))
    args = ap.parse_args()
    cfg = get_config()
    tune(cfg, args) if args.mode == "tune" else evaluate(cfg, args)


if __name__ == "__main__":
    main()
