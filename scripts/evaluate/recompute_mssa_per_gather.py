"""
recompute_mssa_per_gather.py — MSSA on the 276 gathers of the internal test
set (Table 5), with per-gather values saved for the 95% CIs and the paired
Wilcoxon test.

Uses THE SAME masks as DL and POCS in recompute_with_per_gather_lists.py (same
seed), for the three-way comparison DL vs POCS vs MSSA.

Usage:
    python scripts/evaluate/recompute_mssa_per_gather.py \\
        --rank 10 \\
        --n-iter 3
(rank 10, n_iter 3 is the MSSA configuration of the paper; the script default
for --n-iter is 10.)

Expected run time (A6000 GPU):
  - MSSA per gather: ~3-5 seconds (SVD on the GPU)
  - 7 ratios × 276 gathers × 5 s = ~2.7 hours

Expected run time (CPU):
  - MSSA per gather: ~30 seconds
  - 7 ratios × 276 × 30 s = ~16 hours

Output:
  RESULTS_ROOT/statistics/mssa_per_gather_rank{rank}.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import RESULTS_ROOT

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import ConcatDataset

from config import get_config
from jtr_marine.data.dataset import SeismicGatherDataset, SharedGatherDataset
from jtr_marine.baselines.mssa import mssa_interpolate_validity_aware
from jtr_marine.metrics import (
    compute_snr, compute_ssim_metric, compute_fk_energy_ratio,
)


# Optional fixed mask-seed offsets, per strategy (--mask-seed-offset random=K).
# Default (empty): offset = abs(hash(strategy)) % 1000, which depends on
# PYTHONHASHSEED. An explicit offset reproduces a mask realization obtained
# with an unknown hash seed (e.g. the run behind Table 5, see REPRODUCE.md).
MASK_SEED_OFFSET = {}


def parse_mask_seed_offsets(items):
    """['random=123', 'block=45'] -> {'random': 123, 'block': 45}"""
    out = {}
    for it in items or []:
        name, _, val = it.partition("=")
        if not val:
            raise SystemExit(f"--mask-seed-offset expects STRATEGY=K, got {it!r}")
        out[name] = int(val)
    return out


def strategy_seed_offset(masking_strategy):
    if masking_strategy in MASK_SEED_OFFSET:
        return MASK_SEED_OFFSET[masking_strategy]
    return abs(hash(masking_strategy)) % 1000


def build_test_set(config, masking_strategy, masking_ratio):
    """Test set with contiguous split, same protocol as the compact U-Net."""
    dc = config.data
    n_t = int(round(dc.t_max_s * 1000.0 / dc.dt_target_ms))
    n_x = int(round(dc.x_max_m / dc.dx_target_m)) + 1

    test_sets = []
    for path, length in zip(dc.dataset_paths, dc.streamer_lengths):
        full_ds = SeismicGatherDataset(
            data_path           = path,
            streamer_length     = length,
            max_streamer_length = dc.max_streamer_length,
            dt_target_ms        = dc.dt_target_ms,
            t_max_s             = dc.t_max_s,
            x_max_m             = dc.x_max_m,
            dx_target_m         = dc.dx_target_m,
            masking_ratios      = [masking_ratio],
            masking_strategy    = masking_strategy,
            block_size          = dc.block_size,
            clip_sigma          = dc.clip_sigma,
            normalize_per_trace = dc.normalize_per_trace,
            apply_filter        = dc.apply_filter,
            filter_f_cut        = dc.filter_f_cut,
            val_fraction        = dc.val_fraction,
            random_seed         = dc.random_seed,
            split               = "train",
        )
        all_gathers = full_ds.gathers
        n_total = len(all_gathers)
        all_idx = np.arange(n_total, dtype=np.int64)
        n_train_g = int(n_total * 0.70)
        n_val_g   = int(n_total * 0.15)
        test_idx  = all_idx[n_train_g + n_val_g:]
        geom_code = length / dc.max_streamer_length

        # IMPORTANT: same seed as recompute_with_per_gather_lists.py,
        # so that the masks are THE SAME as for DL and POCS
        strategy_seed = dc.random_seed + strategy_seed_offset(masking_strategy)

        test_sets.append(SharedGatherDataset(
            gathers          = all_gathers,
            indices          = test_idx,
            geometry_code    = geom_code,
            n_time           = n_t,
            n_offsets        = n_x,
            masking_ratios   = [masking_ratio],
            masking_strategy = masking_strategy,
            block_size       = dc.block_size,
            random_seed      = strategy_seed,
        ))

    return ConcatDataset(test_sets)


def evaluate_one_setting(config, masking_strategy, masking_ratio,
                           rank, n_iter, dt_seconds, dx_meters):
    """
    Evaluate MSSA on all 276 gathers at one (strategy, ratio).
    Returns per-gather metric lists.
    """
    test_dataset = build_test_set(config, masking_strategy, masking_ratio)
    n_gather = len(test_dataset)

    mssa_snr_miss = []
    mssa_snr_full = []
    mssa_ssim     = []
    mssa_fk       = []

    t0 = time.time()

    for idx in range(n_gather):
        sample = test_dataset[idx]
        target     = sample["target"].squeeze().numpy()       # (T, O)
        masked     = sample["input"][0].numpy()                # (T, O)
        trace_mask = sample["trace_mask"].numpy()              # (O,)
        validity   = sample["validity"].numpy()                # (O,)

        # MSSA inference
        mssa_pred = mssa_interpolate_validity_aware(
            masked, trace_mask, validity,
            rank=rank, n_iter=n_iter,
        )

        # Trace replacement (same as POCS): preserve known traces exactly
        mssa_recon = mssa_pred * (1 - trace_mask)[np.newaxis, :] + \
                     masked    * trace_mask[np.newaxis, :]

        # Metrics
        t_torch = torch.from_numpy(target[None, None]).float()
        m_torch = torch.from_numpy(trace_mask[None]).float()
        v_torch = torch.from_numpy(validity[None]).float()
        p_torch = torch.from_numpy(mssa_recon[None, None]).float()

        mssa_snr_miss.append(compute_snr(p_torch, t_torch, m_torch, v_torch,
                                            missing_only=True).item())
        mssa_snr_full.append(compute_snr(p_torch, t_torch, m_torch, v_torch,
                                            missing_only=False).item())
        mssa_ssim.append(compute_ssim_metric(p_torch, t_torch).item())
        mssa_fk.append(compute_fk_energy_ratio(p_torch, t_torch,
                                                  dt=dt_seconds,
                                                  dx=dx_meters).item())

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (n_gather - idx - 1)
            print(f"    [{idx+1}/{n_gather}] elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    return {
        "n":              n_gather,
        "rank":           rank,
        "n_iter":         n_iter,
        "mssa_snr_miss":  mssa_snr_miss,
        "mssa_snr_full":  mssa_snr_full,
        "mssa_ssim":      mssa_ssim,
        "mssa_fk":        mssa_fk,
    }


def main(rank, n_iter, output_dir, strategies, ratios):
    config = get_config()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"MSSA PER-GATHER EVALUATION")
    print(f"{'='*70}")
    print(f"rank:          {rank}")
    print(f"n_iter:        {n_iter}")
    print(f"Strategies:    {strategies}")
    print(f"Ratios:        {[f'{int(r*100)}%' for r in ratios]}")
    print(f"Device:        {device}")
    print(f"{'='*70}\n")

    dt_seconds = config.data.dt_target_ms * 1e-3
    dx_meters  = float(config.data.dx_target_m)

    all_results = {}
    t_global = time.time()

    for strategy in strategies:
        all_results[strategy] = {}
        for ratio in ratios:
            ratio_key = f"{int(ratio*100)}pct"
            print(f"\n--- Strategy: {strategy} | Ratio: {ratio_key} ---")
            t_exp = time.time()
            results = evaluate_one_setting(
                config, strategy, ratio,
                rank, n_iter,
                dt_seconds, dx_meters,
            )
            all_results[strategy][ratio_key] = results

            mssa_m = np.mean(results["mssa_snr_miss"])
            print(f"  Completed in {time.time()-t_exp:.0f}s (n={results['n']})")
            print(f"    MSSA SNR_miss = {mssa_m:+.2f} dB")

    t_total = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"DONE. Total time: {t_total/60:.1f} min")
    print(f"{'='*70}\n")

    output_file = out_dir / f"mssa_per_gather_rank{rank}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=10,
                        help="SVD truncation rank (default 10)")
    parser.add_argument("--n-iter", type=int, default=10,
                        help="MSSA iterations (default 10)")
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT / "statistics"))
    parser.add_argument("--strategies", nargs="+", default=["random"])
    parser.add_argument("--ratios", nargs="+", type=float,
                        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--mask-seed-offset", nargs="+", metavar="STRATEGY=K",
                        help="fixed mask-seed offset per strategy instead of "
                             "abs(hash(strategy)) %% 1000 (default: not used)")
    args = parser.parse_args()
    MASK_SEED_OFFSET.update(parse_mask_seed_offsets(args.mask_seed_offset))
    if MASK_SEED_OFFSET:
        print(f"Mask-seed offsets fixed: {MASK_SEED_OFFSET}")

    main(rank=args.rank, n_iter=args.n_iter,
         output_dir=args.output_dir,
         strategies=args.strategies,
         ratios=args.ratios)
