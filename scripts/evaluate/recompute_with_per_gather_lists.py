"""
recompute_with_per_gather_lists.py — DL + POCS on the 276 gathers of the
internal test set (Table 5), saving the metrics of EVERY gather (not only the
aggregates).

The per-gather values are needed for:
  - 95% CIs with n=276
  - paired Wilcoxon test (DL vs POCS on the same gathers)
  - bootstrap CI of the difference
  - Cliff's δ (non-parametric effect size)

What is computed:
  1. ALL the test gathers (276) are evaluated
  2. The complete per-gather LISTS are saved, not only mean/std
  3. BOTH SNR_missing and SNR_full, for both DL and POCS
  4. Same gathers and same mask for DL and POCS (paired)

Usage:
    python scripts/evaluate/recompute_with_per_gather_lists.py \\
        --checkpoint checkpoints/v6_grid25m/within_geology/best.pt \\
        --n-iter-pocs 100
(the paper uses 100 POCS iterations; the script default is 50.)

Option --skip-pocs: only the DL model is evaluated. The masks do not depend on
POCS, so with the same PYTHONHASHSEED (or --mask-seed-offset) they are
identical to those of the complete runs. In the JSON the pocs_* fields are null.

Expected run time: dominated by POCS (~5 s per gather, against ~0.1 s per
gather for DL); a few minutes with --skip-pocs.

Output:
  RESULTS_ROOT/statistics/per_gather_results.json
    format:
    {
      "random": {
        "30pct": {
          "n": 276,
          "gather_ids": [0, 1, ..., 275],
          "dl_snr_miss":   [v1, v2, ..., v276],
          "dl_snr_full":   [...],
          "dl_ssim":       [...],
          "dl_fk":         [...],
          "pocs_snr_miss": [...],
          "pocs_snr_full": [...],
          "pocs_ssim":     [...],
          "pocs_fk":       [...]
        },
        ...
      }
    }
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT, RESULTS_ROOT

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from copy import deepcopy
from torch.utils.data import ConcatDataset, DataLoader

from config import get_config
from jtr_marine.data.dataset import SeismicGatherDataset, SharedGatherDataset
from jtr_marine.models.unet import build_model
from jtr_marine.model_loader import load_checkpoint
from jtr_marine.baselines.pocs import pocs_interpolate_validity_aware
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
    """Test set with contiguous split (no leakage), same setup as the training of the compact U-Net."""
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


def evaluate_one_setting(model, device, config, masking_strategy, masking_ratio,
                           dt_seconds, dx_meters, n_iter_pocs=50, skip_pocs=False):
    """
    For one (strategy, ratio): evaluate ALL 276 gathers both for DL and POCS,
    on identical missing patterns. Return per-gather metric lists.
    """
    test_dataset = build_test_set(config, masking_strategy, masking_ratio)
    n_gather = len(test_dataset)

    # Per-gather storage
    dl_snr_miss   = []
    dl_snr_full   = []
    dl_ssim       = []
    dl_fk         = []
    pocs_snr_miss = []
    pocs_snr_full = []
    pocs_ssim     = []
    pocs_fk       = []

    model.eval()
    t0 = time.time()

    for idx in range(n_gather):
        sample = test_dataset[idx]
        target     = sample["target"].squeeze().numpy()       # (T, O)
        masked     = sample["input"][0].numpy()                # (T, O)
        trace_mask = sample["trace_mask"].numpy()              # (O,)
        validity   = sample["validity"].numpy()                # (O,)
        geometry   = sample["geometry"].item()                  # scalar

        # ── DL inference ──────────────────────────────────────────────────
        x = sample["input"].unsqueeze(0).to(device)              # (1, 2, T, O)
        g = torch.tensor([geometry], device=device)
        with torch.no_grad():
            dl_pred = model(x, g).cpu().squeeze().numpy()        # (T, O)

        # Trace replacement DL
        dl_recon = dl_pred * (1 - trace_mask)[np.newaxis, :] + \
                   masked  * trace_mask[np.newaxis, :]

        # ── POCS inference ────────────────────────────────────────────────
        if not skip_pocs:
            pocs_pred = pocs_interpolate_validity_aware(
                masked, trace_mask, validity, n_iter=n_iter_pocs,
            )
            pocs_recon = pocs_pred * (1 - trace_mask)[np.newaxis, :] + \
                         masked    * trace_mask[np.newaxis, :]

        # ── Compute metrics ───────────────────────────────────────────────
        t_torch = torch.from_numpy(target[None, None]).float()
        m_torch = torch.from_numpy(trace_mask[None]).float()
        v_torch = torch.from_numpy(validity[None]).float()

        dl_torch   = torch.from_numpy(dl_recon[None, None]).float()

        dl_snr_miss.append(  compute_snr(dl_torch, t_torch, m_torch, v_torch,
                                            missing_only=True).item())
        dl_snr_full.append(  compute_snr(dl_torch, t_torch, m_torch, v_torch,
                                            missing_only=False).item())
        dl_ssim.append(      compute_ssim_metric(dl_torch, t_torch).item())
        dl_fk.append(        compute_fk_energy_ratio(dl_torch, t_torch,
                                                       dt=dt_seconds,
                                                       dx=dx_meters).item())

        if not skip_pocs:
            pocs_torch = torch.from_numpy(pocs_recon[None, None]).float()
            pocs_snr_miss.append(compute_snr(pocs_torch, t_torch, m_torch, v_torch,
                                                missing_only=True).item())
            pocs_snr_full.append(compute_snr(pocs_torch, t_torch, m_torch, v_torch,
                                                missing_only=False).item())
            pocs_ssim.append(    compute_ssim_metric(pocs_torch, t_torch).item())
            pocs_fk.append(      compute_fk_energy_ratio(pocs_torch, t_torch,
                                                           dt=dt_seconds,
                                                           dx=dx_meters).item())

        # Progress every 50 gathers
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (n_gather - idx - 1)
            print(f"    [{idx+1}/{n_gather}] elapsed={elapsed:.0f}s "
                  f"ETA={eta:.0f}s")

    return {
        "n":              n_gather,
        "gather_ids":     list(range(n_gather)),
        "dl_snr_miss":    dl_snr_miss,
        "dl_snr_full":    dl_snr_full,
        "dl_ssim":        dl_ssim,
        "dl_fk":          dl_fk,
        "pocs_snr_miss":  None if skip_pocs else pocs_snr_miss,
        "pocs_snr_full":  None if skip_pocs else pocs_snr_full,
        "pocs_ssim":      None if skip_pocs else pocs_ssim,
        "pocs_fk":        None if skip_pocs else pocs_fk,
    }


def main(checkpoint, output_dir, strategies, ratios, n_iter_pocs=50,
         skip_pocs=False):
    config = get_config()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"PER-GATHER EVALUATION — n=ALL test gathers (~276), DL + POCS")
    print(f"{'='*70}")
    print(f"Checkpoint:    {checkpoint}")
    print(f"Output:        {out_dir}")
    print(f"Strategies:    {strategies}")
    print(f"Ratios:        {[f'{int(r*100)}%' for r in ratios]}")
    print(f"POCS n_iter:   {'skipped' if skip_pocs else n_iter_pocs}")
    print(f"Device:        {device}")
    print(f"{'='*70}\n")

    # Build & load model
    print("Loading model...")
    model, ckpt = load_checkpoint(checkpoint, config, device)
    print(f"  Loaded checkpoint, val_snr = {ckpt.get('val_snr_db', '?')}\n")

    # Pre-compute physical constants for FK metric
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
                model, device, config, strategy, ratio,
                dt_seconds, dx_meters, n_iter_pocs=n_iter_pocs,
                skip_pocs=skip_pocs,
            )
            all_results[strategy][ratio_key] = results

            # Summary
            dl_m  = np.mean(results["dl_snr_miss"])
            print(f"  Completed in {time.time()-t_exp:.0f}s "
                  f"(n={results['n']})")
            print(f"    DL   SNR_miss = {dl_m:+.2f} dB")
            if not skip_pocs:
                po_m  = np.mean(results["pocs_snr_miss"])
                print(f"    POCS SNR_miss = {po_m:+.2f} dB")
                print(f"    Δ             = {dl_m-po_m:+.2f} dB")

    t_total = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"DONE. Total time: {t_total/60:.1f} min")
    print(f"{'='*70}\n")

    # Save
    output_file = out_dir / "per_gather_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {output_file}")
    print(f"  Run compute_statistics.py next.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default=str(CKPT_ROOT / "v6_grid25m/within_geology/best.pt"))
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT / "statistics"))
    parser.add_argument("--strategies", nargs="+",
                        default=["random", "block", "far"])
    parser.add_argument("--ratios", nargs="+", type=float,
                        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--n-iter-pocs", type=int, default=50)
    parser.add_argument("--skip-pocs", action="store_true",
                        help="DL only (masks unchanged; pocs_* saved as null)")
    parser.add_argument("--mask-seed-offset", nargs="+", metavar="STRATEGY=K",
                        help="fixed mask-seed offset per strategy instead of "
                             "abs(hash(strategy)) %% 1000 (default: not used)")
    args = parser.parse_args()
    MASK_SEED_OFFSET.update(parse_mask_seed_offsets(args.mask_seed_offset))
    if MASK_SEED_OFFSET:
        print(f"Mask-seed offsets fixed: {MASK_SEED_OFFSET}")

    main(checkpoint   = args.checkpoint,
         output_dir   = args.output_dir,
         strategies   = args.strategies,
         ratios       = args.ratios,
         n_iter_pocs  = args.n_iter_pocs,
         skip_pocs    = args.skip_pocs)
