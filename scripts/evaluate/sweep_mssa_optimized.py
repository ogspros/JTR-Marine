"""
sweep_mssa_optimized.py — Reduced (faster) sweep of the MSSA parameters.

Reductions with respect to the full sweep (sweep_mssa_full.py):
  1. The 1500 m lines are SKIPPED (MSSA always gives ~0 dB there, a
     structural limit)
     → 4 lines instead of 6
  2. FEWER configurations (only the rank is varied; the configurations with
     more iterations or a different damping are dropped)
     → 4 configurations instead of 8
  3. FEWER samples (10 instead of 20; the standard error is already small)
  4. Incremental CHECKPOINT: the JSON is saved after every configuration
     → the partial results survive an interrupted run

Lines tested:
  - IT90-65    (3000m, 66% coverage)
  - IT89-33    (3000m, 66% coverage)
  - CROP-M31   (4500m, 97% coverage)
  - CROP-M32   (4500m, 97% coverage)

Configurations tested:
  - baseline_current (rank=30, n_iter=2, damping=0.5) ← for comparison
  - low_rank_10
  - low_rank_20
  - high_rank_50

As in the full sweep, the gathers are drawn at random (fixed seed) from the
last 15% of each line.

Estimated run time: ~2 hours (against ~14 h for the full sweep)

Usage:
  python -u scripts/evaluate/sweep_mssa_optimized.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import RESULTS_ROOT
from jtr_marine.baselines.mssa import mssa_interpolate_validity_aware
from jtr_marine.data.dataset import SeismicGatherDataset
from config import get_config


# ============================================================================
# MSSA configurations to test (REDUCED set)
# ============================================================================

MSSA_CONFIGS = [
    # (name,              rank, n_iter, damping)
    ("baseline_current",    30,      2,    0.5),
    ("low_rank_10",         10,      2,    0.5),
    ("low_rank_20",         20,      2,    0.5),
    ("high_rank_50",        50,      2,    0.5),
]


# ============================================================================
# Helpers
# ============================================================================

def snr_miss_db(gt, recon, mask, validity):
    mv = (mask == 0) & (validity > 0.5)
    if mv.sum() == 0:
        return float("nan")
    err = gt[:, mv] - recon[:, mv]
    sig_e = (gt[:, mv] ** 2).sum()
    err_e = (err ** 2).sum()
    if err_e < 1e-15:
        return float("inf")
    return 10.0 * np.log10(sig_e / err_e)


def load_dataset(sgy_path, streamer_length, seed):
    cfg = get_config(); dc = cfg.data
    ds = SeismicGatherDataset(
        data_path=sgy_path,
        streamer_length=streamer_length,
        max_streamer_length=dc.max_streamer_length,
        dt_target_ms=dc.dt_target_ms, t_max_s=dc.t_max_s,
        x_max_m=dc.x_max_m, dx_target_m=dc.dx_target_m,
        masking_ratios=[0.3], masking_strategy="random",
        block_size=dc.block_size, clip_sigma=dc.clip_sigma,
        normalize_per_trace=dc.normalize_per_trace,
        apply_filter=dc.apply_filter, filter_f_cut=dc.filter_f_cut,
        val_fraction=0.0, random_seed=seed, split="train",
    )
    return ds


def compute_validity(gather):
    col_energy = (gather ** 2).sum(axis=0)
    return (col_energy > 1e-10).astype(np.float32)


def make_random_mask(validity, ratio, seed):
    rng = np.random.default_rng(seed)
    valid_idx = np.where(validity > 0.5)[0]
    n_valid = len(valid_idx)
    n_miss = max(1, int(ratio * n_valid))
    miss = rng.choice(valid_idx, size=n_miss, replace=False)
    mask = np.ones(len(validity), dtype=np.float32)
    mask[miss] = 0
    return mask


def save_checkpoint(out_dir, results, partial=True):
    """Save the results (partial after each configuration, or final)."""
    suffix = "partial" if partial else "final"
    path = out_dir / f"mssa_sweep_{suffix}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Gathers per (dataset, ratio) (default 10)")
    parser.add_argument("--out-dir", default=str(RESULTS_ROOT / "mssa_sweep"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_config()
    # SKIP 1500m datasets (MSSA structurally ~0 there)
    training_pool = [
        ("IT90-65",    cfg.data.dataset_paths[2], cfg.data.streamer_lengths[2]),
        ("IT89-33",    cfg.data.dataset_paths[3], cfg.data.streamer_lengths[3]),
        ("CROP-M31",   cfg.data.dataset_paths[4], cfg.data.streamer_lengths[4]),
        ("CROP-M32",   cfg.data.dataset_paths[5], cfg.data.streamer_lengths[5]),
    ]

    ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    seed_base = 42

    print("=" * 78)
    print("MSSA PARAMETER SWEEP — OPTIMIZED (skip 1500m, reduced configs/samples)")
    print("=" * 78)
    print(f"Datasets: {len(training_pool)} (SKIP 1500m: MSSA always ~0 there)")
    for name, _, L in training_pool:
        print(f"           {name} ({L}m)")
    print(f"Ratios:   {ratios}")
    print(f"Configs:  {len(MSSA_CONFIGS)} (reduced)")
    for name, r, ni, d in MSSA_CONFIGS:
        print(f"           {name}: rank={r}, n_iter={ni}, damping={d}")
    print(f"Samples:  {args.n_samples} per (dataset, ratio)")
    n_calls = len(MSSA_CONFIGS) * len(training_pool) * len(ratios) * args.n_samples
    print(f"Total MSSA calls: {n_calls}")
    print()

    # Pre-load datasets
    print("Loading datasets...", flush=True)
    datasets = {}
    for name, path, length in training_pool:
        print(f"  {name} ({length}m) ...", flush=True)
        ds = load_dataset(path, length, seed=seed_base)
        rng = np.random.default_rng(seed_base)
        n_total = len(ds.gathers)
        test_start = int(0.85 * n_total)
        test_idx = rng.choice(
            np.arange(test_start, n_total),
            size=min(args.n_samples, n_total - test_start),
            replace=False,
        )
        gathers = []
        for idx in test_idx:
            g = ds.gathers[idx].astype(np.float32)
            if g.ndim == 3:
                g = g[0]
            gathers.append(g)
        datasets[name] = {
            "gathers": gathers,
            "length": length,
            "n_gathers": len(gathers),
        }
        print(f"    -> {len(gathers)} test gathers", flush=True)

    # Run sweep
    print("\nRunning sweep...", flush=True)
    t0 = time.time()
    results = []

    for ci, (cfg_name, rank, n_iter, damping) in enumerate(MSSA_CONFIGS):
        config_t0 = time.time()
        print(f"\n--- Config {ci+1}/{len(MSSA_CONFIGS)}: {cfg_name} "
              f"(rank={rank}, n_iter={n_iter}, damping={damping}) ---",
              flush=True)

        for ds_name, ds_data in datasets.items():
            ds_t0 = time.time()
            for ratio in ratios:
                snrs = []
                for gi, gt in enumerate(ds_data["gathers"]):
                    validity = compute_validity(gt)
                    mask = make_random_mask(
                        validity, ratio,
                        seed=seed_base + gi * 100 + int(ratio * 100),
                    )
                    masked = gt * mask[None, :]
                    try:
                        recon = mssa_interpolate_validity_aware(
                            masked, mask, validity,
                            rank=rank, n_iter=n_iter, damping=damping,
                        )
                        snr = snr_miss_db(gt, recon, mask, validity)
                        if np.isfinite(snr):
                            snrs.append(snr)
                    except Exception as e:
                        print(f"    [WARN] {ds_name} ratio={ratio} "
                              f"gather={gi}: {e}", flush=True)

                snr_mean = float(np.mean(snrs)) if snrs else float("nan")
                snr_std = float(np.std(snrs)) if snrs else float("nan")
                results.append({
                    "config": cfg_name,
                    "rank": rank,
                    "n_iter": n_iter,
                    "damping": damping,
                    "dataset": ds_name,
                    "streamer_length": ds_data["length"],
                    "ratio": ratio,
                    "snr_mean": snr_mean,
                    "snr_std": snr_std,
                    "n_samples": len(snrs),
                })

            ds_elapsed = time.time() - ds_t0
            row_means = [r["snr_mean"] for r in results
                         if r["config"] == cfg_name and r["dataset"] == ds_name
                         and np.isfinite(r["snr_mean"])]
            print(f"  {ds_name:<12s} ({int(ds_data['length']):4d}m): "
                  f"mean over ratios = {np.mean(row_means):+.2f} dB "
                  f"({ds_elapsed:.0f}s)",
                  flush=True)

        # CHECKPOINT — save after every configuration
        cp_path = save_checkpoint(out_dir, results, partial=True)
        config_elapsed = time.time() - config_t0
        print(f"  >>> Config completed in {config_elapsed/60:.1f} min. "
              f"Checkpoint saved: {cp_path}", flush=True)

    elapsed = time.time() - t0
    print(f"\nSweep completed in {elapsed/60:.1f} minutes", flush=True)

    # Final save
    save_checkpoint(out_dir, results, partial=False)
    print(f"Saved final: {out_dir / 'mssa_sweep_final.json'}")

    # ===== Aggregations =====

    print("\n" + "=" * 78)
    print("AGGREGATED RESULTS — overall (avg across 4 datasets, 7 ratios)")
    print("=" * 78)
    print(f"{'Config':<22s} {'rank':>5s} {'n_iter':>7s} {'damp':>6s} "
          f"{'SNR mean':>12s}")
    overall = {}
    for cfg_name, rank, n_iter, damping in MSSA_CONFIGS:
        means = [r["snr_mean"] for r in results
                 if r["config"] == cfg_name and np.isfinite(r["snr_mean"])]
        overall[cfg_name] = float(np.mean(means)) if means else float("nan")
        print(f"{cfg_name:<22s} {rank:>5d} {n_iter:>7d} {damping:>6.2f} "
              f"{overall[cfg_name]:>+10.2f} dB")

    best_overall = max(overall.items(), key=lambda x: x[1])
    print(f"\n>>> BEST OVERALL: {best_overall[0]} ({best_overall[1]:+.2f} dB)")

    # Per streamer length (3000m, 4500m)
    print("\n" + "=" * 78)
    print("AGGREGATED RESULTS — per streamer length")
    print("=" * 78)
    by_length = defaultdict(lambda: defaultdict(list))
    for r in results:
        if np.isfinite(r["snr_mean"]):
            by_length[r["streamer_length"]][r["config"]].append(r["snr_mean"])

    streamer_lengths = sorted(by_length.keys())
    print(f"{'Config':<22s}", end="")
    for L in streamer_lengths:
        print(f"  {int(L)}m".rjust(11), end="")
    print()
    print("-" * 78)
    best_per_length = {}
    for cfg_name, _, _, _ in MSSA_CONFIGS:
        print(f"{cfg_name:<22s}", end="")
        for L in streamer_lengths:
            mean_val = np.mean(by_length[L][cfg_name])
            print(f"  {mean_val:+.2f} dB".rjust(11), end="")
        print()
    print()
    for L in streamer_lengths:
        best_cfg = max(by_length[L].items(),
                        key=lambda x: np.mean(x[1]))
        best_per_length[L] = (best_cfg[0], float(np.mean(best_cfg[1])))
        print(f"  Best for {int(L)}m: {best_cfg[0]} "
              f"({np.mean(best_cfg[1]):+.2f} dB)")

    # Per ratio (best overall config)
    print("\n" + "=" * 78)
    print(f"BEST CONFIG ({best_overall[0]}) — breakdown by ratio")
    print("=" * 78)
    print(f"{'Ratio':<10s}", end="")
    for L in streamer_lengths:
        print(f"  {int(L)}m".rjust(11), end="")
    print(f"  {'overall':>11s}")
    print("-" * 78)
    for ratio in ratios:
        print(f"{int(ratio*100):<3d}%      ", end="")
        for L in streamer_lengths:
            vals = [r["snr_mean"] for r in results
                    if r["config"] == best_overall[0]
                    and r["streamer_length"] == L
                    and r["ratio"] == ratio
                    and np.isfinite(r["snr_mean"])]
            print(f"  {np.mean(vals):+.2f} dB".rjust(11), end="")
        all_vals = [r["snr_mean"] for r in results
                    if r["config"] == best_overall[0]
                    and r["ratio"] == ratio
                    and np.isfinite(r["snr_mean"])]
        print(f"  {np.mean(all_vals):+.2f} dB".rjust(11))

    if "baseline_current" in overall and best_overall[0] != "baseline_current":
        delta = best_overall[1] - overall["baseline_current"]
        print(f"\n>>> Improvement vs baseline_current: {delta:+.2f} dB")

    # Save summary
    summary = {
        "overall": overall,
        "best_overall": {"config": best_overall[0], "snr": best_overall[1]},
        "best_per_length": {str(L): {"config": v[0], "snr": v[1]}
                            for L, v in best_per_length.items()},
        "configs_tested": [
            {"name": n, "rank": r, "n_iter": ni, "damping": d}
            for n, r, ni, d in MSSA_CONFIGS
        ],
        "note": "1500m datasets skipped (MSSA ~0 dB by structural limit)",
    }
    with open(out_dir / "mssa_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out_dir / 'mssa_sweep_summary.json'}")


if __name__ == "__main__":
    main()
