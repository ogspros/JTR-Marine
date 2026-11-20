"""
sweep_pocs_full.py — Full sweep of the POCS parameters on the six lines of
the training pool.

Each POCS configuration is evaluated on:
  - all 6 lines of the training pool (1500 m × 2, 3000 m × 2, 4500 m × 2)
  - all 7 missing ratios (20, 30, 40, 50, 60, 70, 80%)
  - n_samples gathers per line × ratio (default 20), drawn at random with a
    fixed seed from the last 15% of each line
  - strategy: random masking

Output (RESULTS_ROOT/pocs_sweep):
  - JSON with the mean SNR of every (config, dataset, ratio)
  - aggregated table: best configuration overall
  - table per streamer length: 1500 m, 3000 m, 4500 m
  - table per ratio
  - comparison with the configuration "baseline_old" (t_max 90, t_min 5,
    50 iterations, exponential decay)

Usage:
  python scripts/evaluate/sweep_pocs_full.py                  # default n_samples=20
  python scripts/evaluate/sweep_pocs_full.py --n-samples 10   # faster
  python scripts/evaluate/sweep_pocs_full.py --n-samples 30   # more robust
"""

import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import RESULTS_ROOT
from jtr_marine.baselines.pocs import pocs_interpolate_validity_aware
from jtr_marine.data.dataset import SeismicGatherDataset
from config import get_config


# ============================================================================
# POCS configurations to test
# ============================================================================

POCS_CONFIGS = [
    # (name,            n_iter, t_max,  t_min,  decay)
    ("baseline_old",       50,    90.0,    5.0, "exponential"),
    ("more_iter",         100,    90.0,    5.0, "exponential"),
    ("linear_baseline",   100,    90.0,    5.0, "linear"),
    ("high_tmax_95",      100,    95.0,    5.0, "exponential"),
    ("high_tmax_99",      100,    99.0,    1.0, "exponential"),
    ("aggressive",        200,    99.0,    1.0, "exponential"),
    ("permissive",        100,    80.0,   10.0, "exponential"),
    ("balanced",          100,    95.0,    2.0, "exponential"),
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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Gathers per (dataset, ratio) (default 20)")
    parser.add_argument("--out-dir", default=str(RESULTS_ROOT / "pocs_sweep"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training pool: name, path, streamer length
    cfg = get_config()
    training_pool = [
        ("SISOC-MEPA", cfg.data.dataset_paths[0], cfg.data.streamer_lengths[0]),
        ("GC-01",      cfg.data.dataset_paths[1], cfg.data.streamer_lengths[1]),
        ("IT90-65",    cfg.data.dataset_paths[2], cfg.data.streamer_lengths[2]),
        ("IT89-33",    cfg.data.dataset_paths[3], cfg.data.streamer_lengths[3]),
        ("CROP-M31",   cfg.data.dataset_paths[4], cfg.data.streamer_lengths[4]),
        ("CROP-M32",   cfg.data.dataset_paths[5], cfg.data.streamer_lengths[5]),
    ]

    ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    seed_base = 42

    print("=" * 78)
    print("POCS PARAMETER SWEEP — full training pool, all ratios")
    print("=" * 78)
    print(f"Datasets: {len(training_pool)}")
    print(f"Ratios:   {ratios}")
    print(f"Configs:  {len(POCS_CONFIGS)}")
    print(f"Samples:  {args.n_samples} per (dataset, ratio)")
    n_calls = len(POCS_CONFIGS) * len(training_pool) * len(ratios) * args.n_samples
    print(f"Total POCS calls: {n_calls}")
    print()

    # Pre-load all datasets once
    print("Loading datasets...")
    datasets = {}
    for name, path, length in training_pool:
        print(f"  {name} ({length}m) ...")
        ds = load_dataset(path, length, seed=seed_base)
        # Select n_samples random gathers (fixed seed, for reproducibility)
        rng = np.random.default_rng(seed_base)
        n_total = len(ds.gathers)
        # from the test split: last 15% of the line
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
        print(f"    -> {len(gathers)} test gathers")

    # Run sweep
    print("\nRunning sweep...")
    t0 = time.time()
    results = []
    # results: list of dicts: {config, ds_name, length, ratio, snr_mean, snr_std}

    for ci, (cfg_name, n_iter, t_max, t_min, decay) in enumerate(POCS_CONFIGS):
        print(f"\n--- Config {ci+1}/{len(POCS_CONFIGS)}: {cfg_name} "
              f"(n_iter={n_iter}, t_max={t_max}, t_min={t_min}, {decay}) ---")

        for ds_name, ds_data in datasets.items():
            for ratio in ratios:
                snrs = []
                for gi, gt in enumerate(ds_data["gathers"]):
                    validity = compute_validity(gt)
                    mask = make_random_mask(
                        validity, ratio,
                        seed=seed_base + gi * 100 + int(ratio * 100),
                    )
                    masked = gt * mask[None, :]
                    recon = pocs_interpolate_validity_aware(
                        masked, mask, validity,
                        n_iter=n_iter,
                        threshold_max_pct=t_max,
                        threshold_min_pct=t_min,
                        decay=decay,
                    )
                    snr = snr_miss_db(gt, recon, mask, validity)
                    if np.isfinite(snr):
                        snrs.append(snr)

                snr_mean = float(np.mean(snrs)) if snrs else float("nan")
                snr_std = float(np.std(snrs)) if snrs else float("nan")
                results.append({
                    "config": cfg_name,
                    "n_iter": n_iter,
                    "t_max": t_max,
                    "t_min": t_min,
                    "decay": decay,
                    "dataset": ds_name,
                    "streamer_length": ds_data["length"],
                    "ratio": ratio,
                    "snr_mean": snr_mean,
                    "snr_std": snr_std,
                    "n_samples": len(snrs),
                })

            row_means = [r["snr_mean"] for r in results
                         if r["config"] == cfg_name and r["dataset"] == ds_name]
            print(f"  {ds_name:<12s} ({int(ds_data['length']):4d}m): "
                  f"mean over ratios = {np.mean(row_means):+.2f} dB")

    elapsed = time.time() - t0
    print(f"\nSweep completed in {elapsed/60:.1f} minutes")

    # Save raw results
    with open(out_dir / "pocs_sweep_raw.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_dir / 'pocs_sweep_raw.json'}")

    # ===== Aggregations =====

    print("\n" + "=" * 78)
    print("AGGREGATED RESULTS — overall (avg across all 6 datasets, 7 ratios)")
    print("=" * 78)
    print(f"{'Config':<20s} {'n_iter':>6s} {'t_max':>6s} {'t_min':>6s} "
          f"{'decay':>12s} {'SNR mean':>10s}")
    overall = {}
    for cfg_name, n_iter, t_max, t_min, decay in POCS_CONFIGS:
        means = [r["snr_mean"] for r in results
                 if r["config"] == cfg_name and np.isfinite(r["snr_mean"])]
        overall[cfg_name] = float(np.mean(means)) if means else float("nan")
        print(f"{cfg_name:<20s} {n_iter:>6d} {t_max:>6.1f} {t_min:>6.1f} "
              f"{decay:>12s} {overall[cfg_name]:>+8.2f} dB")

    best_overall = max(overall.items(), key=lambda x: x[1])
    print(f"\n>>> BEST OVERALL: {best_overall[0]} ({best_overall[1]:+.2f} dB)")

    # Per streamer length
    print("\n" + "=" * 78)
    print("AGGREGATED RESULTS — per streamer length")
    print("=" * 78)
    by_length = defaultdict(lambda: defaultdict(list))
    for r in results:
        if np.isfinite(r["snr_mean"]):
            by_length[r["streamer_length"]][r["config"]].append(r["snr_mean"])

    streamer_lengths = sorted(by_length.keys())
    print(f"{'Config':<20s}", end="")
    for L in streamer_lengths:
        print(f"  {int(L)}m".rjust(10), end="")
    print()
    print("-" * 78)
    best_per_length = {}
    for cfg_name, _, _, _, _ in POCS_CONFIGS:
        print(f"{cfg_name:<20s}", end="")
        for L in streamer_lengths:
            mean_val = np.mean(by_length[L][cfg_name])
            print(f"  {mean_val:+.2f} dB".rjust(10), end="")
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
        print(f"  {int(L)}m".rjust(10), end="")
    print(f"  {'overall':>10s}")
    print("-" * 78)
    for ratio in ratios:
        print(f"{int(ratio*100):<3d}%      ", end="")
        for L in streamer_lengths:
            vals = [r["snr_mean"] for r in results
                    if r["config"] == best_overall[0]
                    and r["streamer_length"] == L
                    and r["ratio"] == ratio
                    and np.isfinite(r["snr_mean"])]
            print(f"  {np.mean(vals):+.2f} dB".rjust(10), end="")
        all_vals = [r["snr_mean"] for r in results
                    if r["config"] == best_overall[0]
                    and r["ratio"] == ratio
                    and np.isfinite(r["snr_mean"])]
        print(f"  {np.mean(all_vals):+.2f} dB".rjust(10))

    # vs baseline_old
    if "baseline_old" in overall and best_overall[0] != "baseline_old":
        delta = best_overall[1] - overall["baseline_old"]
        print(f"\n>>> Improvement vs baseline_old: {delta:+.2f} dB")

    # Save summary
    summary = {
        "overall": overall,
        "best_overall": {"config": best_overall[0], "snr": best_overall[1]},
        "best_per_length": {str(L): {"config": v[0], "snr": v[1]}
                            for L, v in best_per_length.items()},
        "configs_tested": [
            {"name": n, "n_iter": ni, "t_max": tx, "t_min": tn, "decay": d}
            for n, ni, tx, tn, d in POCS_CONFIGS
        ],
    }
    with open(out_dir / "pocs_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out_dir / 'pocs_sweep_summary.json'}")


if __name__ == "__main__":
    main()
