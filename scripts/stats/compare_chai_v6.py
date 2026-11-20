#!/usr/bin/env python3
"""
compare_chai_v6.py — Paired comparison of two recompute runs on identical masks
(Table S7: Chai et al. 2020 variants against the compact U-Net).

    python scripts/stats/compare_chai_v6.py \
        results/tableS7/chai_mse/per_gather_results.json \
        results/table9/compact_seed42/per_gather_results.json

1. Mask check: POCS values must coincide in the two files (same PYTHONHASHSEED
   -> same masks). If they do not, the comparison is paired by gather only.
2. For each strategy x ratio and metric: mean A, mean B, delta A-B, 95% CI of
   the delta by moving-block bootstrap along the line (blocks of consecutive
   gathers, never crossing a survey boundary), fraction of gathers where A > B.
The test set is 6 surveys x 46 consecutive gathers (gather_ids 0..275).
"""

import json
import sys
import numpy as np

N_PER_SURVEY = 46
BLOCK = 10
N_BOOT = 10000
METRICS = [("snr_miss", "SNR_miss"), ("snr_full", "SNR_full"),
           ("ssim", "SSIM"), ("fk", "R_fk")]


def block_bootstrap_ci(diff, rng, block=BLOCK, n_boot=N_BOOT):
    """Moving-block bootstrap of the mean, resampling within each survey."""
    diff = np.asarray(diff, float)
    surveys = [diff[i:i + N_PER_SURVEY] for i in range(0, len(diff), N_PER_SURVEY)]
    means = np.empty(n_boot)
    for b in range(n_boot):
        parts = []
        for s in surveys:
            n = len(s)
            starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
            parts.append(np.concatenate([s[st:st + block] for st in starts])[:n])
        means[b] = np.concatenate(parts).mean()
    return np.percentile(means, [2.5, 97.5])


def main(path_a, path_b, name_a="Chai", name_b="v6"):
    A = json.load(open(path_a))
    B = json.load(open(path_b))
    rng = np.random.default_rng(0)

    # mask check
    max_dev = 0.0
    for s in A:
        for r in A[s]:
            if A[s][r]["gather_ids"] != B[s][r]["gather_ids"]:
                sys.exit(f"gather_ids differ at {s}/{r}")
            pa, pb = A[s][r].get("pocs_snr_miss"), B[s][r].get("pocs_snr_miss")
            if pa is None or pb is None:
                max_dev = None
                continue
            if max_dev is not None:
                max_dev = max(max_dev, float(np.max(np.abs(np.array(pa) - np.array(pb)))))
    if max_dev is None:
        print("Mask check: skipped (POCS not stored in one file; rely on PYTHONHASHSEED)\n")
    else:
        print(f"Mask check: max |POCS_A - POCS_B| = {max_dev:.2e} dB "
              f"-> {'IDENTICAL masks' if max_dev < 1e-6 else 'DIFFERENT masks (paired by gather only)'}\n")

    summary = {}
    for key, label in METRICS:
        print(f"=== {label}:  {name_a} - {name_b}  (block bootstrap, block={BLOCK}) ===")
        print(f"{'strategy':8s} {'ratio':>6s} {name_a:>8s} {name_b:>8s} {'delta':>7s} "
              f"{'95% CI':>18s} {'wins_A':>7s}")
        for s in A:
            for r in A[s]:
                a = np.array(A[s][r][f"dl_{key}"], float)
                b = np.array(B[s][r][f"dl_{key}"], float)
                d = a - b
                lo, hi = block_bootstrap_ci(d, rng)
                fmt = "{:8.3f}" if key in ("ssim", "fk") else "{:8.2f}"
                print(f"{s:8s} {r:>6s} " + fmt.format(a.mean()) + " " + fmt.format(b.mean())
                      + f" {d.mean():+7.3f} [{lo:+7.3f},{hi:+7.3f}] {np.mean(d > 0):7.2f}")
                summary.setdefault(s, {}).setdefault(r, {})[key] = {
                    "mean_a": a.mean(), "mean_b": b.mean(), "delta": d.mean(),
                    "ci95": [lo, hi], "frac_a_better": float(np.mean(d > 0))}
        print()

    with open("compare_chai_v6_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved: compare_chai_v6_summary.json")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
