"""
compute_statistics.py — Load the per-gather values and compute:
  - analytic 95% CI of every metric (DL and POCS)
  - bootstrap 95% CI (n_boot=10000), for robustness
  - median, q25, q75 (robust statistics)
  - paired Wilcoxon signed-rank test (DL vs POCS): p-value
  - bootstrap 95% CI of the difference Δ = DL − POCS
  - Cliff's delta (non-parametric effect size)

Input:
  RESULTS_ROOT/statistics/per_gather_results.json
      (from scripts/evaluate/recompute_with_per_gather_lists.py)
  RESULTS_ROOT/statistics/cross_survey_per_gather.json  (if present; from
      scripts/evaluate/recompute_cross_survey_per_gather.py)

Output:
  RESULTS_ROOT/statistics/statistics_full.json

Usage:
    python scripts/stats/compute_statistics.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import RESULTS_ROOT

import json
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, t as student_t
from typing import Dict, List, Tuple


def _ci95_analytic(values: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Mean and 95% CI using Student's t (small-sample correction).
    Returns: (mean, std, ci_low, ci_high)
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    m = float(values.mean())
    s = float(values.std(ddof=1)) if n > 1 else 0.0
    se = s / np.sqrt(n) if n > 1 else 0.0
    if n > 1:
        t_crit = student_t.ppf(0.975, df=n-1)
        ci_low  = m - t_crit * se
        ci_high = m + t_crit * se
    else:
        ci_low = ci_high = m
    return m, s, ci_low, ci_high


def _ci95_bootstrap(values: np.ndarray, n_boot: int = 10000,
                     rng: np.random.Generator = None) -> Tuple[float, float]:
    """
    95% bootstrap CI on the mean (percentile method).
    Returns: (ci_low, ci_high)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        return float(values[0]), float(values[0])
    # Bootstrap means
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), \
           float(np.percentile(boot_means, 97.5))


def _ci95_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                           rng: np.random.Generator = None) -> Tuple[float, float, float]:
    """
    95% bootstrap CI on the PAIRED difference (a - b).
    Returns: (mean_diff, ci_low, ci_high)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    return float(diff.mean()), \
           float(np.percentile(boot_means, 2.5)), \
           float(np.percentile(boot_means, 97.5))


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta: non-parametric effect size in [-1, 1].
       δ = (#{a > b} - #{a < b}) / (n_a * n_b)
    Magnitude conventions (Romano et al.):
       |δ| < 0.147  negligible
       |δ| < 0.33   small
       |δ| < 0.474  medium
       |δ| >= 0.474 large
    Returns: cliff's delta scalar
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # Pairwise: each ai vs all bj
    diff_matrix = a[:, None] - b[None, :]
    n_greater = (diff_matrix > 0).sum()
    n_less    = (diff_matrix < 0).sum()
    return float((n_greater - n_less) / (len(a) * len(b)))


def _cliffs_interpretation(delta: float) -> str:
    """Map magnitude to qualitative label."""
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    elif a < 0.33:
        return "small"
    elif a < 0.474:
        return "medium"
    else:
        return "large"


def stats_for_single_method(values: List[float],
                              n_boot: int = 10000) -> Dict:
    """
    Compute descriptive + CI statistics for a single list of metric values.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    arr = arr[finite]
    n_invalid = int((~finite).sum())

    mean, std, ci_lo_a, ci_hi_a = _ci95_analytic(arr)
    ci_lo_b, ci_hi_b = _ci95_bootstrap(arr, n_boot=n_boot)

    return {
        "n":             int(len(arr)),
        "n_invalid":     n_invalid,
        "mean":          mean,
        "std":           std,
        "se":            float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "ci95_analytic": [ci_lo_a, ci_hi_a],
        "ci95_bootstrap": [ci_lo_b, ci_hi_b],
        "median":        float(np.median(arr)),
        "q25":           float(np.percentile(arr, 25)),
        "q75":           float(np.percentile(arr, 75)),
        "min":           float(arr.min()),
        "max":           float(arr.max()),
    }


def paired_comparison_stats(a: List[float], b: List[float],
                              n_boot: int = 10000) -> Dict:
    """
    Paired comparison of two methods on the same gathers (a vs b).
    Computes Δ = a − b, Wilcoxon signed-rank, bootstrap CI on Δ, Cliff's δ.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a_arr) & np.isfinite(b_arr)
    a_arr = a_arr[finite]
    b_arr = b_arr[finite]
    n_paired = len(a_arr)

    # Wilcoxon paired (alternative="greater": H1: a > b)
    if n_paired >= 2:
        try:
            stat, pval = wilcoxon(a_arr, b_arr, alternative="greater",
                                    zero_method="wilcox", method="auto")
            wilcox_stat = float(stat)
            wilcox_p    = float(pval)
        except Exception as e:
            wilcox_stat = float("nan")
            wilcox_p    = float("nan")
    else:
        wilcox_stat = float("nan")
        wilcox_p    = float("nan")

    # Bootstrap CI on the paired difference
    diff_mean, diff_lo, diff_hi = _ci95_bootstrap_diff(
        a_arr, b_arr, n_boot=n_boot,
    )

    # Cliff's delta as unpaired effect size (descriptive)
    cliff = _cliffs_delta(a_arr, b_arr)

    return {
        "n_paired":          n_paired,
        "diff_mean":         diff_mean,
        "diff_ci95_boot":    [diff_lo, diff_hi],
        "wilcoxon_stat":     wilcox_stat,
        "wilcoxon_p":        wilcox_p,
        "wilcoxon_test":     "paired signed-rank, one-sided (H1: a > b)",
        "cliffs_delta":      cliff,
        "cliffs_magnitude":  _cliffs_interpretation(cliff),
    }


def process_dl_vs_pocs(per_gather_json: Path, n_boot: int = 10000) -> Dict:
    """
    Build statistics for DL vs POCS comparison (within-test set).
    """
    data = json.loads(per_gather_json.read_text())
    results = {}
    metrics = [
        ("snr_miss", "dl_snr_miss",   "pocs_snr_miss"),
        ("snr_full", "dl_snr_full",   "pocs_snr_full"),
        ("ssim",     "dl_ssim",       "pocs_ssim"),
        ("fk",       "dl_fk",         "pocs_fk"),
    ]

    for strategy, ratios_dict in data.items():
        results[strategy] = {}
        for ratio_key, gather_data in ratios_dict.items():
            results[strategy][ratio_key] = {"n": gather_data["n"]}

            for metric_name, dl_key, pocs_key in metrics:
                dl_vals   = gather_data[dl_key]
                pocs_vals = gather_data[pocs_key]

                results[strategy][ratio_key][f"{metric_name}_dl"]   = \
                    stats_for_single_method(dl_vals, n_boot=n_boot)
                results[strategy][ratio_key][f"{metric_name}_pocs"] = \
                    stats_for_single_method(pocs_vals, n_boot=n_boot)
                results[strategy][ratio_key][f"{metric_name}_paired"] = \
                    paired_comparison_stats(dl_vals, pocs_vals, n_boot=n_boot)

    return results


def process_cross_survey(cross_survey_json: Path, n_boot: int = 10000) -> Dict:
    """
    Statistics for cross-survey (CG03, IT245, CROP_M18) — single method (DL).
    """
    data = json.loads(cross_survey_json.read_text())
    results = {}
    metrics = ["snr_miss", "snr_full", "ssim", "fk"]

    for ds_name, ds_data in data.items():
        results[ds_name] = {}
        for ratio_key, ratio_data in ds_data.items():
            results[ds_name][ratio_key] = {"n": ratio_data["n"]}
            for metric in metrics:
                if metric in ratio_data:
                    results[ds_name][ratio_key][metric] = \
                        stats_for_single_method(ratio_data[metric],
                                                  n_boot=n_boot)
    return results


def print_summary_dl_vs_pocs(stats: Dict, strategy: str = "random"):
    """Pretty-print summary table for one strategy."""
    print(f"\n{'='*100}")
    print(f"SUMMARY DL vs POCS — strategy: {strategy}")
    print(f"{'='*100}")
    print(f"{'Ratio':<8} {'DL_miss [CI95]':<24} {'POCS_miss [CI95]':<24} "
          f"{'Δ [CI95]':<22} {'p_W':<10} {'Cliff δ':<14}")
    print("-" * 100)
    if strategy not in stats:
        print(f"  No data for strategy {strategy}")
        return
    for ratio_key in sorted(stats[strategy].keys()):
        r = stats[strategy][ratio_key]
        dl  = r["snr_miss_dl"]
        po  = r["snr_miss_pocs"]
        pr  = r["snr_miss_paired"]
        dl_str   = f"{dl['mean']:+.2f} [{dl['ci95_bootstrap'][0]:+.2f},{dl['ci95_bootstrap'][1]:+.2f}]"
        po_str   = f"{po['mean']:+.2f} [{po['ci95_bootstrap'][0]:+.2f},{po['ci95_bootstrap'][1]:+.2f}]"
        diff_str = f"{pr['diff_mean']:+.2f} [{pr['diff_ci95_boot'][0]:+.2f},{pr['diff_ci95_boot'][1]:+.2f}]"
        p_str = f"{pr['wilcoxon_p']:.2e}" if pr['wilcoxon_p'] >= 1e-300 else "<1e-300"
        cliff_str = f"{pr['cliffs_delta']:+.3f} ({pr['cliffs_magnitude']})"
        print(f"{ratio_key:<8} {dl_str:<24} {po_str:<24} {diff_str:<22} "
              f"{p_str:<10} {cliff_str:<14}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-gather",
                        default=str(RESULTS_ROOT / "statistics/per_gather_results.json"),
                        help="JSON with the per-gather DL+POCS values (output of recompute_with_per_gather_lists.py)")
    parser.add_argument("--cross-survey",
                        default=str(RESULTS_ROOT / "statistics/cross_survey_per_gather.json"),
                        help="(optional) JSON with the per-gather cross-survey values")
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="Number of bootstrap resamples (default 10000)")
    parser.add_argument("--output",
                        default=str(RESULTS_ROOT / "statistics/statistics_full.json"))
    args = parser.parse_args()

    output = {}
    per_gather_path = Path(args.per_gather)
    if per_gather_path.exists():
        print(f"Processing DL vs POCS: {per_gather_path}")
        output["dl_vs_pocs"] = process_dl_vs_pocs(per_gather_path,
                                                    n_boot=args.n_boot)
        print_summary_dl_vs_pocs(output["dl_vs_pocs"], strategy="random")
    else:
        print(f"WARNING: {per_gather_path} not found, skipping DL vs POCS")

    cross_survey_path = Path(args.cross_survey)
    if cross_survey_path.exists():
        print(f"\nProcessing cross-survey: {cross_survey_path}")
        output["cross_survey"] = process_cross_survey(cross_survey_path,
                                                       n_boot=args.n_boot)
    else:
        print(f"INFO: {cross_survey_path} not found (optional), skipping cross-survey")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved statistics to {out_path}")


if __name__ == "__main__":
    main()
