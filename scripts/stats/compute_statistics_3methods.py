"""
compute_statistics_3methods.py — Three-way statistics DL/POCS/MSSA on the
internal test set (Table 5; read by make_fig4_decimation.py for Fig. 4).

Same statistics as compute_statistics.py, and in addition:
  - mssa_per_gather_rank{rank}.json is loaded too
  - the paired comparisons DL vs MSSA and MSSA vs POCS are added

Input:
  RESULTS_ROOT/statistics/per_gather_results.json        DL + POCS
      (from scripts/evaluate/recompute_with_per_gather_lists.py)
  RESULTS_ROOT/statistics/mssa_per_gather_rank30.json    MSSA (default of --mssa)
      Pass with --mssa the file written by
      scripts/evaluate/recompute_mssa_per_gather.py for the rank in use
      (mssa_per_gather_rank{rank}.json).

Output:
  RESULTS_ROOT/statistics/statistics_3methods.json

Usage:
    python scripts/stats/compute_statistics_3methods.py \\
        --mssa results/statistics/mssa_per_gather_rank10.json
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
    if rng is None:
        rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), \
           float(np.percentile(boot_means, 97.5))


def _ci95_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                           rng: np.random.Generator = None) -> Tuple[float, float, float]:
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
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff_matrix = a[:, None] - b[None, :]
    n_greater = (diff_matrix > 0).sum()
    n_less    = (diff_matrix < 0).sum()
    return float((n_greater - n_less) / (len(a) * len(b)))


def _cliffs_interpretation(delta: float) -> str:
    a = abs(delta)
    if a < 0.147:  return "negligible"
    if a < 0.33:   return "small"
    if a < 0.474:  return "medium"
    return "large"


def stats_for_single_method(values: List[float], n_boot: int = 10000) -> Dict:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    arr = arr[finite]

    mean, std, ci_lo_a, ci_hi_a = _ci95_analytic(arr)
    ci_lo_b, ci_hi_b = _ci95_bootstrap(arr, n_boot=n_boot)

    return {
        "n":              int(len(arr)),
        "mean":           mean,
        "std":            std,
        "se":             float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "ci95_analytic":  [ci_lo_a, ci_hi_a],
        "ci95_bootstrap": [ci_lo_b, ci_hi_b],
        "median":         float(np.median(arr)),
        "q25":            float(np.percentile(arr, 25)),
        "q75":            float(np.percentile(arr, 75)),
    }


def paired_comparison_stats(a: List[float], b: List[float],
                              n_boot: int = 10000) -> Dict:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a_arr) & np.isfinite(b_arr)
    a_arr = a_arr[finite]
    b_arr = b_arr[finite]
    n_paired = len(a_arr)

    if n_paired >= 2:
        try:
            stat, pval = wilcoxon(a_arr, b_arr, alternative="greater",
                                    zero_method="wilcox", method="auto")
            wilcox_stat = float(stat)
            wilcox_p    = float(pval)
        except Exception:
            wilcox_stat = float("nan")
            wilcox_p    = float("nan")
    else:
        wilcox_stat = float("nan")
        wilcox_p    = float("nan")

    diff_mean, diff_lo, diff_hi = _ci95_bootstrap_diff(a_arr, b_arr,
                                                         n_boot=n_boot)
    cliff = _cliffs_delta(a_arr, b_arr)

    return {
        "n_paired":         n_paired,
        "diff_mean":        diff_mean,
        "diff_ci95_boot":   [diff_lo, diff_hi],
        "wilcoxon_stat":    wilcox_stat,
        "wilcoxon_p":       wilcox_p,
        "cliffs_delta":     cliff,
        "cliffs_magnitude": _cliffs_interpretation(cliff),
    }


def process_3way(dl_pocs_data: Dict, mssa_data: Dict,
                   n_boot: int = 10000) -> Dict:
    """
    Compute statistics for 3-way comparison: DL vs POCS, DL vs MSSA,
    MSSA vs POCS.
    """
    results = {}
    metric_names = ["snr_miss", "snr_full", "ssim", "fk"]

    for strategy in dl_pocs_data.keys():
        if strategy not in mssa_data:
            print(f"WARNING: strategy {strategy} not in MSSA data, skipping")
            continue

        results[strategy] = {}
        for ratio_key in dl_pocs_data[strategy].keys():
            if ratio_key not in mssa_data[strategy]:
                print(f"WARNING: ratio {ratio_key} missing in MSSA, skipping")
                continue

            dlpocs = dl_pocs_data[strategy][ratio_key]
            mssa   = mssa_data[strategy][ratio_key]

            results[strategy][ratio_key] = {
                "n": dlpocs["n"],
            }

            for metric in metric_names:
                dl_vals   = dlpocs[f"dl_{metric}"]
                pocs_vals = dlpocs[f"pocs_{metric}"]
                mssa_vals = mssa[f"mssa_{metric}"]

                # Single-method stats
                results[strategy][ratio_key][f"{metric}_dl"] = \
                    stats_for_single_method(dl_vals, n_boot=n_boot)
                results[strategy][ratio_key][f"{metric}_pocs"] = \
                    stats_for_single_method(pocs_vals, n_boot=n_boot)
                results[strategy][ratio_key][f"{metric}_mssa"] = \
                    stats_for_single_method(mssa_vals, n_boot=n_boot)

                # Pairwise comparisons
                results[strategy][ratio_key][f"{metric}_dl_vs_pocs"] = \
                    paired_comparison_stats(dl_vals, pocs_vals,
                                              n_boot=n_boot)
                results[strategy][ratio_key][f"{metric}_dl_vs_mssa"] = \
                    paired_comparison_stats(dl_vals, mssa_vals,
                                              n_boot=n_boot)
                results[strategy][ratio_key][f"{metric}_mssa_vs_pocs"] = \
                    paired_comparison_stats(mssa_vals, pocs_vals,
                                              n_boot=n_boot)

    return results


def print_summary(stats: Dict, strategy: str = "random"):
    if strategy not in stats:
        print(f"No data for strategy {strategy}")
        return
    print(f"\n{'='*120}")
    print(f"3-WAY COMPARISON — strategy: {strategy} — SNR_miss (dB)")
    print(f"{'='*120}")
    print(f"{'Ratio':<8} {'DL':<24} {'MSSA':<24} {'POCS':<24} "
          f"{'Δ(DL-MSSA)':<14} {'Δ(MSSA-POCS)':<14}")
    print("-" * 120)
    for ratio_key in sorted(stats[strategy].keys()):
        r = stats[strategy][ratio_key]
        dl   = r["snr_miss_dl"]
        ms   = r["snr_miss_mssa"]
        po   = r["snr_miss_pocs"]
        d_dm = r["snr_miss_dl_vs_mssa"]
        d_mp = r["snr_miss_mssa_vs_pocs"]
        dl_s = f"{dl['mean']:+.2f}[{dl['ci95_bootstrap'][0]:+.2f},{dl['ci95_bootstrap'][1]:+.2f}]"
        ms_s = f"{ms['mean']:+.2f}[{ms['ci95_bootstrap'][0]:+.2f},{ms['ci95_bootstrap'][1]:+.2f}]"
        po_s = f"{po['mean']:+.2f}[{po['ci95_bootstrap'][0]:+.2f},{po['ci95_bootstrap'][1]:+.2f}]"
        print(f"{ratio_key:<8} {dl_s:<24} {ms_s:<24} {po_s:<24} "
              f"{d_dm['diff_mean']:+.2f}  {d_mp['diff_mean']:+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-gather",
                        default=str(RESULTS_ROOT / "statistics/per_gather_results.json"))
    parser.add_argument("--mssa",
                        default=str(RESULTS_ROOT / "statistics/mssa_per_gather_rank30.json"))
    parser.add_argument("--cross-survey",
                        default=str(RESULTS_ROOT / "statistics/cross_survey_per_gather.json"))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--output",
                        default=str(RESULTS_ROOT / "statistics/statistics_3methods.json"))
    args = parser.parse_args()

    output = {}

    dl_pocs_path = Path(args.per_gather)
    mssa_path = Path(args.mssa)
    if dl_pocs_path.exists() and mssa_path.exists():
        dl_pocs = json.loads(dl_pocs_path.read_text())
        mssa    = json.loads(mssa_path.read_text())
        print(f"Processing 3-way comparison ({dl_pocs_path}, {mssa_path})")
        output["3way"] = process_3way(dl_pocs, mssa, n_boot=args.n_boot)
        print_summary(output["3way"], strategy="random")
    else:
        if not dl_pocs_path.exists():
            print(f"WARNING: {dl_pocs_path} missing")
        if not mssa_path.exists():
            print(f"WARNING: {mssa_path} missing")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
