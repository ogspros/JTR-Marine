"""
jtr_marine/baselines/pocs.py — POCS (Projection Onto Convex Sets) baseline.

Standard classical method for the interpolation of missing seismic traces.
Reference: Abma & Kabir (2006), 3D interpolation of irregular data with
           a POCS algorithm. Geophysics, 71(6), E91-E97.

Algorithm:
  1. Start from: x0 = gather with missing traces = 0
  2. Iterate N times:
     a. Transform to the f-k domain: X = FFT2D(x)
     b. Threshold: zero the coefficients with |X| < threshold(iter)
     c. Inverse transform: x' = IFFT2D(X)
     d. Re-insert the known traces: x = mask*x_obs + (1-mask)*x'
  3. The threshold (a percentile of |X|) decays from threshold_max to
     threshold_min, with a linear or exponential schedule

Physical rationale:
  Coherent seismic events produce concentrated peaks in f-k.
  Missing traces smear the energy across the spectrum.
  f-k thresholding preserves the true peaks and attenuates the smearing,
  while the "known traces" constraint keeps the observed data intact.

Advantages:
  - No training, deterministic
  - Standard benchmark method in seismic interpolation
  - Works well for regular and moderate missing patterns (< 50%)

Limitations:
  - Performance degrades for high missing ratios (> 60%)
  - Does not learn from similar data
  - Single-shot processing (not conditioned on the geometry)
"""

import numpy as np
from typing import Optional


def pocs_interpolate(gather: np.ndarray,
                     trace_mask: np.ndarray,
                     n_iter: int = 100,
                     threshold_max_pct: float = 99.0,
                     threshold_min_pct: float = 1.0,
                     decay: str = "exponential",
                     verbose_iter: bool = False) -> np.ndarray:
    """
    POCS interpolation of a single seismic gather.
    Abma & Kabir (2006) scheme with percentile-based f-k thresholding.

    Args:
        gather:     (n_time, n_offsets) gather with missing traces = 0
        trace_mask: (n_offsets,) 1=present, 0=missing
        n_iter:     number of iterations (default 100)
        threshold_max_pct: initial percentile of |X| used as threshold
                           (default 99: only the top 1% of the f-k
                           coefficients is kept at the first iteration)
        threshold_min_pct: final percentile (default 1)
        decay:      "linear" or "exponential" (default). "linear": the
                    percentile decreases linearly; "exponential": it decays
                    geometrically from threshold_max_pct to
                    max(threshold_min_pct, 0.1).
        verbose_iter: print the RMS of the missing traces every 20 iterations

    Returns:
        gather_recon: (n_time, n_offsets) reconstructed gather
    """
    gather = gather.astype(np.float64).copy()  # float64 for numerical stability

    # 2D mask aligned with the gather
    mask_2d = np.tile(trace_mask[np.newaxis, :].astype(np.float64),
                      (gather.shape[0], 1))

    # Observed data: known traces kept, missing = 0
    x_obs = gather * mask_2d
    x     = x_obs.copy()

    # Threshold schedule
    if decay == "linear":
        thresholds = np.linspace(threshold_max_pct, threshold_min_pct, n_iter)
    elif decay == "exponential":
        # Exponential (geometric) decay of the percentile
        thresholds = threshold_max_pct * np.exp(
            np.linspace(0, np.log(max(threshold_min_pct, 0.1) / threshold_max_pct),
                        n_iter)
        )
    else:
        raise ValueError(f"decay must be 'linear' or 'exponential'")

    for i, thr_pct in enumerate(thresholds):
        # 1. f-k transform
        X     = np.fft.fft2(x)
        X_abs = np.abs(X)

        # 2. Threshold: keep the coefficients above the thr_pct percentile
        if X_abs.max() < 1e-15:
            # (Almost) null data: nothing to do
            break
        thr_value = np.percentile(X_abs, thr_pct)
        X_thresh  = np.where(X_abs >= thr_value, X, 0.0 + 0.0j)

        # 3. Inverse transform
        x_filtered = np.real(np.fft.ifft2(X_thresh))

        # 4. Data-consistency constraint: keep the observed traces, update
        #    only the missing ones from the reconstruction
        x = mask_2d * x_obs + (1.0 - mask_2d) * x_filtered

        if verbose_iter and (i % 20 == 0 or i == n_iter - 1):
            # Diagnostic: RMS amplitude of the missing traces only
            missing_pos = (1.0 - mask_2d).astype(bool)
            if missing_pos.any():
                err = np.sqrt(np.mean(x[missing_pos]**2))
                print(f"    iter {i:3d} | thr={thr_pct:5.1f}% | "
                      f"RMS missing={err:.4f}")

    return x.astype(np.float32)


def pocs_batch(gathers: np.ndarray,
               trace_masks: np.ndarray,
               n_iter: int = 50,
               threshold_max_pct: float = 99.0,
               threshold_min_pct: float = 1.0,
               verbose: bool = False) -> np.ndarray:
    """
    Apply POCS to a batch of gathers.

    Args:
        gathers:     (B, n_time, n_offsets) batch of masked gathers
        trace_masks: (B, n_offsets) binary mask of each gather
        n_iter:      POCS iterations per gather

    Returns:
        recon: (B, n_time, n_offsets) batch of reconstructed gathers
    """
    B = gathers.shape[0]
    recon = np.zeros_like(gathers)

    for b in range(B):
        recon[b] = pocs_interpolate(
            gathers[b], trace_masks[b],
            n_iter=n_iter,
            threshold_max_pct=threshold_max_pct,
            threshold_min_pct=threshold_min_pct,
        )
        if verbose and (b + 1) % 10 == 0:
            print(f"    POCS: {b+1}/{B} gathers processed")

    return recon





def pocs_interpolate_validity_aware(gather: np.ndarray,
                                     trace_mask: np.ndarray,
                                     validity: np.ndarray,
                                     n_iter: int = 100,
                                     threshold_max_pct: float = 99.0,
                                     threshold_min_pct: float = 1.0,
                                     decay: str = "exponential") -> np.ndarray:
    """
    POCS restricted to the part of the grid actually covered by the
    streamer (valid range). Padding traces (zeros from a shorter streamer)
    are left out: POCS runs only where there is physical data.

    This makes the comparison with the DL model fairer: the padded region
    is excluded from the f-k transform instead of being treated as data,
    which would unduly penalize plain POCS.

    Args:
        gather:     (n_time, n_offsets) gather with missing traces = 0
        trace_mask: (n_offsets,) 1=trace present, 0=trace removed
        validity:   (n_offsets,) 1=physically recorded, 0=padding
        n_iter:     POCS iterations
        threshold_max_pct, threshold_min_pct, decay: as in pocs_interpolate

    Returns:
        gather_recon: (n_time, n_offsets) reconstructed gather (padding stays 0)
    """
    # Find the valid range (first and last offset with real data)
    valid_indices = np.where(validity > 0)[0]
    if len(valid_indices) < 2:
        return gather.astype(np.float32).copy()

    i_start = valid_indices[0]
    i_end   = valid_indices[-1] + 1   # exclusive

    # Extract the sub-gather and the corresponding mask
    sub_gather = gather[:, i_start:i_end].copy()
    sub_mask   = trace_mask[i_start:i_end]

    # Apply POCS to the valid range
    sub_recon = pocs_interpolate(
        sub_gather, sub_mask,
        n_iter=n_iter,
        threshold_max_pct=threshold_max_pct,
        threshold_min_pct=threshold_min_pct,
        decay=decay,
    )

    # Full-size output: padding stays zero, the valid range is reconstructed
    recon = np.zeros_like(gather, dtype=np.float32)
    recon[:, i_start:i_end] = sub_recon
    return recon


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: POCS vs ground truth on the within-geology test set
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pocs_baseline(config=None,
                            n_iter: int = 50,
                            n_test_samples: Optional[int] = None,
                            output_dir: str = "results/pocs_baseline"):
    """
    Compute the POCS metrics on the within-geology test set.
    Uses the same preprocessing as the DL model for a fair comparison.

    Args:
        config:         configuration (default: get_config())
        n_iter:         POCS iterations
        n_test_samples: limit to N gathers for speed (None = all)
        output_dir:     folder where the results are saved
    """
    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader
    import json
    import sys
    sys.path.insert(0, ".")

    from config import get_config
    from jtr_marine.data.dataset import build_within_geology_datasets
    from jtr_marine.metrics import compute_snr, compute_mse, compute_ssim_metric, \
                              compute_fk_energy_ratio

    if config is None:
        config = get_config()

    print("="*70)
    print("POCS BASELINE EVALUATION")
    print("="*70)

    # Same test set as the DL model
    _, _, test_dataset = build_within_geology_datasets(config)
    print(f"\nTotal test gathers: {len(test_dataset)}")

    if n_test_samples is not None:
        n_test = min(n_test_samples, len(test_dataset))
        print(f"Limited to {n_test} gathers for a quick evaluation")
    else:
        n_test = len(test_dataset)

    all_metrics = {"snr": [], "mse": [], "ssim": [], "fk": []}

    for i in range(n_test):
        sample = test_dataset[i]

        # Tensors already have the right shape
        masked_gather = sample["input"][0].numpy()         # (T, O) — ch0 is the masked gather
        trace_mask    = sample["trace_mask"].numpy()       # (O,)
        target        = sample["target"][0].numpy()        # (T, O)
        validity      = sample["validity"].numpy()         # (O,)

        # POCS is run only on the VALID offset range (short-streamer padding
        # is excluded); the padding traces stay zero in the output
        recon = pocs_interpolate_validity_aware(
            masked_gather, trace_mask, validity,
            n_iter=n_iter,
        )

        # Compute the metrics with torch (same functions as for the DL model)
        pred_t   = torch.from_numpy(recon).unsqueeze(0).unsqueeze(0)        # (1,1,T,O)
        target_t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
        mask_t   = torch.from_numpy(trace_mask).unsqueeze(0)                # (1,O)
        valid_t  = torch.from_numpy(validity).unsqueeze(0)

        snr  = compute_snr(pred_t, target_t, mask_t, valid_t).item()
        mse  = compute_mse(pred_t, target_t, mask_t, valid_t).item()
        ssim = compute_ssim_metric(pred_t, target_t).item()
        fk   = compute_fk_energy_ratio(
            pred_t, target_t,
            dt=config.data.dt_target_ms * 1e-3,
            dx=config.data.dx_target_m,
        ).item()

        all_metrics["snr"].append(snr)
        all_metrics["mse"].append(mse)
        all_metrics["ssim"].append(ssim)
        all_metrics["fk"].append(fk)

        if (i + 1) % 20 == 0:
            mean_snr = np.mean(all_metrics["snr"])
            print(f"  {i+1}/{n_test} gathers | mean SNR so far: {mean_snr:.2f} dB")

    # Aggregation
    summary = {
        "n_samples":       n_test,
        "n_iter":          n_iter,
        "snr_mean":        float(np.mean(all_metrics["snr"])),
        "snr_std":         float(np.std(all_metrics["snr"])),
        "snr_median":      float(np.median(all_metrics["snr"])),
        "mse_mean":        float(np.mean(all_metrics["mse"])),
        "ssim_mean":       float(np.mean(all_metrics["ssim"])),
        "ssim_std":        float(np.std(all_metrics["ssim"])),
        "fk_energy_mean":  float(np.mean(all_metrics["fk"])),
    }

    print(f"\n{'='*70}")
    print(f"POCS BASELINE RESULTS (n_iter={n_iter}, {n_test} gather)")
    print(f"{'='*70}")
    print(f"  SNR mean      = {summary['snr_mean']:.2f} ± {summary['snr_std']:.2f} dB")
    print(f"  SNR median    = {summary['snr_median']:.2f} dB")
    print(f"  SSIM mean     = {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
    print(f"  FK energy     = {summary['fk_energy_mean']:.4f}")
    print(f"  MSE mean      = {summary['mse_mean']:.6f}")

    # Save the results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "pocs_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path / 'pocs_baseline_summary.json'}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="POCS baseline for seismic interpolation")
    parser.add_argument("--n-iter", type=int, default=50,
                        help="POCS iterations (default 50)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit to N gathers (default: all)")
    args = parser.parse_args()

    evaluate_pocs_baseline(n_iter=args.n_iter, n_test_samples=args.n_samples)
