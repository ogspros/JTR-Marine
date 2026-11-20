"""
jtr_marine/metrics.py — Evaluation metrics for seismic reconstruction.

Metrics:
  - SNR:              signal-to-noise ratio on missing traces
  - SSIM:             structural similarity
  - MSE:              mean squared error on missing traces
  - R_fk:             ratio of f-k spectral amplitudes (prediction / target)
                      inside an apparent-velocity band (see
                      compute_fk_energy_ratio)
"""

import torch
import numpy as np
from typing import Dict, List
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr(pred: torch.Tensor,
                target: torch.Tensor,
                trace_mask: torch.Tensor,
                validity: torch.Tensor,
                missing_only: bool = True) -> torch.Tensor:
    """
    Signal-to-Noise Ratio on reconstructed traces.

    SNR = 10 * log10( ||target||² / ||target - pred||² )

    If missing_only=True, compute only over traces that were removed.
    Higher is better. Typical acceptable range: >10 dB.

    Args:
        pred, target: (B, 1, T, O)
        trace_mask:   (B, O) — 1=present, 0=removed
        validity:     (B, O) — 1=real, 0=padding
    """
    if missing_only:
        mask = ((1 - trace_mask) * validity)          # (B, O)
        mask_2d = mask.unsqueeze(1).unsqueeze(2).expand_as(pred)
    else:
        mask_2d = validity.unsqueeze(1).unsqueeze(2).expand_as(pred)

    signal_power = (target ** 2 * mask_2d).sum()
    noise_power  = ((target - pred) ** 2 * mask_2d).sum()

    snr = 10.0 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr


def compute_mse(pred: torch.Tensor,
                target: torch.Tensor,
                trace_mask: torch.Tensor,
                validity: torch.Tensor,
                missing_only: bool = True) -> torch.Tensor:
    """MSE on missing traces."""
    if missing_only:
        mask = ((1 - trace_mask) * validity)
    else:
        mask = validity
    mask_2d = mask.unsqueeze(1).unsqueeze(2).expand_as(pred)
    n = mask_2d.sum().clamp(min=1.0)
    return ((pred - target) ** 2 * mask_2d).sum() / n


def compute_ssim_metric(pred: torch.Tensor,
                        target: torch.Tensor,
                        window_size: int = 11) -> torch.Tensor:
    """
    SSIM as a quality metric (not as loss). Returns mean SSIM in [0, 1].
    """
    from jtr_marine.losses.composite import SSIM2D
    ssim_fn = SSIM2D(window_size=window_size).to(pred.device)
    ssim_loss = ssim_fn(pred, target)
    return 1.0 - ssim_loss   # SSIM metric (higher = better)


def compute_fk_energy_ratio(pred: torch.Tensor,
                             target: torch.Tensor,
                             dt: float = 0.002,
                             dx: float = 12.5,
                             v_min: float = 1400.0,
                             v_max: float = 5000.0) -> torch.Tensor:
    """
    R_fk: ratio of f-k spectral *amplitudes* (not energies) of prediction
    and target inside an apparent-velocity band.

    Both inputs are transformed with an orthonormal 2-D FFT (time x offset).
    The band keeps the f-k bins whose apparent velocity |f| / |k| lies in
    [v_min, v_max] (default 1400-5000 m/s); the k = 0 column is excluded.
    The returned value is

        sum_band |FFT(pred)| / sum_band |FFT(target)|

    i.e. the ratio of the L1 norms of the FFT moduli inside the band,
    summed over the whole batch (one scalar per batch, not per gather).
    Values near 1.0 mean the reconstruction carries the same in-band
    spectral amplitude as the target. Despite the function name, no
    squared magnitudes are used and the energy outside the band is not
    involved; the name is kept for compatibility with the paper's code.

    Args:
        pred, target: (B, 1, T, O)
        dt:           sampling interval in seconds
        dx:           receiver spacing in meters
        v_min, v_max: apparent-velocity bounds of the band (m/s)
    """
    B, _, T, O = pred.shape

    # Full fft2 over both axes (time and offset). rfft2 would halve only the
    # last axis and make the f-k grid asymmetric; fft2 + fftfreq is correct
    # for the apparent-velocity band.
    pred_fft   = torch.fft.fft2(pred,   norm="ortho")
    target_fft = torch.fft.fft2(target, norm="ortho")

    pred_mag   = torch.abs(pred_fft)    # (B, 1, T, O)
    target_mag = torch.abs(target_fft)  # (B, 1, T, O)

    # Frequency / wavenumber grids: fftfreq returns T and O values
    freqs = torch.fft.fftfreq(T, d=dt).to(pred.device)   # (T,) in Hz
    kwave = torch.fft.fftfreq(O, d=dx).to(pred.device)   # (O,) in 1/m

    f_grid = freqs.unsqueeze(1).abs()   # (T, 1)
    k_grid = kwave.unsqueeze(0).abs()   # (1, O)

    # Apparent-velocity band: v_min <= |f|/|k| <= v_max
    k_safe     = k_grid.clamp(min=1e-6)
    apparent_v = f_grid / k_safe        # (T, O)
    cone_mask  = (apparent_v >= v_min) & (apparent_v <= v_max)
    cone_mask[:, 0] = False             # k = 0 column excluded

    # Broadcast to (B, 1, T, O)
    cone_mask = cone_mask.unsqueeze(0).unsqueeze(0).expand_as(pred_mag)

    pred_in_cone   = (pred_mag   * cone_mask).sum()
    target_in_cone = (target_mag * cone_mask).sum()

    return pred_in_cone / (target_in_cone + 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_batch(pred: torch.Tensor,
                   target: torch.Tensor,
                   trace_mask: torch.Tensor,
                   validity: torch.Tensor,
                   dt: float = 0.002,
                   dx: float = 12.5) -> Dict[str, float]:
    """Compute all metrics for one batch. Returns dict of floats."""
    snr  = compute_snr(pred, target, trace_mask, validity).item()
    mse  = compute_mse(pred, target, trace_mask, validity).item()
    ssim = compute_ssim_metric(pred, target).item()
    fk   = compute_fk_energy_ratio(pred, target, dt=dt, dx=dx).item()

    return {
        "snr_db":        snr,
        "mse":           mse,
        "ssim":          ssim,
        "fk_energy_ratio": fk,
    }


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average a list of per-batch metric dicts.
    Returns mean ± std for each metric.
    """
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    result = {}
    for k in keys:
        values = [m[k] for m in metric_list]
        result[f"{k}_mean"] = float(np.mean(values))
        result[f"{k}_std"]  = float(np.std(values))
    return result
