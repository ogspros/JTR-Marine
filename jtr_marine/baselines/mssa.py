"""
jtr_marine/baselines/mssa.py — Multichannel Singular Spectrum Analysis (MSSA)
for seismic trace interpolation, validity-aware.

Reference:
  Oropeza, V., and Sacchi, M. D. (2011). "Simultaneous seismic data
  denoising and reconstruction via multichannel singular spectrum
  analysis." Geophysics, 76(3), V25-V32.

Algorithm:
  For each frequency bin f:
    1. Build a Hankel matrix H(f) from the spatial trace samples at f
    2. Truncate H(f) to rank k via SVD
    3. Anti-diagonal averaging to recover smoothed trace samples
  Iterate (5-15 iterations) with trace replacement (preserve known traces).

This implementation:
  - Operates in the frequency domain (FFT along time axis)
  - Supports GPU acceleration via PyTorch (svd is the bottleneck)
  - Validity-aware: only the physically recorded traces participate
  - Same interface as jtr_marine/baselines/pocs.py for drop-in compatibility

Usage:
    from jtr_marine.baselines.mssa import mssa_interpolate_validity_aware
    recon = mssa_interpolate_validity_aware(
        gather, trace_mask, validity,
        rank=10, n_iter=10,
    )
"""

import numpy as np
import torch
from typing import Optional


def _build_hankel(x: torch.Tensor, L: int) -> torch.Tensor:
    """
    Build the Hankel matrix of a 1-D complex vector x of length N.
    
    Hankel matrix has shape (L, N-L+1), where each row is a shifted
    copy of x. For L = floor(N/2)+1, this is the standard Hankel
    matrix used in MSSA.
    
    Args:
        x: (N,) complex tensor
        L: number of rows of the Hankel matrix
    
    Returns:
        H: (L, N-L+1) complex matrix
    """
    N = x.shape[0]
    K = N - L + 1
    # Build via indexing: H[i, j] = x[i + j]
    idx_i = torch.arange(L, device=x.device).unsqueeze(1)   # (L, 1)
    idx_j = torch.arange(K, device=x.device).unsqueeze(0)   # (1, K)
    return x[idx_i + idx_j]                                  # (L, K)


def _anti_diagonal_average(H: torch.Tensor) -> torch.Tensor:
    """
    Recover the 1-D signal of length N = L + K - 1 from a Hankel matrix
    of shape (L, K) by anti-diagonal averaging.
    
    H[i, j] should be close to x[i + j], so we average over each
    anti-diagonal i + j = s, s = 0, ..., N-1.
    """
    L, K = H.shape
    N = L + K - 1
    device = H.device
    
    # Vectorized anti-diagonal averaging
    x_out  = torch.zeros(N, dtype=H.dtype, device=device)
    counts = torch.zeros(N, dtype=H.real.dtype, device=device)
    
    for i in range(L):
        # For row i: x[i + j] += H[i, j], for j = 0..K-1
        x_out[i:i+K]  += H[i, :]
        counts[i:i+K] += 1
    
    return x_out / counts


def _mssa_one_frequency(x: torch.Tensor, mask_known: torch.Tensor,
                          rank: int, n_iter: int,
                          damping: float = 0.5) -> torch.Tensor:
    """
    MSSA inner loop for a single frequency.
    
    Args:
        x:           (N,) complex tensor (data at one frequency bin)
                     missing positions should be zero on input
        mask_known:  (N,) bool tensor, True = known (preserve), False = missing
        rank:        truncation rank for SVD
        n_iter:      number of iterations
        damping:     factor for iterative update (0 = full replace, 1 = no update).
                     Typical 0.3-0.7. Without damping (alpha=0) MSSA diverges:
                     repeated rank truncation on the same low-rank estimate
                     amplifies artefacts. With damping, the missing-trace
                     update is averaged with the previous estimate, providing
                     stability.
    
    Returns:
        x_recon: (N,) complex tensor with missing positions filled in
    """
    N = x.shape[0]
    L = N // 2 + 1
    x_known = x.clone()                  # save original known values
    
    for it in range(n_iter):
        x_previous = x.clone()           # remember pre-iteration state

        # Build Hankel, truncate via SVD, recover
        H = _build_hankel(x, L)          # (L, K)
        
        # Truncated SVD: keep only the top-`rank` singular components
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)
        k_eff = min(rank, S.shape[0])
        # NOTE: SVD of complex matrices returns U, Vh complex but S real.
        # Multiply via broadcasting (column-scale U) to avoid type mismatch
        # that occurs with torch.diag(S) @ Vh.
        H_truncated = (U[:, :k_eff] * S[:k_eff]) @ Vh[:k_eff, :]
        
        x_smoothed = _anti_diagonal_average(H_truncated)
        
        # Damped update: averaging with previous estimate prevents
        # divergence to noise at high n_iter
        x_updated = (1.0 - damping) * x_smoothed + damping * x_previous
        
        # Trace replacement: known positions are kept exactly
        x = torch.where(mask_known, x_known, x_updated)
    
    return x


def mssa_interpolate_validity_aware(
    gather: np.ndarray,
    trace_mask: np.ndarray,
    validity: np.ndarray,
    rank: int = 10,
    n_iter: int = 10,
    damping: float = 0.5,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    MSSA-based seismic trace reconstruction, validity-aware.
    
    Args:
        gather:     (T, O) input gather with missing traces (zero at missing).
                    T = time samples, O = offset positions on common grid.
        trace_mask: (O,) {0,1} per-trace mask. 1 = present, 0 = missing.
        validity:   (O,) {0,1} per-trace validity. 1 = physical recording,
                    0 = outside streamer (zero-pad).
        rank:       SVD truncation rank (default 10, typical 5-20).
        n_iter:     number of MSSA iterations (default 10).
        damping:    iterative update damping factor in [0, 1] (default 0.5).
                    0 = full replace (can diverge), 1 = no update.
                    Typical 0.3-0.7. Required for stability with n_iter > 5.
        device:     'cuda' or 'cpu' (default: 'cuda' if available).
    
    Returns:
        recon: (T, O) reconstructed gather with same shape as input.
               Outside the validity region the gather stays zero.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    T, O = gather.shape
    
    # Restrict to valid offset positions only
    valid_idx = np.where(validity > 0.5)[0]
    if len(valid_idx) < 4:
        # Too few valid traces — skip MSSA, return zeros
        return np.zeros_like(gather)
    
    # Slice to valid range
    sub_gather = gather[:, valid_idx]          # (T, n_valid)
    sub_mask   = trace_mask[valid_idx]          # (n_valid,)
    n_valid    = len(valid_idx)
    
    # FFT along time (rfft for real input — half spectrum)
    sub_torch = torch.from_numpy(sub_gather.astype(np.float32)).to(device)
    X_freq = torch.fft.rfft(sub_torch, dim=0)   # (T//2+1, n_valid) complex
    
    mask_known = torch.from_numpy(sub_mask > 0.5).to(device)
    
    n_freq = X_freq.shape[0]
    X_recon = X_freq.clone()
    
    # Process each frequency independently
    # (could be batched but SVD of varying matrices is tricky)
    for f_idx in range(n_freq):
        X_recon[f_idx] = _mssa_one_frequency(
            X_freq[f_idx], mask_known,
            rank=rank, n_iter=n_iter, damping=damping,
        )
    
    # Inverse FFT
    recon_torch = torch.fft.irfft(X_recon, n=T, dim=0)    # (T, n_valid) real
    recon_sub = recon_torch.cpu().numpy()
    
    # Re-embed in full output (preserves zero-padded regions)
    recon = np.zeros_like(gather)
    recon[:, valid_idx] = recon_sub
    
    return recon


def mssa_interpolate_validity_aware_fast(
    gather: np.ndarray,
    trace_mask: np.ndarray,
    validity: np.ndarray,
    rank: int = 10,
    n_iter: int = 10,
    n_freq_max: Optional[int] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Faster variant: limits MSSA to the low-frequency band (n_freq_max bins).
    Higher frequencies are left as in the input, i.e. not reconstructed
    (zero on the missing traces); rationale: most reflection energy is at
    low frequencies after the 8 Hz high-pass.
    
    For our 2 ms / 4.5 s gathers, low-pass at 80 Hz corresponds to
    n_freq_max = 360 (out of 1126). With n_freq_max=360, speed-up is 3x.
    """
    if n_freq_max is None:
        return mssa_interpolate_validity_aware(
            gather, trace_mask, validity, rank, n_iter, device,
        )
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    T, O = gather.shape
    valid_idx = np.where(validity > 0.5)[0]
    if len(valid_idx) < 4:
        return np.zeros_like(gather)
    
    sub_gather = gather[:, valid_idx]
    sub_mask   = trace_mask[valid_idx]
    
    sub_torch = torch.from_numpy(sub_gather.astype(np.float32)).to(device)
    X_freq = torch.fft.rfft(sub_torch, dim=0)
    
    mask_known = torch.from_numpy(sub_mask > 0.5).to(device)
    
    n_freq = min(n_freq_max, X_freq.shape[0])
    X_recon = X_freq.clone()
    
    for f_idx in range(n_freq):
        X_recon[f_idx] = _mssa_one_frequency(
            X_freq[f_idx], mask_known,
            rank=rank, n_iter=n_iter,
        )
    # Higher frequencies are kept as in original (already zero on missing)
    
    recon_torch = torch.fft.irfft(X_recon, n=T, dim=0)
    recon_sub   = recon_torch.cpu().numpy()
    
    recon = np.zeros_like(gather)
    recon[:, valid_idx] = recon_sub
    
    return recon
