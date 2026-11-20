"""
jtr_marine/losses/composite.py — Composite loss for seismic reconstruction.

L_total = λ1 * L1_masked + λ2 * SSIM + λ3 * L1_fk

The f-k term (frequency-wavenumber) forces spectral coherence:
the model must reconstruct not just amplitudes but the spatial
continuity of wave events — this is the physics-aware component.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SSIM (Structural Similarity Index)
# ─────────────────────────────────────────────────────────────────────────────

class SSIM2D(nn.Module):
    """
    Differentiable SSIM for 2D feature maps (B, 1, T, O).
    Uses Gaussian-weighted local windows.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._gaussian_window(window_size, sigma))

    @staticmethod
    def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g[:, None] * g[None, :]
        return window[None, None, :, :]  # (1, 1, W, W)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: (B, 1, T, O)
        Returns:
            scalar loss (1 - mean_SSIM), in [0, 2]
        """
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        w = self.window.to(pred.device)

        pad = self.window_size // 2
        mu1 = F.conv2d(pred,   w, padding=pad)
        mu2 = F.conv2d(target, w, padding=pad)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu12   = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred,   w, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(target * target, w, padding=pad) - mu2_sq
        sigma12   = F.conv2d(pred * target,  w, padding=pad) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return 1.0 - ssim_map.mean()


# ─────────────────────────────────────────────────────────────────────────────
# F-K domain loss
# ─────────────────────────────────────────────────────────────────────────────

def fk_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 loss on the magnitude of the 2D FFT (f-k spectrum).

    The f-k domain is the natural space for seismic data:
    - Each linear event (refraction, direct wave) maps to a point
    - Aliased energy falls outside the signal cone
    - Matching f-k spectra ensures spatial coherence of reconstructed events

    Args:
        pred, target: (B, 1, T, O)
    Returns:
        scalar loss
    """
    # 2D FFT along time (dim=-2) and offset (dim=-1)
    pred_fft   = torch.fft.rfft2(pred,   norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")

    # L1 on magnitude (phase matching is too strict for this task)
    pred_mag   = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    return F.l1_loss(pred_mag, target_mag)


# ─────────────────────────────────────────────────────────────────────────────
# Masked L1
# ─────────────────────────────────────────────────────────────────────────────

def masked_l1(pred: torch.Tensor,
              target: torch.Tensor,
              trace_mask: torch.Tensor,
              validity: torch.Tensor,
              use_masked: bool = True) -> torch.Tensor:
    """
    L1 loss computed only on missing offsets (the ones the model must reconstruct).

    If use_masked=False, computes standard L1 over all valid traces.

    Args:
        pred:       (B, 1, T, O)
        target:     (B, 1, T, O)
        trace_mask: (B, O) — 1 = present, 0 = removed by masking
        validity:   (B, O) — 1 = real data, 0 = padding
        use_masked: whether to restrict loss to missing offsets only
    """
    if use_masked:
        # Compute loss only where trace was REMOVED and data is VALID (not padding)
        loss_mask = ((1 - trace_mask) * validity)  # (B, O): 1 where missing + valid
        loss_mask_2d = loss_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, O)
        loss_mask_2d = loss_mask_2d.expand_as(pred)         # (B, 1, T, O)

        n_valid = loss_mask_2d.sum().clamp(min=1.0)
        return (torch.abs(pred - target) * loss_mask_2d).sum() / n_valid
    else:
        # Standard L1 over all valid positions
        valid_mask = validity.unsqueeze(1).unsqueeze(2).expand_as(pred)
        n_valid = valid_mask.sum().clamp(min=1.0)
        return (torch.abs(pred - target) * valid_mask).sum() / n_valid


# ─────────────────────────────────────────────────────────────────────────────
# Composite loss
# ─────────────────────────────────────────────────────────────────────────────

class SeismicReconstructionLoss(nn.Module):
    """
    L_total = λ_l1 * L1_masked
            + λ_ssim * (1 - SSIM)
            + λ_fk * L1_fk

    The three terms are complementary:
    - L1_masked:  amplitude accuracy on missing traces
    - SSIM:       structural / perceptual coherence (continuity of events)
    - L1_fk:      frequency-wavenumber consistency (physical prior)
    """

    def __init__(self,
                 lambda_l1:   float = 1.0,
                 lambda_ssim: float = 0.5,
                 lambda_fk:   float = 0.3,
                 use_masked_loss: bool = True):
        super().__init__()
        self.lambda_l1   = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_fk   = lambda_fk
        self.use_masked  = use_masked_loss
        self.ssim = SSIM2D()

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                trace_mask: torch.Tensor,
                validity: torch.Tensor
                ) -> dict:
        """
        Args:
            pred:       (B, 1, T, O)
            target:     (B, 1, T, O)
            trace_mask: (B, O) binary — 1=present, 0=removed
            validity:   (B, O) binary — 1=real data, 0=padding

        Returns:
            dict with 'loss' (total) and individual component losses
        """
        l_l1   = masked_l1(pred, target, trace_mask, validity, self.use_masked)
        l_ssim = self.ssim(pred, target)
        l_fk   = fk_loss(pred, target)

        total = (self.lambda_l1   * l_l1 +
                 self.lambda_ssim * l_ssim +
                 self.lambda_fk   * l_fk)

        return {
            "loss":      total,
            "l1":        l_l1.detach(),
            "ssim_loss": l_ssim.detach(),
            "fk_loss":   l_fk.detach(),
        }


def build_loss(config) -> SeismicReconstructionLoss:
    tc = config.train
    return SeismicReconstructionLoss(
        lambda_l1=tc.lambda_l1,
        lambda_ssim=tc.lambda_ssim,
        lambda_fk=tc.lambda_fk,
        use_masked_loss=tc.use_masked_loss,
    )
