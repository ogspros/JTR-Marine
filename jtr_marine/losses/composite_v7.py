"""
jtr_marine/losses/composite_v7.py — Composite loss with Charbonnier instead of L1.

L_total = lambda_charb * Charbonnier_masked + lambda_ssim * (1 - SSIM) + lambda_fk * L1_fk

Differences from composite.py:
  - Charbonnier loss instead of L1 for the amplitude term
    sqrt((pred - target)^2 + eps^2)
  - More stable near zero than L1 (smoother gradient)
  - Less sensitive to outliers than L2
  - Standard in image super-resolution

Keeps the same interface as composite.py (same forward signature, same
output dict) for drop-in compatibility with the training scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .composite import SSIM2D, fk_loss   # shared with composite.py


# ─────────────────────────────────────────────────────────────────────────────
# Charbonnier loss masked
# ─────────────────────────────────────────────────────────────────────────────

def masked_charbonnier(pred: torch.Tensor,
                       target: torch.Tensor,
                       trace_mask: torch.Tensor,
                       validity: torch.Tensor,
                       eps: float = 1e-3,
                       use_masked: bool = True) -> torch.Tensor:
    """
    Charbonnier loss = sqrt((pred - target)^2 + eps^2)

    Computed only on the masked traces (the ones the model must
    reconstruct) if use_masked=True; otherwise on all valid positions.

    Args:
        pred:       (B, 1, T, O)
        target:     (B, 1, T, O)
        trace_mask: (B, O) — 1=present, 0=removed by masking
        validity:   (B, O) — 1=real data, 0=padding
        eps:        smoothing factor (default 1e-3)
        use_masked: restrict loss to missing offsets only

    Returns:
        scalar loss
    """
    diff_sq = (pred - target) ** 2
    per_pixel = torch.sqrt(diff_sq + eps ** 2)   # Charbonnier per-pixel

    if use_masked:
        loss_mask = ((1 - trace_mask) * validity)              # (B, O)
        loss_mask_2d = loss_mask.unsqueeze(1).unsqueeze(2)     # (B, 1, 1, O)
        loss_mask_2d = loss_mask_2d.expand_as(per_pixel)       # (B, 1, T, O)

        n_valid = loss_mask_2d.sum().clamp(min=1.0)
        return (per_pixel * loss_mask_2d).sum() / n_valid
    else:
        valid_mask = validity.unsqueeze(1).unsqueeze(2).expand_as(per_pixel)
        n_valid = valid_mask.sum().clamp(min=1.0)
        return (per_pixel * valid_mask).sum() / n_valid


# ─────────────────────────────────────────────────────────────────────────────
# Composite loss with Charbonnier term
# ─────────────────────────────────────────────────────────────────────────────

class SeismicReconstructionLossV7(nn.Module):
    """
    L_total = lambda_charb * Charbonnier_masked
            + lambda_ssim  * (1 - SSIM)
            + lambda_fk    * L1_fk

    Same signature and output as SeismicReconstructionLoss (composite.py),
    so it can be used as a drop-in replacement.
    """

    def __init__(self,
                 lambda_charb: float = 1.0,
                 lambda_ssim:  float = 0.5,
                 lambda_fk:    float = 0.3,
                 use_masked_loss: bool = True,
                 eps: float = 1e-3):
        super().__init__()
        self.lambda_charb = lambda_charb
        self.lambda_ssim  = lambda_ssim
        self.lambda_fk    = lambda_fk
        self.use_masked   = use_masked_loss
        self.eps          = eps
        self.ssim = SSIM2D()

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                trace_mask: torch.Tensor,
                validity: torch.Tensor) -> dict:
        l_charb = masked_charbonnier(pred, target, trace_mask, validity,
                                      self.eps, self.use_masked)
        l_ssim  = self.ssim(pred, target)
        l_fk    = fk_loss(pred, target)

        total = (self.lambda_charb * l_charb +
                 self.lambda_ssim  * l_ssim  +
                 self.lambda_fk    * l_fk)

        return {
            "loss":      total,
            "l1":        l_charb.detach(),   # key 'l1' kept for compatibility with the logging code
            "ssim_loss": l_ssim.detach(),
            "fk_loss":   l_fk.detach(),
        }


def build_loss_v7(config) -> SeismicReconstructionLossV7:
    tc = config.train
    # Reuses the config parameter names (lambda_l1 etc.), so config.py does
    # not change: config.train.lambda_l1 is used as lambda_charb.
    return SeismicReconstructionLossV7(
        lambda_charb=tc.lambda_l1,
        lambda_ssim =tc.lambda_ssim,
        lambda_fk   =tc.lambda_fk,
        use_masked_loss=tc.use_masked_loss,
        eps=1e-3,
    )
