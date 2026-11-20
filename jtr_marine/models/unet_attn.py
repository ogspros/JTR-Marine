"""
jtr_marine/models/unet_attn.py — Geometry-conditioned U-Net with attention at the bottleneck.

Identical to models/unet.py EXCEPT for the addition of a self-attention
block applied AFTER FiLM modulation. The attention block can be one of:
  - 'channel'      Squeeze-and-Excitation (lightweight)
  - 'spatial'      Multi-head attention (medium)
  - 'transformer'  Stack of transformer blocks (heavy)

This file is intentionally separate from unet.py so that the compact U-Net is
preserved unchanged. To use the new model in training, import build_model
from this file instead.

Use:
    from jtr_marine.models.unet_attn import build_model
    model = build_model(config)   # default uses attention from config
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Reuse all building blocks from unet.py
from .unet import (
    SinusoidalEmbedding, FiLMLayer, ConvBlock,
    DownBlock, UpBlock, get_norm,
)
from .attention_blocks import build_attention


class SeismicUNetAttn(nn.Module):
    """
    Geometry-conditioned U-Net with optional attention at the bottleneck.

    Identical signature to SeismicUNet plus two extra arguments:
        attention_type:   'none' | 'channel' | 'spatial' | 'transformer'
        attention_kwargs: dict passed to the attention block
    """

    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 1,
                 enc_channels: Tuple = (32, 64, 128, 256),
                 bottleneck_channels: int = 512,
                 encoder1_kernel: Tuple[int, int] = (3, 5),
                 film_hidden_dim: int = 64,
                 norm_type: str = "instance",
                 dropout: float = 0.1,
                 n_groups: int = 8,
                 attention_type: str = "none",
                 attention_kwargs: dict = None):
        super().__init__()

        C = enc_channels
        B = bottleneck_channels
        self.attention_type = attention_type
        attention_kwargs = attention_kwargs or {}

        # ── Encoder (identical to unet.py) ─────────────────────────────────────
        self.enc1 = DownBlock(in_channels, C[0],
                              kernel=encoder1_kernel,
                              norm_type=norm_type, n_groups=n_groups)
        self.enc2 = DownBlock(C[0], C[1], norm_type=norm_type, n_groups=n_groups)
        self.enc3 = DownBlock(C[1], C[2], norm_type=norm_type, n_groups=n_groups)
        self.enc4 = DownBlock(C[2], C[3], norm_type=norm_type, n_groups=n_groups)

        # ── Bottleneck: ConvBlock + FiLM + Attention ──────────────────────
        self.bottleneck_conv = ConvBlock(C[3], B, norm_type=norm_type, n_groups=n_groups)
        self.film = FiLMLayer(n_channels=B, hidden_dim=film_hidden_dim)
        # NEW: attention block after FiLM
        self.attention = build_attention(attention_type, channels=B,
                                          **attention_kwargs)

        # ── Decoder (identical to unet.py) ─────────────────────────────────────
        self.dec4 = UpBlock(B,    C[3], C[3], norm_type=norm_type,
                            dropout=dropout, n_groups=n_groups)
        self.dec3 = UpBlock(C[3], C[2], C[2], norm_type=norm_type,
                            dropout=dropout, n_groups=n_groups)
        self.dec2 = UpBlock(C[2], C[1], C[1], norm_type=norm_type,
                            dropout=dropout, n_groups=n_groups)
        self.dec1 = UpBlock(C[1], C[0], C[0], norm_type=norm_type,
                            dropout=dropout, n_groups=n_groups)

        # ── Output head ───────────────────────────────────────────────────
        self.output_conv = nn.Conv2d(C[0], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv layers, zero-init for output head.
        Identical to unet.py initialization to ensure same starting point.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                          nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Output head: zero init so the network starts with near-zero prediction.
        # Without this, the initial loss is dominated by random output magnitude
        # and the network wastes many epochs recovering.
        nn.init.zeros_(self.output_conv.weight)
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)

    # ── Same forward as unet.py, with attention applied after FiLM ─────────────
    def forward(self, x: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1, skip1 = self.enc1(x)
        e2, skip2 = self.enc2(e1)
        e3, skip3 = self.enc3(e2)
        e4, skip4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck_conv(e4)
        b = self.film(b, geometry)
        b = self.attention(b)        # ← attention applied AFTER FiLM modulation

        # Decoder
        d4 = self.dec4(b,  skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        return self.output_conv(d1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config) -> SeismicUNetAttn:
    """
    Build the attention-enabled model from a project config.

    Reads from config.model:
      attention_type    (default 'spatial')
      attention_kwargs  (default None → uses defaults of the attention class)

    Falls back gracefully if config.model does not contain these fields.
    """
    mc = config.model
    attention_type = getattr(mc, "attention_type", "spatial")
    attention_kwargs = getattr(mc, "attention_kwargs", None)

    return SeismicUNetAttn(
        in_channels         = mc.in_channels,
        out_channels        = mc.out_channels,
        enc_channels        = tuple(mc.enc_channels),
        bottleneck_channels = mc.bottleneck_channels,
        encoder1_kernel     = tuple(mc.encoder1_kernel),
        film_hidden_dim     = mc.film_hidden_dim,
        norm_type           = mc.norm_type,
        dropout             = mc.dropout,
        n_groups            = getattr(mc, "n_groups", 8),
        attention_type      = attention_type,
        attention_kwargs    = attention_kwargs,
    )
