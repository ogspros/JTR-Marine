"""
jtr_marine/models/unet_ablation.py — The proposed U-Net (models/unet.py) with two
design choices made configurable, to measure their effect in isolation:

    kernel     : kernel size of every convolution (paper model: 3x3, 3x5 at level 1)
    norm_type  : "instance" (paper model), "group", "batch", or "none" (no normalisation
                 inside the conv blocks, as in Chai et al. 2020)

Everything else is identical to the paper model: channel widths, residual 1x1 connections,
GELU, bilinear upsampling, decoder dropout, FiLM at the bottleneck (its own
GroupNorm is part of the FiLM layer and is kept), weight initialisation,
zero-initialised output head. models/unet.py is not modified.

Parameter names match models/unet.py exactly (Identity layers carry no parameters, so the
conv weights keep indices block.0 and block.3). model_loader.py recognises the
variant from the kernel shape and from the presence of norm parameters.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from jtr_marine.models.unet import FiLMLayer


def get_norm(norm_type: str, n_channels: int, n_groups: int = 8) -> nn.Module:
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "instance":
        return nn.InstanceNorm2d(n_channels, affine=True)
    if norm_type == "group":
        return nn.GroupNorm(num_groups=min(n_groups, n_channels), num_channels=n_channels)
    if norm_type == "batch":
        return nn.BatchNorm2d(n_channels)
    raise ValueError(f"Unknown norm: {norm_type}")


def _pad(k: Tuple[int, int]) -> Tuple[int, int]:
    return (k[0] // 2, k[1] // 2)


class ConvBlock(nn.Module):
    """As ConvBlock of unet.py, but the second convolution uses `kernel2` instead of fixed 3x3."""

    def __init__(self, in_ch, out_ch, kernel=(3, 3), kernel2=(3, 3),
                 norm_type="instance", dropout=0.0, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=_pad(kernel), bias=False),
            get_norm(norm_type, out_ch, n_groups),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel2, padding=_pad(kernel2), bias=False),
            get_norm(norm_type, out_ch, n_groups),
            nn.GELU(),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.residual = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        return self.block(x) + self.residual(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, kernel2, norm_type, n_groups):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel, kernel2, norm_type, n_groups=n_groups)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, kernel, norm_type, dropout, n_groups):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, kernel, kernel, norm_type,
                              dropout=dropout, n_groups=n_groups)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


class SeismicUNetAblation(nn.Module):
    def __init__(self, in_channels=2, out_channels=1,
                 enc_channels=(32, 64, 128, 256), bottleneck_channels=512,
                 kernel=(3, 3), encoder1_kernel=(3, 5), film_hidden_dim=64,
                 norm_type="instance", dropout=0.1, n_groups=8):
        super().__init__()
        self.kernel = tuple(kernel)
        self.norm_type = norm_type
        C, B, k = enc_channels, bottleneck_channels, tuple(kernel)

        self.enc1 = DownBlock(in_channels, C[0], tuple(encoder1_kernel), k, norm_type, n_groups)
        self.enc2 = DownBlock(C[0], C[1], k, k, norm_type, n_groups)
        self.enc3 = DownBlock(C[1], C[2], k, k, norm_type, n_groups)
        self.enc4 = DownBlock(C[2], C[3], k, k, norm_type, n_groups)

        self.bottleneck_conv = ConvBlock(C[3], B, k, k, norm_type, n_groups=n_groups)
        self.film = FiLMLayer(n_channels=B, hidden_dim=film_hidden_dim)

        self.dec4 = UpBlock(B,    C[3], C[3], k, norm_type, dropout, n_groups)
        self.dec3 = UpBlock(C[3], C[2], C[2], k, norm_type, dropout, n_groups)
        self.dec2 = UpBlock(C[2], C[1], C[1], k, norm_type, dropout, n_groups)
        self.dec1 = UpBlock(C[1], C[0], C[0], k, norm_type, dropout, n_groups)

        self.output_conv = nn.Conv2d(C[0], out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.output_conv.weight)
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, geometry):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        b = self.film(self.bottleneck_conv(x4), geometry)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.output_conv(d1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_ablation_model(config, kernel=3, norm_type="instance"):
    """Compact U-Net from config, with the ablated kernel size and normalisation.
    kernel=3 keeps the paper kernels (3x5 at level 1); kernel=K sets KxK everywhere."""
    mc = config.model
    k = (kernel, kernel)
    enc1 = tuple(mc.encoder1_kernel) if kernel == 3 else k
    return SeismicUNetAblation(
        in_channels=mc.in_channels, out_channels=mc.out_channels,
        enc_channels=mc.enc_channels, bottleneck_channels=mc.bottleneck_channels,
        kernel=k, encoder1_kernel=enc1, film_hidden_dim=mc.film_hidden_dim,
        norm_type=norm_type, dropout=mc.dropout, n_groups=mc.group_norm_groups,
    )
