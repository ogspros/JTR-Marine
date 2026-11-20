"""
jtr_marine/models/chai2020_cond.py — U-Net of Chai et al. (2020) with the conditioning of the
proposed model: mask channel at the input and FiLM geometry conditioning at the
bottleneck.

Purpose: separate the contribution of the backbone from the contribution of the
conditioning and of the training objective. The backbone is Chai2020UNet from
chai2020_unet.py, unchanged except for the number of input channels; the FiLM
layer is the one of the proposed model (models/unet.py FiLMLayer: GroupNorm on the
bottleneck features, sinusoidal embedding of the normalised streamer length,
MLP -> gamma, beta, initialised to identity), applied at the same place, i.e.
after the bottleneck convolutions and before the first decoder level.

Variants (constructor flags):
    in_channels=2, use_film=True   -> Chai + mask + FiLM         (full)
    in_channels=2, use_film=False  -> Chai + mask, no FiLM        (ablation)
    in_channels=1, use_film=False  -> equivalent to Chai2020Wrapper

Parameter names: "net.*" (backbone, as in Chai2020Wrapper) and "film.*".
model_loader.py rebuilds the right variant from the stored weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from jtr_marine.models.chai2020_unet import Chai2020UNet
from jtr_marine.models.unet import FiLMLayer


class Chai2020CondWrapper(nn.Module):
    """forward(x, geometry) -> (B, 1, T, O); x is (B, 2, T, O)."""

    MULT = 16

    def __init__(self, in_channels: int = 2, use_film: bool = True,
                 film_hidden_dim: int = 64, base_channels: int = 64,
                 num_levels: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.use_film = use_film
        self.net = Chai2020UNet(in_channels=in_channels, base_channels=base_channels,
                                num_levels=num_levels)
        bott_ch = base_channels * 2 ** num_levels            # 1024
        self.film = (FiLMLayer(n_channels=bott_ch, hidden_dim=film_hidden_dim)
                     if use_film else None)

    def forward(self, x, geometry=None):
        x = x[:, :self.in_channels]
        T, O = x.shape[-2:]
        pt = (-T) % self.MULT
        po = (-O) % self.MULT
        if pt or po:
            # zeros: padded area = no data (gather) and not recorded (mask)
            x = F.pad(x, (0, po, 0, pt))

        net = self.net
        skips = []
        h = x
        for enc in net.encoders:
            h = enc(h)
            skips.append(h)
            h = net.pool(h)
        h = net.bottleneck(h)
        if self.film is not None:
            if geometry is None:
                raise ValueError("geometry is required when use_film=True")
            h = self.film(h, geometry)
        for dec, skip in zip(net.decoders, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = dec(torch.cat([h, skip], dim=1))
        y = net.head(h)
        return y[..., :T, :O]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
