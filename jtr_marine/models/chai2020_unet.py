"""
jtr_marine/models/chai2020_unet.py — Re-implementation of the U-Net of Chai et al. (2020),
"Deep Learning for Regularly Missing Data Reconstruction", IEEE TGRS,
doi:10.1109/TGRS.2019.2961015, used as an independent published baseline.

Architecture (paper-explicit, as documented in the public GeoBrain benchmark
reproduction, huggingface.co/GeoBrain/seismic-interpolation-benchmark):
  4 encoder levels + bottleneck, two 5x5 convolutions each (64, 128, 256, 512,
  1024 channels), ReLU, no normalisation, 2x2 max pooling; decoder with x2
  upsampling (nearest; the only reproduction assumption), concatenation of the
  skip connection, two 5x5 convolutions; 1x1 output convolution.
  19 convolutions, 18 ReLU, 87,149,953 parameters. Single input channel (the
  decimated gather): no mask channel, no conditioning. Loss: MSE.

Parameter names follow the GeoBrain checkpoint exactly (encoders.i.block.0/3,
bottleneck.block.0/3, decoders.i.block.0/3, head), so that the structure can be
verified by loading their state_dict with strict=True.

Chai2020Wrapper makes the network callable as the project model,
model(x, geometry), with x of shape (B, 2, T, O): it uses only channel 0 (the
masked gather), ignores the mask channel and the geometry code, pads T and O to
multiples of 16 with zeros and crops the output back.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(cin, cout, k=5):
    p = k // 2
    # indices 0 and 3 carry parameters, as in the reference checkpoint
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=p, bias=True), nn.Identity(), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, k, padding=p, bias=True), nn.Identity(), nn.ReLU(inplace=True),
    )


class _Block(nn.Module):
    def __init__(self, cin, cout, k=5):
        super().__init__()
        self.block = double_conv(cin, cout, k)

    def forward(self, x):
        return self.block(x)


class Chai2020UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 num_levels=4, kernel_size=5):
        super().__init__()
        ch = [base_channels * 2 ** i for i in range(num_levels + 1)]   # 64..1024
        self.encoders = nn.ModuleList()
        cin = in_channels
        for i in range(num_levels):
            self.encoders.append(_Block(cin, ch[i], kernel_size))
            cin = ch[i]
        self.bottleneck = _Block(ch[num_levels - 1], ch[num_levels], kernel_size)
        self.decoders = nn.ModuleList()
        for i in reversed(range(num_levels)):
            self.decoders.append(_Block(ch[i + 1] + ch[i], ch[i], kernel_size))
        self.head = nn.Conv2d(ch[0], out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


class Chai2020Wrapper(nn.Module):
    """Project-compatible interface: forward(x, geometry) -> (B, 1, T, O)."""

    MULT = 16

    def __init__(self, **kw):
        super().__init__()
        self.net = Chai2020UNet(**kw)

    def forward(self, x, geometry=None):
        x = x[:, :1]                                   # masked gather only
        T, O = x.shape[-2:]
        pt = (-T) % self.MULT
        po = (-O) % self.MULT
        if pt or po:
            x = F.pad(x, (0, po, 0, pt))               # zeros, as in the masked data
        y = self.net(x)
        return y[..., :T, :O]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
