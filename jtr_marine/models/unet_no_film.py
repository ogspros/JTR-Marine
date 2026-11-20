"""
jtr_marine/models/unet_no_film.py — Plain U-Net without FiLM conditioning.

Same enc_channels, anisotropic first kernel and Instance Norm as the
FiLM-conditioned model, but without the FiLM block at the bottleneck.
The geometry passed to forward() is ignored. Note that the conv blocks here
use ReLU, no residual connections and dropout applied in forward, unlike
ConvBlock in models/unet.py.

Used by scripts/train/train_no_film.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm(norm_type: str, num_channels: int, num_groups: int = 8):
    """Return the requested normalization module."""
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups, num_channels)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class ConvBlock(nn.Module):
    """Double conv + norm + ReLU, then dropout."""
    def __init__(self, in_ch, out_ch, kernel_size,
                 norm_type="instance", num_groups=8, dropout=0.0):
        super().__init__()
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.norm1 = get_norm(norm_type, out_ch, num_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = get_norm(norm_type, out_ch, num_groups)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.dropout(x)
        return x


class UNetNoFiLM(nn.Module):
    """
    Plain U-Net without geometry conditioning: the bottleneck receives
    no FiLM modulation.
    """
    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 1,
                 enc_channels=(32, 64, 128, 256),
                 bottleneck_channels: int = 512,
                 encoder1_kernel=(3, 5),
                 default_kernel=(3, 3),
                 norm_type: str = "instance",
                 group_norm_groups: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.depth = len(enc_channels)

        # Encoder
        self.encoders = nn.ModuleList()
        ch_in = in_channels
        for i, ch_out in enumerate(enc_channels):
            kernel = encoder1_kernel if i == 0 else default_kernel
            self.encoders.append(ConvBlock(
                ch_in, ch_out, kernel_size=kernel,
                norm_type=norm_type, num_groups=group_norm_groups,
                dropout=dropout if i > 0 else 0.0,
            ))
            ch_in = ch_out

        # Bottleneck — NO FiLM
        self.bottleneck = ConvBlock(
            ch_in, bottleneck_channels, kernel_size=default_kernel,
            norm_type=norm_type, num_groups=group_norm_groups,
            dropout=dropout,
        )

        # Decoder
        self.decoders = nn.ModuleList()
        ch_in = bottleneck_channels
        dec_channels = list(reversed(enc_channels))
        for ch_out in dec_channels:
            self.decoders.append(ConvBlock(
                ch_in + ch_out, ch_out, kernel_size=default_kernel,
                norm_type=norm_type, num_groups=group_norm_groups,
                dropout=dropout,
            ))
            ch_in = ch_out

        self.out_conv = nn.Conv2d(ch_in, out_channels, 1)

    def forward(self, x, geometry=None):
        """
        Args:
            x:        [B, in_channels, T, X]
            geometry: ignored (kept for signature compatibility)
        """
        # Encoder
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        # Bottleneck (NO FiLM)
        x = self.bottleneck(x)

        # Decoder with skip connections
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, scale_factor=2, mode="bilinear",
                             align_corners=False)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                 align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.out_conv(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model_no_film(config):
    """Build the no-FiLM model from the project config (config.model)."""
    mc = config.model
    return UNetNoFiLM(
        in_channels         = mc.in_channels,
        out_channels        = mc.out_channels,
        enc_channels        = mc.enc_channels,
        bottleneck_channels = mc.bottleneck_channels,
        encoder1_kernel     = mc.encoder1_kernel,
        default_kernel      = mc.default_kernel,
        norm_type           = mc.norm_type,
        group_norm_groups   = mc.group_norm_groups,
        dropout             = mc.dropout,
    )
