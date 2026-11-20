"""
jtr_marine/models/unet.py — Geometry-conditioned U-Net for seismic offset reconstruction.

Architecture:
  - Encoder: 4 levels, anisotropic kernel at level 1 (3×5 instead of 3×3)
  - Bottleneck: FiLM conditioning on streamer length
  - Decoder: 4 levels with skip connections via concatenation
  - Normalization: Instance Norm (robust to small batch sizes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# FiLM: Feature-wise Linear Modulation
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """
    Encodes a scalar (streamer length in [0,1]) as a sinusoidal embedding.
    More expressive than a raw scalar input to the MLP.
    Inspired by positional encodings in transformers.
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,) scalar values in [0, 1]
        device = x.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device) * -(math.log(10000) / (half - 1))
        )
        args = x[:, None] * freqs[None, :]    # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on geometry.

    Given feature map F of shape (B, C, T, O) and conditioning scalar g:
        γ, β = MLP(embed(g))          → shape (B, C)
        output = γ * GroupNorm(F) + β

    This allows the bottleneck to globally modulate all channels
    based on the streamer length before reconstruction begins.
    """
    def __init__(self, n_channels: int, hidden_dim: int = 64, embed_dim: int = 64):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(32, n_channels), num_channels=n_channels)
        self.embed = SinusoidalEmbedding(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_channels * 2),  # → γ and β
        )
        # Initialize γ near 1, β near 0 for stable training start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.ones_(self.mlp[-1].bias[:n_channels])    # γ bias = 1
        nn.init.zeros_(self.mlp[-1].bias[n_channels:])   # β bias = 0

    def forward(self, x: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, O), geometry: (B,)
        emb = self.embed(geometry)             # (B, embed_dim)
        params = self.mlp(emb)                 # (B, 2C)
        gamma, beta = params.chunk(2, dim=-1)  # each (B, C)

        # Reshape for broadcasting over T and O dimensions
        gamma = gamma[:, :, None, None]        # (B, C, 1, 1)
        beta  = beta[:, :, None, None]

        x_norm = self.norm(x)
        return gamma * x_norm + beta


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def get_norm(norm_type: str, n_channels: int, n_groups: int = 8) -> nn.Module:
    if norm_type == "instance":
        return nn.InstanceNorm2d(n_channels, affine=True)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups=min(n_groups, n_channels), num_channels=n_channels)
    elif norm_type == "batch":
        return nn.BatchNorm2d(n_channels)
    else:
        raise ValueError(f"Unknown norm: {norm_type}")


class ConvBlock(nn.Module):
    """
    Residual double-conv block: Conv → Norm → GELU → Conv → Norm → GELU,
    plus a residual connection. Used in encoder and decoder.
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel: Tuple[int, int] = (3, 3),
                 norm_type: str = "instance",
                 dropout: float = 0.0,
                 n_groups: int = 8):
        super().__init__()
        # Padding to maintain spatial size
        pad_t = kernel[0] // 2
        pad_x = kernel[1] // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=(pad_t, pad_x), bias=False),
            get_norm(norm_type, out_ch, n_groups),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, (3, 3), padding=1, bias=False),
            get_norm(norm_type, out_ch, n_groups),
            nn.GELU(),
        )
        # Dropout module (non-zero rate only in decoder blocks); it is created
        # here but not applied in forward()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Residual connection (1×1 conv if channel dims differ)
        self.residual = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.residual(x)


class DownBlock(nn.Module):
    """Encoder block: ConvBlock + max-pool ↓2"""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: Tuple[int, int] = (3, 3),
                 norm_type: str = "instance",
                 n_groups: int = 8):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel, norm_type, n_groups=n_groups)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)      # stored for skip connection
        down = self.pool(skip)   # downsampled for next level
        return down, skip


class UpBlock(nn.Module):
    """
    Decoder block: bilinear ↑2 + concat(skip) + ConvBlock
    Uses bilinear upsampling instead of transposed conv to avoid checkerboard.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 norm_type: str = "instance",
                 dropout: float = 0.0,
                 n_groups: int = 8):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # in_ch (upsampled) + skip_ch (from encoder) → out_ch
        self.conv = ConvBlock(in_ch + skip_ch, out_ch,
                              kernel=(3, 3), norm_type=norm_type,
                              dropout=dropout, n_groups=n_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle size mismatch from odd dimensions (important with variable offsets)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=True)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full U-Net
# ─────────────────────────────────────────────────────────────────────────────

class SeismicUNet(nn.Module):
    """
    Geometry-conditioned U-Net for seismic offset reconstruction.

    Args:
        in_channels:        2 (masked gather + binary mask)
        out_channels:       1 (reconstructed gather)
        enc_channels:       channel progression in encoder, e.g. (32, 64, 128, 256)
        bottleneck_channels: channels at bottleneck, e.g. 512
        encoder1_kernel:    anisotropic kernel for first encoder level, e.g. (3, 5)
        film_hidden_dim:    MLP width for FiLM geometry conditioning
        norm_type:          "instance" | "group" | "batch"
        dropout:            dropout rate passed to the decoder blocks (the
                            Dropout2d module is created but not applied in
                            ConvBlock.forward)
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
                 n_groups: int = 8):
        super().__init__()

        C = enc_channels
        B = bottleneck_channels

        # ── Encoder ──────────────────────────────────────────────────────────
        # Level 1: anisotropic kernel to capture seismic anisotropy
        self.enc1 = DownBlock(in_channels, C[0],
                              kernel=encoder1_kernel,
                              norm_type=norm_type, n_groups=n_groups)
        # Levels 2–4: standard 3×3 kernels
        self.enc2 = DownBlock(C[0], C[1], norm_type=norm_type, n_groups=n_groups)
        self.enc3 = DownBlock(C[1], C[2], norm_type=norm_type, n_groups=n_groups)
        self.enc4 = DownBlock(C[2], C[3], norm_type=norm_type, n_groups=n_groups)

        # ── Bottleneck with FiLM ─────────────────────────────────────────────
        self.bottleneck_conv = ConvBlock(C[3], B, norm_type=norm_type, n_groups=n_groups)
        self.film = FiLMLayer(n_channels=B, hidden_dim=film_hidden_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec4 = UpBlock(B,    C[3], C[3], norm_type=norm_type,
                            dropout=dropout, n_groups=n_groups)
        self.dec3 = UpBlock(C[3], C[2], C[2], norm_type=norm_type,
                            dropout=dropout, n_groups=n_groups)
        self.dec2 = UpBlock(C[2], C[1], C[1], norm_type=norm_type,
                            dropout=dropout, n_groups=n_groups)
        self.dec1 = UpBlock(C[1], C[0], C[0], norm_type=norm_type,
                            dropout=dropout, n_groups=n_groups)

        # ── Output head ──────────────────────────────────────────────────────
        # 1×1 conv to project to output channels, no activation
        # (regression output — tanh is optional but may help if data is in [-1, 1])
        self.output_conv = nn.Conv2d(C[0], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv layers, zero-init for output head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Output head: small init to start near zero prediction
        nn.init.zeros_(self.output_conv.weight)
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)

    def forward(self,
                x: torch.Tensor,
                geometry: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        (B, 2, T, O) — masked gather + binary mask
            geometry: (B,)         — normalized streamer length in [0, 1]

        Returns:
            (B, 1, T, O) — reconstructed gather (all offsets)
        """
        # ── Encoder ──
        x1, s1 = self.enc1(x)    # s1: skip (B, C0, T,   O)
        x2, s2 = self.enc2(x1)   # s2: skip (B, C1, T/2, O/2)
        x3, s3 = self.enc3(x2)   # s3: skip (B, C2, T/4, O/4)
        x4, s4 = self.enc4(x3)   # s4: skip (B, C3, T/8, O/8)

        # ── Bottleneck + FiLM ──
        b = self.bottleneck_conv(x4)      # (B, B_ch, T/16, O/16)
        b = self.film(b, geometry)        # modulate by streamer length

        # ── Decoder ──
        d4 = self.dec4(b,  s4)   # (B, C3, T/8,  O/8)
        d3 = self.dec3(d4, s3)   # (B, C2, T/4,  O/4)
        d2 = self.dec2(d3, s2)   # (B, C1, T/2,  O/2)
        d1 = self.dec1(d2, s1)   # (B, C0, T,    O)

        out = self.output_conv(d1)   # (B, 1, T, O)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config) -> SeismicUNet:
    """Build model from config."""
    mc = config.model
    return SeismicUNet(
        in_channels=mc.in_channels,
        out_channels=mc.out_channels,
        enc_channels=mc.enc_channels,
        bottleneck_channels=mc.bottleneck_channels,
        encoder1_kernel=mc.encoder1_kernel,
        film_hidden_dim=mc.film_hidden_dim,
        norm_type=mc.norm_type,
        dropout=mc.dropout,
        n_groups=mc.group_norm_groups,
    )
