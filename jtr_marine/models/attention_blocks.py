"""
jtr_marine/models/attention_blocks.py — Self-attention modules for the U-Net bottleneck.

Three variants are provided, each with a different memory/compute tradeoff:

  1. ChannelAttention (SE-like)
       - Lightweight, no memory issues
       - Operates only on channel statistics, ignores spatial correlations
       - Expected gain: ~+0.3-0.5 dB

  2. SpatialMHA
       - Multi-head self-attention with 2x downsample to manage memory
       - Captures long-range spatial dependencies
       - Expected gain: ~+0.5-1.0 dB
       - Memory: ~12 GB at batch_size=8

  3. MiniTransformer
       - Stack of N transformer blocks (default N=2)
       - Pre-norm + MLP after attention
       - Most expressive
       - Expected gain: ~+1.0-1.5 dB
       - Memory: ~18 GB at batch_size=8 with N=2

All variants accept input shape (B, C, H, W) and return same shape.

Use in unet_attn.py to wrap the bottleneck:
    self.attention = build_attention(attention_type, channels=B)
    ...
    b = self.bottleneck_conv(x)
    b = self.film(b, geometry)
    b = self.attention(b)        # ← attention HERE, AFTER FiLM modulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Variant 1: Channel attention (Squeeze-and-Excitation-like)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel-wise attention via global pooling + small MLP.

    Cost: negligible. Memory: O(C), not O(H*W).

    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C = x.shape[:2]
        s = self.gap(x).view(B, C)         # (B, C)
        w = self.mlp(s).view(B, C, 1, 1)   # (B, C, 1, 1)
        return x * w


# ─────────────────────────────────────────────────────────────────────────────
# Variant 2: Spatial multi-head attention with downsample
# ─────────────────────────────────────────────────────────────────────────────

class SpatialMHA(nn.Module):
    """
    Multi-head self-attention applied to the spatial dimension of the
    bottleneck feature map. To control memory cost, the feature map is
    optionally downsampled (default 2x) before attention and upsampled
    afterwards.

    Workflow:
      (B, C, H, W) → avg-pool ↓2 → flatten → MHA → unflatten → bilinear ↑2

    Attention is computed on (H*W)/4 tokens, reducing memory by 16x.
    """
    def __init__(self,
                 channels: int,
                 num_heads: int = 8,
                 downsample: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels   = channels
        self.num_heads  = num_heads
        self.downsample = downsample
        self.norm       = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Learnable positional embedding (initialized lazily on first forward)
        self.pos_emb = None
        self._max_tokens = None

    def _maybe_init_pos_emb(self, n_tokens: int, device, dtype):
        """Initialize positional embedding on first forward call."""
        if self.pos_emb is None or self._max_tokens < n_tokens:
            pos = torch.randn(1, n_tokens, self.channels,
                              device=device, dtype=dtype) * 0.02
            self.pos_emb = nn.Parameter(pos)
            self._max_tokens = n_tokens
            # register as parameter
            self.register_parameter("pos_emb_param", self.pos_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Downsample
        if self.downsample > 1:
            x_ds = F.avg_pool2d(x, kernel_size=self.downsample)
        else:
            x_ds = x
        _, _, Hd, Wd = x_ds.shape
        n_tokens = Hd * Wd

        # Lazy init of positional embedding
        self._maybe_init_pos_emb(n_tokens, x.device, x.dtype)

        # Flatten spatial dims → (B, n_tokens, C)
        seq = x_ds.flatten(2).transpose(1, 2)

        # Pre-norm + positional embedding
        seq = self.norm(seq) + self.pos_emb[:, :n_tokens]

        # Self-attention with residual
        attn_out, _ = self.mha(seq, seq, seq, need_weights=False)
        seq = seq + attn_out                              # (B, n_tokens, C)

        # Unflatten back to (B, C, Hd, Wd)
        out_ds = seq.transpose(1, 2).reshape(B, C, Hd, Wd)

        # Upsample to original resolution
        if self.downsample > 1:
            out = F.interpolate(out_ds, size=(H, W),
                                mode="bilinear", align_corners=False)
        else:
            out = out_ds

        # Residual on the full-resolution path
        return x + out


# ─────────────────────────────────────────────────────────────────────────────
# Variant 3: Mini transformer (N stacked blocks)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block: LN → MHA → residual → LN → MLP → residual
    """
    def __init__(self,
                 channels: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 2.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_tokens, C)
        h = self.norm1(x)
        attn_out, _ = self.mha(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MiniTransformer(nn.Module):
    """
    Stack of transformer blocks operating on the (downsampled) bottleneck.

    Cost grows linearly with num_layers. Memory grows as O(n_tokens^2).
    """
    def __init__(self,
                 channels: int,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 mlp_ratio: float = 2.0,
                 downsample: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.channels   = channels
        self.downsample = downsample

        self.blocks = nn.ModuleList([
            TransformerBlock(channels, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.pos_emb = None
        self._max_tokens = None

    def _maybe_init_pos_emb(self, n_tokens: int, device, dtype):
        if self.pos_emb is None or self._max_tokens < n_tokens:
            pos = torch.randn(1, n_tokens, self.channels,
                              device=device, dtype=dtype) * 0.02
            self.pos_emb = nn.Parameter(pos)
            self._max_tokens = n_tokens
            self.register_parameter("pos_emb_param", self.pos_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Downsample
        if self.downsample > 1:
            x_ds = F.avg_pool2d(x, kernel_size=self.downsample)
        else:
            x_ds = x
        _, _, Hd, Wd = x_ds.shape
        n_tokens = Hd * Wd

        # Positional embedding
        self._maybe_init_pos_emb(n_tokens, x.device, x.dtype)

        # Flatten + pos
        seq = x_ds.flatten(2).transpose(1, 2)             # (B, n_tokens, C)
        seq = seq + self.pos_emb[:, :n_tokens]

        # Stack of transformer blocks
        for block in self.blocks:
            seq = block(seq)

        # Unflatten + upsample
        out_ds = seq.transpose(1, 2).reshape(B, C, Hd, Wd)
        if self.downsample > 1:
            out = F.interpolate(out_ds, size=(H, W),
                                mode="bilinear", align_corners=False)
        else:
            out = out_ds

        return x + out


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_attention(attention_type: str, channels: int, **kwargs) -> nn.Module:
    """
    Build an attention block by name.

    Args:
        attention_type: one of {'none', 'channel', 'spatial', 'transformer'}
        channels:       feature channels of the bottleneck
        **kwargs:       passed to the specific module

    Returns:
        nn.Module that accepts (B, C, H, W) and returns same shape.
    """
    if attention_type in (None, "none"):
        return nn.Identity()
    elif attention_type == "channel":
        return ChannelAttention(channels=channels,
                                reduction=kwargs.get("reduction", 16))
    elif attention_type == "spatial":
        return SpatialMHA(channels=channels,
                          num_heads=kwargs.get("num_heads", 8),
                          downsample=kwargs.get("downsample", 2),
                          dropout=kwargs.get("dropout", 0.0))
    elif attention_type == "transformer":
        return MiniTransformer(channels=channels,
                               num_layers=kwargs.get("num_layers", 2),
                               num_heads=kwargs.get("num_heads", 8),
                               mlp_ratio=kwargs.get("mlp_ratio", 2.0),
                               downsample=kwargs.get("downsample", 2),
                               dropout=kwargs.get("dropout", 0.0))
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")
