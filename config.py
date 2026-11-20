"""
config.py — U-Net + FiLM for missing-trace interpolation across
multiple marine streamer geometries (1500, 3000, 4500 m).

Reference paper: Brancatelli et al., "Joint training across streamer
geometries for missing-trace reconstruction: A comparative evaluation on
real marine data", Journal of Applied Geophysics (2026),
doi:10.1016/j.jappgeo.2026.106621.

Setup:
  - 6 datasets, 25 m common spatial grid
  - 70/15/15 contiguous split along shot-point axis
  - U-Net (8.36 M params) with FiLM conditioning on streamer length
  - 150 epochs, AdamW, cosine annealing
  - Validation SNR_miss at 30% random masking on the in-distribution
    test set: ~6.9 dB

Path configuration:
  All paths are read from environment variables, with defaults relative
  to the repository root (the folder containing this file):

    DATA_ROOT     SEG-Y files of the field lines   (default: ./data/raw)
    MAPS_ROOT     navigation, bathymetry, basemap  (default: ./data/maps)
    CKPT_ROOT     model checkpoints                (default: ./checkpoints)
    RESULTS_ROOT  JSON results, tables, figures    (default: ./results)

  If DATA_ROOT is not set but SEISMIC_DATA_ROOT is, SEG-Y files are read
  from $SEISMIC_DATA_ROOT/raw/.

  Example:
    export DATA_ROOT=/path/to/your/segy/files
    python train.py

  See data/README.md for the list of expected filenames and the
  download sources.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent


def _resolve_data_root() -> Path:
    if "DATA_ROOT" in os.environ:
        return Path(os.environ["DATA_ROOT"])
    if "SEISMIC_DATA_ROOT" in os.environ:          # alternative variable
        return Path(os.environ["SEISMIC_DATA_ROOT"]) / "raw"
    return REPO_ROOT / "data" / "raw"


DATA_ROOT    = _resolve_data_root()
MAPS_ROOT    = Path(os.environ.get("MAPS_ROOT",    REPO_ROOT / "data" / "maps"))
CKPT_ROOT    = Path(os.environ.get("CKPT_ROOT",    REPO_ROOT / "checkpoints"))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "results"))


@dataclass
class DataConfig:
    """Dataset paths and physical grid configuration."""

    dataset_paths: List[str] = field(default_factory=lambda: [
        # 1500 m streamers
        str(DATA_ROOT / "MEPA_SP_200_500.sgy"),
        str(DATA_ROOT / "GC01_SP1100_1400.sgy"),
        # 3000 m streamers
        str(DATA_ROOT / "IT065_SP1000_1300.sgy"),
        str(DATA_ROOT / "IT89_33_SP600_900.sgy"),
        # 4500 m streamers
        str(DATA_ROOT / "M31_SP250_550.sgy"),
        str(DATA_ROOT / "M32_SP500_800.sgy"),
    ])
    streamer_lengths: List[float] = field(default_factory=lambda: [
        1500.0, 1500.0,    # MEPA, GC01
        3000.0, 3000.0,    # IT065, IT89_33
        4500.0, 4500.0,    # M31, M32
    ])
    max_streamer_length: float = 4500.0

    # Common physical grid (25 m spacing in offset, 2 ms in time)
    dt_target_ms:  float = 2.0
    t_max_s:       float = 4.5
    x_max_m:       float = 4500.0
    dx_target_m:   float = 25.0

    n_time_samples: int   = 2250
    n_max_offsets:  int   = 181
    dt_ms:          float = 2.0

    # Masking strategy
    masking_ratios: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5])
    masking_strategy: str = "random"
    block_size:       int = 4

    # Preprocessing
    clip_sigma:          float = 3.0
    normalize_per_trace: bool  = True

    # High-pass filter to suppress swell noise
    apply_filter: bool  = True
    filter_f_cut: float = 8.0

    val_fraction: float = 0.15
    random_seed:  int   = 42


@dataclass
class ModelConfig:
    """U-Net + FiLM architecture (8.36 M parameters)."""

    enc_channels:        Tuple = (32, 64, 128, 256)
    bottleneck_channels: int   = 512

    encoder1_kernel: Tuple[int, int] = (3, 5)  # anisotropic first kernel (time x offset)
    default_kernel:  Tuple[int, int] = (3, 3)

    norm_type:         str = "instance"
    group_norm_groups: int = 8

    # FiLM conditioning
    film_hidden_dim:     int = 64
    film_embedding_type: str = "sinusoidal"

    dropout:      float = 0.1
    in_channels:  int   = 2     # masked gather + binary mask
    out_channels: int   = 1


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Composite loss weights
    lambda_l1:   float = 1.0
    lambda_ssim: float = 0.3
    lambda_fk:   float = 0.1

    use_masked_loss: bool = True

    # Optimizer
    optimizer:     str   = "adamw"
    learning_rate: float = 3e-4
    weight_decay:  float = 1e-4
    grad_clip:     float = 1.0

    # Scheduler
    scheduler: str   = "cosine"
    T_max:     int   = 150
    eta_min:   float = 1e-6

    # Training loop
    epochs:      int = 150
    batch_size:  int = 8
    num_workers: int = 8

    n_folds: int = 3

    checkpoint_dir:           str = str(CKPT_ROOT / "v6_grid25m")
    save_every_n_epochs:      int = 10
    early_stopping_patience:  int = 30


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    metrics: List[str] = field(default_factory=lambda: [
        "snr", "ssim", "mse", "fk_energy_ratio"
    ])
    save_outputs: bool = True
    output_dir:   str  = str(RESULTS_ROOT / "v6_grid25m")


@dataclass
class Config:
    """Top-level configuration container."""
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval:  EvalConfig  = field(default_factory=EvalConfig)
    experiment_name: str = "streamer_interpolation_v6_grid25m"


def get_config() -> Config:
    return Config()
