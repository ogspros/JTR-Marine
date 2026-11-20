#!/usr/bin/env python3
"""
train_v6_seeds.py — Re-run the proposed v6 model with a controlled torch seed,
to measure run-to-run variance (compact U-Net, seeds 43 and 44 of Table 9).

Why
---
config.data.random_seed is passed to numpy RNGs in jtr_marine/data/dataset.py,
so the splits and the masking patterns are deterministic. But train.py does
not set the torch seed unless --seed is given, so by default weight
initialisation, dropout and DataLoader shuffling are NOT controlled: every
run of train.py differs.

This wrapper sets the torch seed explicitly (as train_gan.py does), so the
proposed model and the GAN baseline are compared under the same protocol:
identical data pipeline, identical number of epochs, seed-controlled training
stochasticity.

Usage
-----
    python scripts/train/train_v6_seeds.py --seed 43
    python scripts/train/train_v6_seeds.py --seed 44

Each run writes to its own checkpoint directory (CKPT_ROOT/v6_seed<SEED>/),
leaving the checkpoints of the paper model (CKPT_ROOT/v6_grid25m/) untouched.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT

import argparse

import numpy as np
import torch

from config import get_config
from train import train_within_geology


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True,
                   help="torch seed for weight init / dropout / shuffling")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="default: CKPT_ROOT/v6_seed<SEED>")
    args = p.parse_args()

    # Control the training stochasticity that train.py leaves unseeded by default.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = get_config()

    # NOTE: config.data.random_seed is deliberately left at its default, so the
    # splits and masking patterns are identical across runs. Only the training
    # stochasticity varies, which is what we want to measure.
    ckpt_dir = args.checkpoint_dir or str(CKPT_ROOT / f"v6_seed{args.seed}")
    config.train.checkpoint_dir = ckpt_dir

    print("=" * 62)
    print(f"v6 MODEL — variance run, torch seed = {args.seed}")
    print(f"checkpoint_dir = {ckpt_dir}")
    print(f"data random_seed (splits/masking) = {config.data.random_seed}")
    print("=" * 62)

    train_within_geology(config)


if __name__ == "__main__":
    main()
