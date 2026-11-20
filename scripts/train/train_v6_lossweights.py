#!/usr/bin/env python3
"""
train_v6_lossweights.py — Train the proposed v6 model with given loss weights.

Used by sweep_loss_weights.py for the sensitivity analysis of the loss
weights around the default lambda_l1 = 1.0, lambda_ssim = 0.3, lambda_fk = 0.1.

Only the loss weights change. Data pipeline, splits, masking, optimiser,
schedule and number of epochs are those of train.py. The torch seed is fixed
so that differences between configurations are not due to initialisation
(residual cuDNN non-determinism remains, of the order of a few hundredths
of a dB).

Usage
-----
    python scripts/train/train_v6_lossweights.py --lambda-ssim 0.3 --lambda-fk 0.1
    python scripts/train/train_v6_lossweights.py --lambda-ssim 0 --lambda-fk 0 --tag l1only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT

import argparse
from pathlib import Path

import numpy as np
import torch

from config import get_config
from train import train_within_geology


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda-l1", type=float, default=1.0)
    p.add_argument("--lambda-ssim", type=float, required=True)
    p.add_argument("--lambda-fk", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None,
                   help="override config.train.epochs (default: keep config)")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--root", type=str, default=str(CKPT_ROOT / "lossweights"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = get_config()
    config.train.lambda_l1 = args.lambda_l1
    config.train.lambda_ssim = args.lambda_ssim
    config.train.lambda_fk = args.lambda_fk
    if args.epochs is not None:
        config.train.epochs = args.epochs

    tag = args.tag or (f"l1{args.lambda_l1:g}_ssim{args.lambda_ssim:g}"
                       f"_fk{args.lambda_fk:g}").replace(".", "p")
    run_dir = Path(args.root) / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Each run writes checkpoints AND its summary into its own directory,
    # so nothing overwrites the files of the paper model.
    config.train.checkpoint_dir = str(run_dir)
    config.eval.output_dir = str(run_dir)

    print("=" * 62)
    print(f"LOSS-WEIGHT SENSITIVITY  [{tag}]")
    print(f"  lambda_l1   = {config.train.lambda_l1}")
    print(f"  lambda_ssim = {config.train.lambda_ssim}")
    print(f"  lambda_fk   = {config.train.lambda_fk}")
    print(f"  torch seed  = {args.seed}   epochs = {config.train.epochs}")
    print(f"  output dir  = {run_dir}")
    print("=" * 62)

    train_within_geology(config)


if __name__ == "__main__":
    main()
