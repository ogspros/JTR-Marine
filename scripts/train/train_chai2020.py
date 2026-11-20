#!/usr/bin/env python3
"""
train_chai2020.py — Train the U-Net of Chai et al. (2020) as an independent
published baseline, under exactly the protocol of the proposed model
(Table S7).

What changes (method-specific, as published):
  - architecture: Chai2020Wrapper (jtr_marine/models/chai2020_unet.py),
    87.1 M parameters, verified against the public GeoBrain reproduction;
  - input: the masked gather only (no mask channel, no geometry conditioning);
  - loss: mean squared error over the recorded traces of the gather.

What does NOT change (project protocol, from train.py/train_within_geology):
  - training pool, contiguous 70/15/15 splits, masking strategies and ratios,
    preprocessing, number of epochs, AdamW + cosine schedule, gradient clipping,
    validation metric (SNR on missing traces at 30% random) and checkpoint
    selection, early stopping.

The learning rate is the only hyperparameter selected for this baseline: the
reproduction documents 1e-3 in its configuration but 1e-4 in its paper
alignment, so it is chosen on the validation set between two values, with the
same discipline used for the other baselines (run once per value).

The model is saved under "model_state" with keys "net.*", recognised by
jtr_marine/model_loader.py, so all evaluation scripts work unchanged.

Usage
-----
    python scripts/train/train_chai2020.py --lr 1e-4 --epochs 1   # timing / memory test
    python scripts/train/train_chai2020.py --lr 1e-4
    python scripts/train/train_chai2020.py --lr 3e-4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT

import argparse
from pathlib import Path

import numpy as np
import torch

import train as T
from config import get_config
from jtr_marine.models.chai2020_unet import Chai2020Wrapper


class MSEOnRecorded:
    """Mean squared error over the recorded traces (validity mask), i.e. the
    whole gather as in the published method, excluding the zero padding of the
    common grid."""

    def __call__(self, pred, target, trace_mask, validity):
        v = validity[:, None, None, :].to(pred.dtype)            # (B,1,1,O)
        se = ((pred - target) ** 2) * v
        loss = se.sum() / (v.sum() * pred.shape[-2]).clamp_min(1.0)
        return {"loss": loss, "mse": loss.detach()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=None, help="default: config")
    ap.add_argument("--batch-size", type=int, default=None, help="default: config")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=str(CKPT_ROOT / "chai2020"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = get_config()
    config.train.learning_rate = args.lr
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size

    run_dir = Path(args.root) / f"lr{args.lr:.0e}".replace("-0", "-")
    run_dir.mkdir(parents=True, exist_ok=True)
    config.train.checkpoint_dir = str(run_dir)
    config.eval.output_dir = str(run_dir)

    # swap only the model and the loss inside the project training routine
    T.build_model = lambda cfg: Chai2020Wrapper()
    T.build_loss = lambda cfg: MSEOnRecorded()

    print("=" * 62)
    print("INDEPENDENT BASELINE: U-Net of Chai et al. (2020)")
    print(f"  lr = {args.lr:g}   epochs = {config.train.epochs}   "
          f"batch = {config.train.batch_size}   seed = {args.seed}")
    print(f"  output dir = {run_dir}")
    print("=" * 62)

    T.train_within_geology(config)


if __name__ == "__main__":
    main()
