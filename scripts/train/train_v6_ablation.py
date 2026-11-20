#!/usr/bin/env python3
"""
train_v6_ablation.py — Proposed model (v6) with a different kernel size and/or
no normalisation, under exactly the protocol of v6 (same loss, learning rate,
data, splits, masking, epochs, schedule, validation and checkpoint selection).

Purpose: explain the gap between v6 and the U-Net of Chai et al. (2020), which
differs from v6 mainly in width, 5x5 kernels and absence of normalisation.

Usage
-----
    python scripts/train/train_v6_ablation.py --kernel 5 --norm instance --epochs 1   # smoke test
    python scripts/train/train_v6_ablation.py --kernel 5 --norm instance              # 5x5 only
    python scripts/train/train_v6_ablation.py --kernel 3 --norm none                  # no norm only
    python scripts/train/train_v6_ablation.py --kernel 5 --norm none                  # both

Note: 5x5 kernels also increase the number of parameters (about x2.7); the
kernel effect is therefore not separated from the capacity effect.
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
from jtr_marine.models.unet_ablation import build_ablation_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", type=int, choices=[3, 5, 7], required=True)
    ap.add_argument("--norm", choices=["instance", "group", "batch", "none"], required=True)
    ap.add_argument("--epochs", type=int, default=None, help="default: config")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=str(CKPT_ROOT / "v6_ablation"))
    args = ap.parse_args()

    if args.kernel == 3 and args.norm == "instance":
        raise SystemExit("kernel 3 + instance norm is v6 itself: nothing to ablate")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = get_config()
    if args.epochs is not None:
        config.train.epochs = args.epochs

    run_dir = Path(args.root) / f"k{args.kernel}_{args.norm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.train.checkpoint_dir = str(run_dir)
    config.eval.output_dir = str(run_dir)

    # swap only the model; loss, lr and everything else stay as for v6
    T.build_model = lambda cfg: build_ablation_model(cfg, args.kernel, args.norm)

    print("=" * 62)
    print("v6 ABLATION")
    print(f"  kernel = {args.kernel}x{args.kernel}   norm = {args.norm}")
    print(f"  lr = {config.train.learning_rate:g}   epochs = {config.train.epochs}   "
          f"batch = {config.train.batch_size}   seed = {args.seed}")
    print(f"  output dir = {run_dir}")
    print("=" * 62)

    T.train_within_geology(config)


if __name__ == "__main__":
    main()
