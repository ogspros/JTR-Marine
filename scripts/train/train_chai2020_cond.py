#!/usr/bin/env python3
"""
train_chai2020_cond.py — Chai et al. (2020) backbone with the conditioning and
the training objective of the proposed model (variants of Table S7).

What changes with respect to train_chai2020.py:
  - input: masked gather + binary mask (2 channels), as the proposed model;
  - FiLM geometry conditioning at the bottleneck (unless --no-film);
  - loss: the project composite loss, exactly as built by train.build_loss
    from config (same lambda_SSIM and lambda_fk as the proposed model).

What does NOT change: data pool, splits, masking, preprocessing, epochs,
optimiser and schedule, gradient clipping, validation metric, checkpoint
selection, early stopping (all from train.train_within_geology).

Learning rate: default 3e-4, the value selected on validation for the Chai
backbone in train_chai2020.py (not re-tuned here).

Usage
-----
    python scripts/train/train_chai2020_cond.py --epochs 1               # smoke test
    python scripts/train/train_chai2020_cond.py                          # full: mask + FiLM
    python scripts/train/train_chai2020_cond.py --no-film                # ablation: mask only
    python scripts/train/train_chai2020_cond.py --no-film --no-mask      # ablation: loss only
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
from jtr_marine.models.chai2020_cond import Chai2020CondWrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-film", action="store_true")
    ap.add_argument("--no-mask", action="store_true",
                    help="input = masked gather only (1 channel), as in Chai et al.")
    ap.add_argument("--epochs", type=int, default=None, help="default: config")
    ap.add_argument("--batch-size", type=int, default=None, help="default: config")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=str(CKPT_ROOT / "chai2020_cond"))
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

    use_film = not args.no_film
    in_ch = 1 if args.no_mask else 2
    if args.no_mask and use_film:
        raise SystemExit("--no-mask is meant with --no-film (loss-only ablation)")
    if args.no_mask:
        base = "loss_only"
    else:
        base = "mask_film" if use_film else "mask_only"
    tag = base + f"_lr{args.lr:.0e}".replace("-0", "-")
    run_dir = Path(args.root) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    config.train.checkpoint_dir = str(run_dir)
    config.eval.output_dir = str(run_dir)

    film_hidden = config.model.film_hidden_dim
    # swap only the model; the loss stays the project one (T.build_loss)
    T.build_model = lambda cfg: Chai2020CondWrapper(
        in_channels=in_ch, use_film=use_film, film_hidden_dim=film_hidden)

    print("=" * 62)
    print("CHAI (2020) BACKBONE + PROJECT CONDITIONING AND LOSS")
    inp = "gather only" if in_ch == 1 else "gather + mask"
    print(f"  input = {inp}   FiLM = {use_film}   loss = project composite")
    print(f"  lr = {args.lr:g}   epochs = {config.train.epochs}   "
          f"batch = {config.train.batch_size}   seed = {args.seed}")
    print(f"  output dir = {run_dir}")
    print("=" * 62)

    T.train_within_geology(config)


if __name__ == "__main__":
    main()
