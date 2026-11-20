"""
train_no_film.py — Train the vanilla U-Net (no FiLM conditioning) as a deep
learning baseline ("no FiLM" entry of Fig. S4).

Same training conditions as the model with FiLM:
  - all training lines of config.py, mixed (within-geology), contiguous
    70/15/15 split of each line
  - 200 epochs (default) with cosine annealing
  - composite loss L1 + SSIM + f-k
  - masking strategy and ratios of config.py

The only difference: no FiLM at the bottleneck. The geometry code is still
passed to the model, but it is ignored.

Output:
  CKPT_ROOT/no_film/best.pt
  RESULTS_ROOT/no_film/training_log.json
  RESULTS_ROOT/no_film/summary.json

Usage:
    python scripts/train/train_no_film.py --epochs 200
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT, RESULTS_ROOT

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from config import get_config
from jtr_marine.data.dataset import build_within_geology_datasets
from jtr_marine.losses.composite import build_loss
from train import train_one_epoch, validate
from jtr_marine.models.unet_no_film import build_model_no_film


def train_baseline_no_film(epochs: int = 200,
                            batch_size: int = 8,
                            num_workers: int = 8):
    """Training pipeline of the baseline without FiLM."""
    config = get_config()

    # Override the config with the function arguments
    config.train.epochs      = epochs
    config.train.batch_size  = batch_size
    config.train.num_workers = num_workers
    config.train.T_max       = epochs

    ckpt_dir    = Path(str(CKPT_ROOT / "no_film"))
    results_dir = Path(str(RESULTS_ROOT / "no_film"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"BASELINE NO-FILM TRAINING — Vanilla U-Net")
    print(f"{'='*60}")
    print(f"Device:      {device}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Epochs:      {epochs}")
    print(f"Batch size:  {batch_size}")
    print(f"{'='*60}")

    # Same data as the model with FiLM: within-geology mixing
    train_ds, val_ds, test_ds = build_within_geology_datasets(config)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Model WITHOUT FiLM
    model = build_model_no_film(config).to(device)
    print(f"\nModel parameters: {model.count_parameters():,} (no FiLM)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.T_max, eta_min=config.train.eta_min,
    )
    criterion = build_loss(config)

    best_snr = -float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=config.train.grad_clip,
        )
        val_metrics = validate(
            model, val_loader, criterion, device,
            dt=config.data.dt_ms * 1e-3,
        )
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        val_snr = val_metrics.get("snr_db_mean", 0.0)

        log = {
            "epoch": epoch, "lr": lr,
            "train_loss": train_metrics.get("loss_mean", 0.0),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(log)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  E{epoch:03d} | "
                  f"loss={log['train_loss']:.4f} | "
                  f"val_SNR={val_snr:.2f}dB | "
                  f"lr={lr:.2e} | {dt:.1f}s")

        if val_snr > best_snr:
            best_snr = val_snr
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_snr": best_snr,
                "model_type": "no_film",
            }, ckpt_dir / "best.pt")
        else:
            patience_counter += 1

        if patience_counter >= config.train.early_stopping_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    print(f"\n{'='*60}")
    print(f"RESULTS — Vanilla U-Net (no FiLM)")
    print(f"{'='*60}")
    print(f"  Best val SNR:  {best_snr:.2f} dB")

    summary = {
        "model_type":     "no_film",
        "epochs_run":     len(history),
        "best_val_snr":   best_snr,
        "model_params":   model.count_parameters(),
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(results_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {results_dir}/")
    print(f"Checkpoint saved to {ckpt_dir}/best.pt")
    print(f"Best validation SNR (no FiLM): {best_snr:.2f} dB")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    train_baseline_no_film(
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
