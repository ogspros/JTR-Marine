"""
train_single_geometry_all.py — Train a model on ONE geometry only, using ALL
the lines available for that geometry (single-geometry models of the
joint-vs-single-geometry comparison).

All the lines of the target geometry in config.py are used
(MEPA + GC01 for 1500 m, IT065 + IT89_33 for 3000 m, M31 + M32 for 4500 m),
so that the comparison with multi-geometry training is fair:
  - multi-geometry training: 6 lines, 3 different geometries
  - single-geometry training: 2 lines, 1 geometry

Output:
  CKPT_ROOT/single_<streamer>m_all/best.pt
  RESULTS_ROOT/single_<streamer>m_all/training_log.json
  RESULTS_ROOT/single_<streamer>m_all/summary.json

Usage:
    python scripts/train/train_single_geometry_all.py --streamer 1500.0 --epochs 150
    python scripts/train/train_single_geometry_all.py --streamer 3000.0 --epochs 150
    python scripts/train/train_single_geometry_all.py --streamer 4500.0 --epochs 150

Note: the "Test SNR" printed at the end is the internal metric of
train.validate() ("mixed" masking 20-50 %, averaged per batch), not the
SNR_miss at 30 % random of the paper tables; evaluate the checkpoints with
scripts/evaluate/eval_geometry_matrix.py for numbers comparable with the
multi-geometry model.
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
from torch.utils.data import DataLoader, ConcatDataset

from config import get_config
from jtr_marine.data.dataset import SeismicGatherDataset, SharedGatherDataset
from jtr_marine.models.unet import build_model
from jtr_marine.losses.composite import build_loss
from train import train_one_epoch, validate


def build_single_geometry_datasets_all(config, streamer_length: float):
    """
    Build train/val/test on ALL the lines of the target geometry.
    CONTIGUOUS 70/15/15 split of each line, then concatenation.
    """
    dc = config.data
    n_t = int(round(dc.t_max_s * 1000.0 / dc.dt_target_ms))
    n_x = int(round(dc.x_max_m / dc.dx_target_m)) + 1

    # Find ALL the lines of this geometry
    matching_datasets = []
    for i, length in enumerate(dc.streamer_lengths):
        if abs(length - streamer_length) < 1.0:
            matching_datasets.append((i, dc.dataset_paths[i], length))

    if not matching_datasets:
        raise ValueError(
            f"Streamer {streamer_length}m not found. "
            f"Available: {dc.streamer_lengths}"
        )

    print(f"\n  Datasets found for streamer {streamer_length:.0f}m: "
          f"{len(matching_datasets)}")
    for i, path, length in matching_datasets:
        print(f"    [{i}] {Path(path).name}")

    train_sets, val_sets, test_sets = [], [], []
    geom_code = streamer_length / dc.max_streamer_length

    for geo_idx, path, length in matching_datasets:
        print(f"\n  Loading {Path(path).name}...")

        full_ds = SeismicGatherDataset(
            data_path           = path,
            streamer_length     = length,
            max_streamer_length = dc.max_streamer_length,
            dt_target_ms        = dc.dt_target_ms,
            t_max_s             = dc.t_max_s,
            x_max_m             = dc.x_max_m,
            dx_target_m         = dc.dx_target_m,
            masking_ratios      = dc.masking_ratios,
            masking_strategy    = dc.masking_strategy,
            block_size          = dc.block_size,
            clip_sigma          = dc.clip_sigma,
            normalize_per_trace = dc.normalize_per_trace,
            apply_filter        = dc.apply_filter,
            filter_f_cut        = dc.filter_f_cut,
            val_fraction        = dc.val_fraction,
            random_seed         = dc.random_seed,
            split               = "train",
        )

        all_gathers = full_ds.gathers
        n_total     = len(all_gathers)

        # CONTIGUOUS 70/15/15 split (no spatial leakage)
        all_idx   = np.arange(n_total, dtype=np.int64)
        n_train_g = int(n_total * 0.70)
        n_val_g   = int(n_total * 0.15)

        train_idx = all_idx[:n_train_g]
        val_idx   = all_idx[n_train_g:n_train_g + n_val_g]
        test_idx  = all_idx[n_train_g + n_val_g:]

        shared_kwargs = dict(
            n_time         = n_t,
            n_offsets      = n_x,
            masking_ratios = dc.masking_ratios,
            block_size     = dc.block_size,
            random_seed    = dc.random_seed,
            geometry_code  = geom_code,
        )

        train_sets.append(SharedGatherDataset(
            gathers=all_gathers, indices=train_idx,
            masking_strategy=dc.masking_strategy, **shared_kwargs,
        ))
        val_sets.append(SharedGatherDataset(
            gathers=all_gathers, indices=val_idx,
            masking_strategy=dc.masking_strategy, **shared_kwargs,
        ))
        test_sets.append(SharedGatherDataset(
            gathers=all_gathers, indices=test_idx,
            masking_strategy="mixed", **shared_kwargs,
        ))

        print(f"    Split: train={len(train_idx)}, val={len(val_idx)}, "
              f"test={len(test_idx)}")

    n_train_tot = sum(len(s) for s in train_sets)
    n_val_tot   = sum(len(s) for s in val_sets)
    n_test_tot  = sum(len(s) for s in test_sets)
    print(f"\n  Totals: train={n_train_tot}, val={n_val_tot}, test={n_test_tot}")

    return (
        ConcatDataset(train_sets),
        ConcatDataset(val_sets),
        ConcatDataset(test_sets),
    )


def train_single_geometry_all(streamer_length: float,
                               epochs: int = 150,
                               batch_size: int = 8,
                               num_workers: int = 8):
    """
    Training pipeline on ALL the lines of the target geometry.
    """
    config = get_config()

    # Override the config with the function arguments
    config.train.epochs      = epochs
    config.train.batch_size  = batch_size
    config.train.num_workers = num_workers
    config.train.T_max       = epochs

    streamer_int = int(streamer_length)
    ckpt_dir     = Path(str(CKPT_ROOT / f"single_{streamer_int}m_all"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir  = Path(str(RESULTS_ROOT / f"single_{streamer_int}m_all"))
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"SINGLE-GEOMETRY ALL-DATASETS TRAINING — {streamer_int}m")
    print(f"{'='*60}")
    print(f"Device:      {device}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Results:     {results_dir}")
    print(f"Epochs:      {epochs}")
    print(f"Batch size:  {batch_size}")
    print(f"{'='*60}")

    train_ds, val_ds, test_ds = build_single_geometry_datasets_all(
        config, streamer_length,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    model = build_model(config).to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")

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
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(), "val_snr": best_snr,
                "streamer_length": streamer_length,
            }, ckpt_dir / "best.pt")
        else:
            patience_counter += 1

        if patience_counter >= config.train.early_stopping_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    print(f"\n  Loading best checkpoint (val SNR = {best_snr:.2f} dB)...")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device,
                      weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = validate(model, test_loader, criterion, device,
                            dt=config.data.dt_ms * 1e-3)

    print(f"\n{'='*60}")
    print(f"RESULTS — Single geometry ALL DATASETS {streamer_int}m")
    print(f"{'='*60}")
    print(f"  Best val SNR:  {best_snr:.2f} dB")
    print(f"  Test SNR:      {test_metrics.get('snr_db_mean', 0):.2f} +- "
          f"{test_metrics.get('snr_db_std', 0):.2f} dB")
    print(f"  Test SSIM:     {test_metrics.get('ssim_mean', 0):.4f}")
    print(f"  Test FK:       {test_metrics.get('fk_energy_ratio_mean', 0):.4f}")

    summary = {
        "streamer_length": streamer_length,
        "epochs_run":      len(history),
        "best_val_snr":    best_snr,
        "test_metrics":    {k: float(v) for k, v in test_metrics.items()},
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(results_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {results_dir}/")
    print(f"Checkpoint saved to {ckpt_dir}/best.pt")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--streamer", type=float, required=True,
                        choices=[1500.0, 3000.0, 4500.0],
                        help="Streamer length to train on (1500/3000/4500)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    train_single_geometry_all(
        streamer_length = args.streamer,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
    )
