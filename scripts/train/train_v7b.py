"""
train_v7b.py — Training variant v7b: Charbonnier loss (in place of L1), with
optional curriculum learning on the masking ratio ("+ Charbonnier" and
"+ Charbonnier + curriculum" entries of Fig. S4).

No horizontal flip augmentation: the shot gathers are asymmetric
(single-sided), so the flip is not geophysically valid.

Same setup as the proposed model: within-geology, 6 lines, contiguous
70/15/15 split.

Experiments:
  python scripts/train/train_v7b.py                  # baseline: Charbonnier only
  python scripts/train/train_v7b.py --curriculum     # full:     Charbonnier + curriculum

Output:
  without --curriculum: CKPT_ROOT/v7b_baseline/best.pt, RESULTS_ROOT/v7b_baseline/summary.json
  with --curriculum:    CKPT_ROOT/v7b_full/best.pt,     RESULTS_ROOT/v7b_full/summary.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT, RESULTS_ROOT

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from config import get_config
from jtr_marine.data.dataset import SeismicGatherDataset, SharedGatherDataset
from jtr_marine.models.unet import build_model
from jtr_marine.losses.composite_v7 import build_loss_v7
from train import train_one_epoch, validate


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum learning schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_masking_ratios_for_epoch(epoch: int, total_epochs: int):
    """Return the list of masking ratios to sample from at this epoch."""
    progress = epoch / total_epochs
    if progress < 0.34:
        return [0.15, 0.20, 0.25]   # easy
    elif progress < 0.67:
        return [0.20, 0.30, 0.35]   # medium
    else:
        return [0.25, 0.35, 0.45]   # hard


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder — NO FLIP
# ─────────────────────────────────────────────────────────────────────────────

def build_within_geology_datasets_v7b(config):
    """
    Same behaviour as build_within_geology_datasets (jtr_marine/data/dataset.py)
    but with training=True on the train sets (to support set_masking_ratios)
    and enable_flip=False (no flip: the gathers are asymmetric, single-sided).

    Returns: train_sets (a list, to control the ratios), val_ds, test_ds.
    """
    dc = config.data
    n_t = int(round(dc.t_max_s * 1000.0 / dc.dt_target_ms))
    n_x = int(round(dc.x_max_m / dc.dx_target_m)) + 1

    train_sets, val_sets, test_sets = [], [], []
    train_frac = 0.70
    val_frac   = 0.15

    print(f"\nLoading {len(dc.dataset_paths)} datasets (v7b training)...")

    for i, (path, length) in enumerate(zip(dc.dataset_paths, dc.streamer_lengths)):
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

        # CONTIGUOUS 70/15/15 split
        all_idx   = np.arange(n_total, dtype=np.int64)
        n_train_g = int(n_total * train_frac)
        n_val_g   = int(n_total * val_frac)

        train_idx = all_idx[:n_train_g]
        val_idx   = all_idx[n_train_g:n_train_g + n_val_g]
        test_idx  = all_idx[n_train_g + n_val_g:]

        geom_code = length / dc.max_streamer_length

        shared_kwargs = dict(
            n_time         = n_t,
            n_offsets      = n_x,
            block_size     = dc.block_size,
            random_seed    = dc.random_seed,
            geometry_code  = geom_code,
        )

        # ───── train set: training=True (for the curriculum), NO flip
        train_sets.append(SharedGatherDataset(
            gathers          = all_gathers,
            indices          = train_idx,
            masking_strategy = dc.masking_strategy,
            masking_ratios   = list(dc.masking_ratios),
            training         = True,
            enable_flip      = False,    # flip disabled
            **shared_kwargs,
        ))
        val_sets.append(SharedGatherDataset(
            gathers          = all_gathers,
            indices          = val_idx,
            masking_strategy = dc.masking_strategy,
            masking_ratios   = list(dc.masking_ratios),
            training         = False,
            enable_flip      = False,
            **shared_kwargs,
        ))
        test_sets.append(SharedGatherDataset(
            gathers          = all_gathers,
            indices          = test_idx,
            masking_strategy = "mixed",
            masking_ratios   = list(dc.masking_ratios),
            training         = False,
            enable_flip      = False,
            **shared_kwargs,
        ))

        print(f"  {Path(path).name} ({length:.0f}m): "
              f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    n_train_tot = sum(len(s) for s in train_sets)
    n_val_tot   = sum(len(s) for s in val_sets)
    n_test_tot  = sum(len(s) for s in test_sets)
    print(f"\nTotals: train={n_train_tot}, val={n_val_tot}, test={n_test_tot}")
    print(f"Flip augmentation: OFF (asymmetric single-sided gathers)\n")

    return train_sets, ConcatDataset(val_sets), ConcatDataset(test_sets)


# ─────────────────────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_v7b(use_curriculum: bool,
              epochs: int,
              batch_size: int,
              num_workers: int):
    config = get_config()
    config.train.epochs      = epochs
    config.train.batch_size  = batch_size
    config.train.num_workers = num_workers
    config.train.T_max       = epochs

    suffix      = "v7b_full" if use_curriculum else "v7b_baseline"
    ckpt_dir    = Path(str(CKPT_ROOT / f"{suffix}"))
    results_dir = Path(str(RESULTS_ROOT / f"{suffix}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*70}")
    print(f"TRAINING V7b — {suffix}")
    print(f"{'='*70}")
    print(f"Device:        {device}")
    print(f"Checkpoints:   {ckpt_dir}")
    print(f"Results:       {results_dir}")
    print(f"Epochs:        {epochs}")
    print(f"Batch size:    {batch_size}")
    print(f"Flip augm:     OFF (asymmetric gathers)")
    print(f"Charbonnier:   ON")
    print(f"Curriculum:    {'ON' if use_curriculum else 'OFF'}")
    print(f"{'='*70}\n")

    train_sets, val_ds, test_ds = build_within_geology_datasets_v7b(config)
    train_concat = ConcatDataset(train_sets)

    train_loader = DataLoader(train_concat, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    model = build_model(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.T_max, eta_min=config.train.eta_min,
    )
    criterion = build_loss_v7(config)
    print(f"Loss: {type(criterion).__name__}")
    print(f"Initial masking ratios: {train_sets[0].masking_ratios}\n")

    best_snr = -float("inf")
    history  = []
    curriculum_log = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # CURRICULUM: update the masking ratios before every epoch
        if use_curriculum:
            new_ratios = get_masking_ratios_for_epoch(epoch, epochs)
            for ts in train_sets:
                ts.set_masking_ratios(new_ratios)
            # Log the phase changes only
            if not curriculum_log or new_ratios != curriculum_log[-1]["ratios"]:
                curriculum_log.append({"epoch": epoch, "ratios": new_ratios})
                print(f"  [curriculum] epoch {epoch}: ratios -> {new_ratios}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=config.train.grad_clip,
        )
        val_metrics = validate(
            model, val_loader, criterion, device,
            dt=config.data.dt_ms * 1e-3,
        )
        scheduler.step()

        dt_epoch = time.time() - t0
        lr       = optimizer.param_groups[0]["lr"]
        val_snr  = val_metrics.get("snr_db_mean", 0.0)

        log = {
            "epoch": epoch, "lr": lr,
            "ratios": train_sets[0].masking_ratios if use_curriculum else list(config.data.masking_ratios),
            "train_loss": train_metrics.get("loss_mean", 0.0),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(log)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            curr_str = f" | ratios={train_sets[0].masking_ratios}" if use_curriculum else ""
            print(f"  E{epoch:03d} | loss={log['train_loss']:.4f} | "
                  f"val_SNR={val_snr:.2f}dB | lr={lr:.2e} | {dt_epoch:.1f}s{curr_str}")

        if val_snr > best_snr:
            best_snr = val_snr
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(), "val_snr": best_snr,
                "use_curriculum": use_curriculum,
            }, ckpt_dir / "best.pt")

    print(f"\n  Loading best checkpoint (val SNR = {best_snr:.2f} dB)...")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = validate(model, test_loader, criterion, device,
                            dt=config.data.dt_ms * 1e-3)

    print(f"\n{'='*70}")
    print(f"RESULTS V7b — {suffix}")
    print(f"{'='*70}")
    print(f"  Best val SNR:  {best_snr:.2f} dB")
    print(f"  Test SNR  = {test_metrics.get('snr_db_mean', 0):.2f} +- "
          f"{test_metrics.get('snr_db_std', 0):.2f} dB")
    print(f"  Test SSIM = {test_metrics.get('ssim_mean', 0):.4f}")
    print(f"  Test FK   = {test_metrics.get('fk_energy_ratio_mean', 0):.4f}")
    print(f"{'='*70}")

    summary = {
        "variant":         suffix,
        "use_curriculum":  use_curriculum,
        "epochs_run":      len(history),
        "best_val_snr":    best_snr,
        "test_metrics":    {k: float(v) for k, v in test_metrics.items()},
    }
    if use_curriculum:
        summary["curriculum_log"] = curriculum_log

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {results_dir}/")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning on the masking ratio")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    train_v7b(
        use_curriculum = args.curriculum,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
    )
