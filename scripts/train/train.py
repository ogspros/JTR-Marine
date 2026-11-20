"""
train.py — Training of the proposed model (compact U-Net with FiLM geometry
conditioning) for missing-trace reconstruction.

Modes:
  - --within-geology: all training lines mixed, contiguous 70/15/15 split of
    each line (the paper model)
  - --holdout-geometry L: leave-one-geometry-out
  - default, or --fold K: leave-one-line-out folds

Supports:
  - Composite loss (L1 + SSIM + f-k)
  - Early stopping and cosine LR scheduling
  - W&B logging (optional)

Usage:
    python scripts/train/train.py --within-geology
    python scripts/train/train.py --holdout-geometry 1500
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import json
import time
from typing import Dict, Optional

from config import get_config
from jtr_marine.models.unet import build_model
from jtr_marine.losses.composite import build_loss
from jtr_marine.data.dataset import build_fold_datasets
from jtr_marine.metrics import evaluate_batch, aggregate_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion,
                    device: torch.device,
                    grad_clip: float = 1.0) -> Dict[str, float]:
    model.train()
    epoch_losses = []

    for batch in loader:
        x          = batch["input"].to(device)       # (B, 2, T, O)
        target     = batch["target"].to(device)      # (B, 1, T, O)
        geometry   = batch["geometry"].to(device)    # (B,)
        trace_mask = batch["trace_mask"].to(device)  # (B, O)
        validity   = batch["validity"].to(device)    # (B, O)

        optimizer.zero_grad(set_to_none=True)

        pred = model(x, geometry)                    # (B, 1, T, O)
        loss_dict = criterion(pred, target, trace_mask, validity)
        loss = loss_dict["loss"]

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        epoch_losses.append({k: v.item() if hasattr(v, 'item') else v
                              for k, v in loss_dict.items()})

    return aggregate_metrics(epoch_losses)


# ─────────────────────────────────────────────────────────────────────────────
# Validation step
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             criterion,
             device: torch.device,
             dt: float = 0.002,
             dx: float = 12.5) -> Dict[str, float]:
    """
    Internal training metric (used for model selection and for the "Test SNR"
    printed at the end of training). NOTE: it is NOT the SNR_miss of the
    paper tables:
      * masks come from the dataset of the loader (for test sets: "mixed"
        strategy, ratios 20-50 %), not 30 % random;
      * metrics are computed on the raw prediction and averaged per batch,
        not per gather.
    The paper metrics are computed by scripts/evaluate/ (e.g.
    recompute_with_per_gather_lists.py, eval_geometry_matrix.py).
    """
    model.eval()
    all_metrics = []

    for batch in loader:
        x          = batch["input"].to(device)
        target     = batch["target"].to(device)
        geometry   = batch["geometry"].to(device)
        trace_mask = batch["trace_mask"].to(device)
        validity   = batch["validity"].to(device)

        pred = model(x, geometry)

        metrics = evaluate_batch(pred, target, trace_mask, validity,
                                 dt=dt, dx=dx)
        all_metrics.append(metrics)

    return aggregate_metrics(all_metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Single fold training
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(config,
               fold_idx: int,
               device: torch.device,
               use_wandb: bool = False) -> Dict[str, float]:
    """
    Train model for one leave-one-LINE-out fold.

    Args:
        fold_idx: index into config.data.dataset_paths of the line held out
                  (six-line configuration: 0 = SISOC-MEPA, 1 = CG-1,
                  2 = IT90-65). It is not a whole geometry: see
                  --holdout-geometry for leave-one-geometry-out.
    Returns:
        dict of final test metrics (internal metric of validate(), on the
        held-out line)
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}/{config.train.n_folds} — held-out line, streamer: "
          f"{config.data.streamer_lengths[fold_idx]}m streamer")
    print(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dataset, val_dataset, test_dataset = build_fold_datasets(
        config, test_fold_idx=fold_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    print(f"Train: {len(train_dataset)} gathers | "
          f"Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ── Model, optimizer, scheduler ──────────────────────────────────────────
    model = build_model(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.T_max,
        eta_min=config.train.eta_min,
    )
    criterion = build_loss(config)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ckpt_dir = Path(config.train.checkpoint_dir) / f"fold{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_snr = -float("inf")
    patience_counter = 0
    history = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, config.train.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=config.train.grad_clip
        )
        val_metrics = validate(
            model, val_loader, criterion, device,
            dt=config.data.dt_ms * 1e-3,
        )
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        val_snr = val_metrics.get("snr_db_mean", 0.0)

        # Log
        log = {
            "epoch": epoch,
            "lr":    lr,
            "train_loss": train_metrics.get("loss_mean", 0.0),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(log)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  E{epoch:03d} | "
                  f"loss={log['train_loss']:.4f} | "
                  f"val_SNR={val_snr:.2f}dB | "
                  f"lr={lr:.2e} | {dt:.1f}s")

        if use_wandb:
            import wandb
            wandb.log({f"fold{fold_idx}/{k}": v for k, v in log.items()})

        # Checkpoint: best model by val SNR
        if val_snr > best_snr:
            best_snr = val_snr
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_snr":     best_snr,
            }, ckpt_dir / "best.pt")
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % config.train.save_every_n_epochs == 0:
            torch.save({"epoch": epoch, "model_state": model.state_dict()},
                       ckpt_dir / f"epoch_{epoch:03d}.pt")

        # Early stopping
        if patience_counter >= config.train.early_stopping_patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no improvement for {patience_counter} epochs)")
            break

    # ── Test evaluation with best model ──────────────────────────────────────
    print(f"\n  Loading best checkpoint (val SNR = {best_snr:.2f} dB)...")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = validate(
        model, test_loader, criterion, device,
        dt=config.data.dt_ms * 1e-3,
    )
    print(f"  Test SNR  = {test_metrics.get('snr_db_mean', 0):.2f} ±"
          f" {test_metrics.get('snr_db_std', 0):.2f} dB")
    print(f"  Test SSIM = {test_metrics.get('ssim_mean', 0):.4f}")
    print(f"  Test FK   = {test_metrics.get('fk_energy_ratio_mean', 0):.4f}")

    # Save history
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {f"fold{fold_idx}_{k}": v for k, v in test_metrics.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Full cross-geometry experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_geometry_experiment(config=None, use_wandb: bool = False):
    """
    Run the config.train.n_folds leave-one-line-out folds (see train_fold)
    and report the aggregated results (internal metric of validate()).
    """
    if config is None:
        config = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if use_wandb:
        import wandb
        wandb.init(project="seismic-recon", name=config.experiment_name)

    all_results = {}

    for fold_idx in range(config.train.n_folds):
        fold_results = train_fold(config, fold_idx, device, use_wandb)
        all_results.update(fold_results)

    # ── Aggregate across folds ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Leave-one-line-out results (internal validation metric, mean across folds):")
    print(f"{'='*60}")

    # Group by metric
    metrics_by_name: Dict[str, list] = {}
    for key, val in all_results.items():
        # key format: fold0_snr_db_mean
        parts = key.split("_", 1)
        metric_name = parts[1] if len(parts) > 1 else key
        metrics_by_name.setdefault(metric_name, []).append(val)

    summary = {}
    for metric, values in metrics_by_name.items():
        if metric.endswith("_mean"):
            base = metric[:-5]
            summary[base] = {
                "mean": float(np.mean(values)),
                "std":  float(np.std(values)),
                "per_fold": values,
            }
            print(f"  {base:25s}: {summary[base]['mean']:.4f} ± {summary[base]['std']:.4f}")

    # Save summary
    results_dir = Path(config.eval.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "cross_geometry_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_dir / 'cross_geometry_summary.json'}")

    if use_wandb:
        import wandb
        wandb.log({"summary": summary})
        wandb.finish()

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Within-geology training (the paper model; alternative to the folds above)
# ─────────────────────────────────────────────────────────────────────────────

def train_within_geology(config=None, use_wandb: bool = False,
                         exclude_streamer_lengths=None):
    """
    Within-geology setup: all training lines mixed, contiguous 70/15/15 split
    of each line (the paper model).

    exclude_streamer_lengths (optional, default None = paper model): skip
    every line with these streamer lengths (leave-one-geometry-out, see
    --holdout-geometry).
    """
    from jtr_marine.data.dataset import build_within_geology_datasets

    if config is None:
        config = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print(f"WITHIN-GEOLOGY EXPERIMENT")
    print(f"All training lines mixed, contiguous 70/15/15 split of each line")
    print(f"{'='*60}\n")

    if use_wandb:
        import wandb
        wandb.init(project="seismic-recon", name=config.experiment_name + "_within")

    # ── Data ──────────────────────────────────────────────────────────────
    if exclude_streamer_lengths:
        print(f"Held-out geometry (lines excluded): {exclude_streamer_lengths} m\n")
    train_dataset, val_dataset, test_dataset = build_within_geology_datasets(
        config, exclude_streamer_lengths=exclude_streamer_lengths)

    print(f"\nTotals: train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch_size,
        shuffle=True, num_workers=config.train.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.train.batch_size,
        shuffle=False, num_workers=config.train.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.train.batch_size,
        shuffle=False, num_workers=config.train.num_workers,
    )

    # ── Model, optimizer, scheduler ──────────────────────────────────────
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
    criterion = build_loss(config)

    ckpt_dir = Path(config.train.checkpoint_dir) / "within_geology"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_snr = -float("inf")
    patience_counter = 0
    history = []

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(1, config.train.epochs + 1):
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

        if use_wandb:
            import wandb
            wandb.log({f"within/{k}": v for k, v in log.items()})

        if val_snr > best_snr:
            best_snr = val_snr
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(), "val_snr": best_snr,
            }, ckpt_dir / "best.pt")
        else:
            patience_counter += 1

        if patience_counter >= config.train.early_stopping_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── Test evaluation ──────────────────────────────────────────────────
    print(f"\n  Loading best checkpoint (val SNR = {best_snr:.2f} dB)...")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = validate(model, test_loader, criterion, device,
                            dt=config.data.dt_ms * 1e-3)
    print(f"\n{'='*60}")
    print("WITHIN-GEOLOGY RESULTS")
    print(f"{'='*60}")
    print(f"  Test SNR  = {test_metrics.get('snr_db_mean', 0):.2f} ±"
          f" {test_metrics.get('snr_db_std', 0):.2f} dB")
    print(f"  Test SSIM = {test_metrics.get('ssim_mean', 0):.4f}")
    print(f"  Test FK   = {test_metrics.get('fk_energy_ratio_mean', 0):.4f}")

    import json
    results_dir = Path(config.eval.output_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    with open(results_dir / "within_geology_summary.json", "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if use_wandb:
        import wandb
        wandb.finish()

    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train seismic reconstruction U-Net")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run a single leave-one-line-out fold (index of the held-out "
                             "line). If not given, run all folds.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of DataLoader workers (overrides config)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Index of the GPU to use (default: 0)")
    parser.add_argument("--within-geology", action="store_true",
                        help="Use the within-geology setup (the paper model) instead "
                             "of the leave-one-line-out folds")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--holdout-geometry", type=float, default=None,
                        choices=[1500.0, 3000.0, 4500.0],
                        help="leave-one-GEOMETRY-out: within-geology training "
                             "without all lines of this streamer length "
                             "(default: off). Checkpoints in "
                             "CKPT_ROOT/logo_<L>m/within_geology/")
    parser.add_argument("--seed", type=int, default=None,
                        help="torch/numpy seed for weight init, dropout and "
                             "shuffling (default: not set, as for the paper model)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    config = get_config()

    # Override the config with the command-line arguments, if given
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers

    # Print a summary of the configuration
    print(f"\n{'='*60}")
    print(f"Training configuration:")
    print(f"  Epochs:      {config.train.epochs}")
    print(f"  Batch size:  {config.train.batch_size}")
    print(f"  Num workers: {config.train.num_workers}")
    print(f"  GPU:         {args.gpu} ({torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Fold:        {args.fold if args.fold is not None else ('-' if (args.within_geology or args.holdout_geometry) else 'all')}") 
    print(f"{'='*60}\n")

    if args.holdout_geometry is not None:
        from config import CKPT_ROOT, RESULTS_ROOT
        tag = f"logo_{int(args.holdout_geometry)}m"
        config.train.checkpoint_dir = str(CKPT_ROOT / tag)
        config.eval.output_dir = str(RESULTS_ROOT / tag)
        train_within_geology(config, use_wandb=args.wandb,
                             exclude_streamer_lengths=[args.holdout_geometry])
    elif args.within_geology:
        train_within_geology(config, use_wandb=args.wandb)
    elif args.fold is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_fold(config, args.fold, device, use_wandb=args.wandb)
    else:
        run_cross_geometry_experiment(config, use_wandb=args.wandb)
