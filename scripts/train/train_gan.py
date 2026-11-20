#!/usr/bin/env python3
"""
train_gan.py — Conditional-GAN baseline (Table 9).

Purpose
-------
Provide a published-style deep learning baseline to compare against the proposed
U-Net + FiLM model, following the conditional-GAN approach adopted for seismic
interpolation (e.g. Kaur et al., 2021). This is a re-implementation in the style
of that family of methods, not a reproduction of a specific published codebase.

Design
------
The generator is exactly the compact U-Net (SeismicUNet with FiLM conditioning), so
the only difference with respect to the proposed method is the adversarial
training signal. This isolates the training paradigm rather than confounding it
with an architectural change. FiLM is active in both, so the comparison is not
biased by conditioning.

  G loss = composite loss (L1 + SSIM + f-k)  +  lambda_adv * adversarial
  D loss = LSGAN least-squares loss on (condition, real) vs (condition, fake)

The discriminator is a conditional PatchGAN: it sees the masked input together
with either the ground truth or the reconstruction, so it judges consistency
with the observed traces, not just realism.

Everything else — data, contiguous splits, masking, optimiser settings, epochs,
validation metric, model selection — is identical to train.py.

The checkpoint is saved with the generator weights under the key "model_state",
so every evaluation script (recompute_with_per_gather_lists.py,
run_cross_survey_test_v6_final.py, ...) loads it unchanged.

Usage
-----
    python scripts/train/train_gan.py --epochs 150 --seed 42
    python scripts/train/train_gan.py --epochs 150 --seed 43 --lambda-adv 0.005
    python scripts/train/train_gan.py --smoke          # 2 epochs, quick sanity check
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_config
from jtr_marine.data.dataset import build_within_geology_datasets
from jtr_marine.models.unet import build_model
from jtr_marine.losses.composite import build_loss
from train import train_one_epoch, validate   # reused, unchanged


# ─────────────────────────────────────────────────────────────────────────────
# Conditional PatchGAN discriminator
# ─────────────────────────────────────────────────────────────────────────────
class PatchDiscriminator(nn.Module):
    """
    Conditional PatchGAN. Outputs a map of real/fake scores over overlapping
    patches rather than a single scalar, which is the standard choice for
    image-to-image problems and keeps the receptive field local.

    Input: concatenation of the conditioning tensor (the masked gather, 2 ch)
    and the gather being judged (real or reconstructed, 1 ch) -> 3 channels.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 n_layers: int = 3, norm_type: str = "instance"):
        super().__init__()

        def norm(c):
            if norm_type == "instance":
                return nn.InstanceNorm2d(c, affine=True)
            if norm_type == "batch":
                return nn.BatchNorm2d(c)
            return nn.Identity()

        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        c_in = base_channels
        for i in range(1, n_layers):
            c_out = min(base_channels * (2 ** i), 512)
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),
                norm(c_out),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            c_in = c_out

        c_out = min(base_channels * (2 ** n_layers), 512)
        layers += [
            nn.Conv2d(c_in, c_out, kernel_size=4, stride=1, padding=1, bias=False),
            norm(c_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_out, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, condition, sample):
        return self.net(torch.cat([condition, sample], dim=1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_gan(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Device: {device}")
    print(f"{'=' * 62}")
    print("CONDITIONAL-GAN BASELINE  (generator = v6 U-Net + FiLM)")
    print(f"seed={args.seed}  lambda_adv={args.lambda_adv}  epochs={args.epochs}")
    print(f"{'=' * 62}\n")

    # ── Data: identical to train.py ──────────────────────────────────────
    train_ds, val_ds, test_ds = build_within_geology_datasets(config)
    print(f"\nTotals: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=config.train.batch_size, shuffle=True,
        num_workers=config.train.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.train.batch_size, shuffle=False,
        num_workers=config.train.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.train.batch_size, shuffle=False,
        num_workers=config.train.num_workers,
    )

    # ── Generator: the compact U-Net, unchanged ───────────────────────────────
    netG = build_model(config).to(device)
    print(f"Generator parameters:     {netG.count_parameters():,}")

    netD = PatchDiscriminator(
        in_channels=config.model.in_channels + config.model.out_channels,
        base_channels=args.d_channels,
        n_layers=args.d_layers,
        norm_type=config.model.norm_type,
    ).to(device)
    print(f"Discriminator parameters: {netD.count_parameters():,}")

    optG = torch.optim.AdamW(
        netG.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    optD = torch.optim.Adam(
        netD.parameters(), lr=args.d_lr, betas=(0.5, 0.999),
    )
    schedG = torch.optim.lr_scheduler.CosineAnnealingLR(
        optG, T_max=config.train.T_max, eta_min=config.train.eta_min,
    )

    criterion = build_loss(config)          # composite L1 + SSIM + f-k
    adv_loss = nn.MSELoss()                 # LSGAN: more stable than BCE

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_snr = -float("inf")
    history = []

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        netG.train()
        netD.train()

        g_losses, d_losses, adv_losses = [], [], []

        for batch in train_loader:
            x          = batch["input"].to(device)        # (B, 2, T, O)
            target     = batch["target"].to(device)       # (B, 1, T, O)
            geometry   = batch["geometry"].to(device)     # (B,)
            trace_mask = batch["trace_mask"].to(device)   # (B, O)
            validity   = batch["validity"].to(device)     # (B, O)

            # ---- Discriminator -------------------------------------------
            optD.zero_grad(set_to_none=True)

            with torch.no_grad():
                fake = netG(x, geometry)

            pred_real = netD(x, target)
            pred_fake = netD(x, fake)

            d_loss = 0.5 * (
                adv_loss(pred_real, torch.ones_like(pred_real))
                + adv_loss(pred_fake, torch.zeros_like(pred_fake))
            )
            d_loss.backward()
            optD.step()

            # ---- Generator -----------------------------------------------
            optG.zero_grad(set_to_none=True)

            fake = netG(x, geometry)
            pred_fake = netD(x, fake)

            # NOTE: the reconstruction term is the *same* composite loss used
            # by the proposed model, called exactly as in train.train_one_epoch.
            loss_dict = criterion(fake, target, trace_mask, validity)
            rec_loss = loss_dict["loss"]

            g_adv = adv_loss(pred_fake, torch.ones_like(pred_fake))
            g_loss = rec_loss + args.lambda_adv * g_adv

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 1.0)
            optG.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            adv_losses.append(g_adv.item())

        schedG.step()

        # ---- Validation: identical metric to the proposed model ----------
        val_metrics = validate(netG, val_loader, criterion, device)
        val_snr = val_metrics.get("snr_db_mean", -float("inf"))

        history.append({
            "epoch": epoch,
            "lr": optG.param_groups[0]["lr"],
            "g_loss": float(np.mean(g_losses)),
            "d_loss": float(np.mean(d_losses)),
            "adv_loss": float(np.mean(adv_losses)),
            **{k: float(v) for k, v in val_metrics.items()},
        })

        if val_snr > best_snr:
            best_snr = val_snr
            torch.save(
                {
                    # saved as "model_state" so every evaluation script
                    # loads this checkpoint unchanged
                    "model_state": netG.state_dict(),
                    "discriminator_state": netD.state_dict(),
                    "epoch": epoch,
                    "val_snr_db": val_snr,
                    "seed": args.seed,
                    "lambda_adv": args.lambda_adv,
                },
                ckpt_dir / "best.pt",
            )

        if epoch % 10 == 0 or epoch == 1:
            print(f"  E{epoch:03d} | G={np.mean(g_losses):.4f} "
                  f"| D={np.mean(d_losses):.4f} "
                  f"| adv={np.mean(adv_losses):.4f} "
                  f"| val_SNR={val_snr:.2f}dB "
                  f"| lr={optG.param_groups[0]['lr']:.2e} "
                  f"| {time.time() - t0:.1f}s")

    # ── Final evaluation on the internal test set ────────────────────────
    print(f"\n  Loading best checkpoint (val SNR = {best_snr:.2f} dB)...")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
    netG.load_state_dict(ckpt["model_state"])

    test_metrics = validate(netG, test_loader, criterion, device)
    print(f"  Best val SNR:  {best_snr:.2f} dB")
    print(f"  Test SNR  = {test_metrics.get('snr_db_mean', 0):.2f} ±"
          f" {test_metrics.get('snr_db_std', 0):.2f} dB")
    print(f"  Test SSIM = {test_metrics.get('ssim_mean', 0):.4f}")
    print(f"  Test FK   = {test_metrics.get('fk_energy_ratio_mean', 0):.4f}")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "seed": args.seed,
        "lambda_adv": args.lambda_adv,
        "best_val_snr_db": float(best_snr),
        "best_epoch": int(ckpt.get("epoch", -1)),
        **{k: float(v) for k, v in test_metrics.items()},
    }
    with open(ckpt_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Checkpoint: {ckpt_dir / 'best.pt'}")
    print("  Evaluate with the paper's convention:")
    print(f"    python scripts/evaluate/recompute_with_per_gather_lists.py "
          f"--checkpoint {ckpt_dir / 'best.pt'} --strategies random --skip-pocs")
    return summary


def main():
    p = argparse.ArgumentParser(description="Conditional-GAN baseline")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-adv", type=float, default=0.01,
                   help="weight of the adversarial term in the generator loss")
    p.add_argument("--d-lr", type=float, default=2e-4,
                   help="discriminator learning rate")
    p.add_argument("--d-channels", type=int, default=64)
    p.add_argument("--d-layers", type=int, default=3)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs only, to check that everything runs")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 2

    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(CKPT_ROOT / f"gan_seed{args.seed}")

    config = get_config()
    train_gan(config, args)


if __name__ == "__main__":
    main()
