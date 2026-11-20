#!/usr/bin/env python3
"""
amp_check_real.py — Amplitude fidelity of the DL reconstruction on the REAL
internal test set (276 gathers), for comparison with Marmousi.

For each gather: median over the missing traces of RMS(recon)/RMS(truth) and of
the trace-by-trace correlation. Same test set, masks and trace replacement as the
paper's Table 5 (build_test_set from recompute_with_per_gather_lists.py).
DL only: a few minutes. With --baselines also POCS (n_iter=100) and MSSA
(rank=10, n_iter=3), with the parameters of Table 5: about 45 minutes.

Usage
-----
    python scripts/evaluate/amp_check_real.py                    # 30% random
    python scripts/evaluate/amp_check_real.py --ratio 0.5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT

import argparse
import numpy as np
import torch

from config import get_config
from jtr_marine.model_loader import load_checkpoint
from recompute_with_per_gather_lists import build_test_set
from jtr_marine.baselines.pocs import pocs_interpolate_validity_aware
from jtr_marine.baselines.mssa import mssa_interpolate_validity_aware

SURVEYS = ["MEPA", "CG-1", "IT90-65", "IT89-33", "CROP-M31", "CROP-M32"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(CKPT_ROOT / "v6_grid25m/within_geology/best.pt"))
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--strategy", default="random")
    ap.add_argument("--baselines", action="store_true",
                    help="also evaluate POCS and MSSA (about 45 min)")
    args = ap.parse_args()

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, config, device)
    model.eval()
    ds = build_test_set(config, args.strategy, args.ratio)

    methods = ["dl"] + (["mssa", "pocs"] if args.baselines else [])
    rms = {m: [] for m in methods}
    cor = {m: [] for m in methods}
    for idx in range(len(ds)):
        s = ds[idx]
        target = s["target"].squeeze().numpy()
        masked = s["input"][0].numpy()
        mask = s["trace_mask"].numpy()
        valid = s["validity"].numpy()
        with torch.no_grad():
            dl = model(s["input"].unsqueeze(0).to(device),
                       torch.tensor([s["geometry"].item()], device=device)
                       ).cpu().squeeze().numpy()
        preds = {"dl": dl}
        if args.baselines:
            preds["pocs"] = pocs_interpolate_validity_aware(masked, mask, valid, n_iter=100)
            preds["mssa"] = mssa_interpolate_validity_aware(masked, mask, valid,
                                                            rank=10, n_iter=3)
        keep = mask[np.newaxis, :]
        cols = np.flatnonzero(((1 - mask) * valid) > 0)
        rt = np.sqrt((target[:, cols] ** 2).mean(axis=0))
        for m, pred in preds.items():
            recon = pred * (1 - keep) + masked * keep
            rp = np.sqrt((recon[:, cols] ** 2).mean(axis=0))
            rms[m].append(np.median(rp / np.maximum(rt, 1e-12)))
            cor[m].append(np.median([np.corrcoef(recon[:, j], target[:, j])[0, 1]
                                     for j in cols]))
        if (idx + 1) % 46 == 0:
            print(f"  [{idx+1}/{len(ds)}]")

    n_all = len(rms["dl"])
    n = n_all // len(SURVEYS)
    print(f"\nREAL test set, {args.strategy} {int(args.ratio*100)}%, n={n_all}")
    for m in methods:
        r, c = np.array(rms[m]), np.array(cor[m])
        print(f"\n  {m.upper()}: RMS ratio median {np.median(r):.3f} "
              f"[p10, p90] = [{np.percentile(r,10):.2f}, {np.percentile(r,90):.2f}]"
              f"   corr median {np.median(c):.3f}")
        for k, name in enumerate(SURVEYS):
            sl = slice(k * n, (k + 1) * n)
            print(f"    {name:9s} RMS ratio {np.median(r[sl]):.3f}   corr {np.median(c[sl]):.3f}")


if __name__ == "__main__":
    main()
