#!/usr/bin/env python3
"""
find_mask_seed_offset.py -- recover the mask-seed offset of a past run of
recompute_with_per_gather_lists.py made without a fixed PYTHONHASHSEED.

The masks of the internal test use the seed
    config.data.random_seed + abs(hash(strategy)) % 1000,
so a run made with an unknown hash seed used one of 1000 possible offsets.
For every offset K = 0..999 this script masks the first --n-check test gathers
of the first line, runs the model, and compares SNR_miss with the per-gather
values stored in the reference JSON. The offsets that reproduce them are then
passed to recompute_with_per_gather_lists.py (and recompute_mssa_per_gather.py)
with --mask-seed-offset STRATEGY=K to regenerate the whole run.

The data are loaded once; each offset costs n-check forward passes.

Usage:
    python scripts/evaluate/find_mask_seed_offset.py \
        --reference reference_results/statistics/per_gather_results.json \
        --strategies random block
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))      # sibling scripts

import argparse
import json

import numpy as np
import torch

from config import CKPT_ROOT, get_config
from jtr_marine.data.dataset import SharedGatherDataset
from jtr_marine.metrics import compute_snr
from jtr_marine.model_loader import load_checkpoint
from recompute_with_per_gather_lists import build_test_set


def snr_miss_first(model, device, ds, n):
    """SNR_miss of the first n gathers of ds, computed as in evaluate_one_setting."""
    out = []
    for idx in range(n):
        s = ds[idx]
        target = s["target"].squeeze().numpy()
        masked = s["input"][0].numpy()
        tm = s["trace_mask"].numpy()
        x = s["input"].unsqueeze(0).to(device)
        g = torch.tensor([s["geometry"].item()], device=device)
        with torch.no_grad():
            pred = model(x, g).cpu().squeeze().numpy()
        recon = pred * (1 - tm)[np.newaxis, :] + masked * tm[np.newaxis, :]
        out.append(compute_snr(torch.from_numpy(recon[None, None]).float(),
                               torch.from_numpy(target[None, None]).float(),
                               torch.from_numpy(tm[None]).float(),
                               s["validity"][None].float(),
                               missing_only=True).item())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True,
                    help="per_gather_results.json of the run to reproduce")
    ap.add_argument("--checkpoint",
                    default=str(CKPT_ROOT / "v6_grid25m/within_geology/best.pt"))
    ap.add_argument("--strategies", nargs="+", default=["random", "block"])
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--n-check", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    config = get_config()
    dc = config.data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, config, device)
    ref = json.load(open(args.reference))
    rkey = f"{int(args.ratio * 100)}pct"

    # first line of the test set; only the mask seed changes between offsets
    base = build_test_set(config, args.strategies[0], args.ratio).datasets[0]

    found = {}
    for strategy in args.strategies:
        target = np.asarray(ref[strategy][rkey]["dl_snr_miss"][:args.n_check])
        hits = []
        for k in range(1000):
            ds = SharedGatherDataset(
                gathers=base.gathers, indices=base.indices,
                geometry_code=base.geometry_code, n_time=base.n_time,
                n_offsets=base.n_offsets, masking_ratios=[args.ratio],
                masking_strategy=strategy, block_size=base.block_size,
                random_seed=dc.random_seed + k)
            v = snr_miss_first(model, device, ds, args.n_check)
            if np.max(np.abs(v - target)) <= args.tol:
                hits.append(k)
                print(f"  {strategy}: offset {k} matches "
                      f"(max diff {np.max(np.abs(v - target)):.2e})")
        print(f"{strategy}: {len(hits)} matching offset(s): {hits}")
        if len(hits) == 1:
            found[strategy] = hits[0]

    if found:
        opt = " ".join(f"{s}={k}" for s, k in found.items())
        print("\nRegenerate the reference run with:\n"
              f"  python scripts/evaluate/recompute_with_per_gather_lists.py "
              f"--mask-seed-offset {opt} ...")


if __name__ == "__main__":
    main()
