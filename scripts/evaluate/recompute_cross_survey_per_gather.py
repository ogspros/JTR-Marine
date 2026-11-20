"""
recompute_cross_survey_per_gather.py — Cross-survey test with per-gather values.

Runs the DL model on CG03 / IT245 / CROP_M18 (three surveys not seen in
training) and saves the metrics of every single gather, not only the
aggregates. They are needed for the 95% bootstrap CIs and for statistics
consistent with the other tables of the paper.

Requirements:
  - checkpoint of the compact U-Net
  - the three cross-survey SEG-Y files in DATA_ROOT (default ./data/raw/)

Run time: ~5 min (DL inference only, no POCS).

Usage:
    python scripts/evaluate/recompute_cross_survey_per_gather.py

Output:
  RESULTS_ROOT/statistics/cross_survey_per_gather.json

Format:
  {
    "CG03": {
      "30pct": {
        "n": 301,
        "snr_miss": [v1, v2, ..., v301],
        "snr_full": [...],
        "ssim": [...],
        "fk": [...]
      },
      ...
    },
    "IT245": {...},
    "CROP_M18": {...}
  }
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from copy import deepcopy
from torch.utils.data import DataLoader

from config import get_config
from jtr_marine.data.dataset import SeismicGatherDataset, SharedGatherDataset
from jtr_marine.models.unet import build_model
from jtr_marine.metrics import (
    compute_snr, compute_ssim_metric, compute_fk_energy_ratio,
)


# Cross-survey datasets (never seen during training)
CROSS_SURVEY_DATASETS = [
    {
        "name":       "CG03",
        "path":       str(DATA_ROOT / "CG03_SP150_450.sgy"),
        "length":     1500.0,
        "setting":    "Cagliari margin, Sardinia",
    },
    {
        "name":       "IT245",
        "path":       str(DATA_ROOT / "IT245_SP_800_1100.sgy"),
        "length":     3000.0,
        "setting":    "Drake Passage, shallow water",
    },
    {
        "name":       "CROP_M18",
        "path":       str(DATA_ROOT / "CROP_M18_SP_500_800.sgy"),
        "length":     4500.0,
        "setting":    "Upper Adriatic basin",
    },
]


def build_dataset_for_cross_survey(config, sgy_path, streamer_length,
                                       masking_ratio):
    """Build single-dataset SharedGatherDataset for cross-survey eval."""
    dc = config.data
    n_t = int(round(dc.t_max_s * 1000.0 / dc.dt_target_ms))
    n_x = int(round(dc.x_max_m / dc.dx_target_m)) + 1

    full_ds = SeismicGatherDataset(
        data_path           = sgy_path,
        streamer_length     = streamer_length,
        max_streamer_length = dc.max_streamer_length,
        dt_target_ms        = dc.dt_target_ms,
        t_max_s             = dc.t_max_s,
        x_max_m             = dc.x_max_m,
        dx_target_m         = dc.dx_target_m,
        masking_ratios      = [masking_ratio],
        masking_strategy    = "random",
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
    all_idx     = np.arange(n_total, dtype=np.int64)

    geom_code = streamer_length / dc.max_streamer_length

    # Fixed seed for reproducibility, same for all cross-survey
    return SharedGatherDataset(
        gathers          = all_gathers,
        indices          = all_idx,                 # use ALL gathers (no split)
        geometry_code    = geom_code,
        n_time           = n_t,
        n_offsets        = n_x,
        masking_ratios   = [masking_ratio],
        masking_strategy = "random",
        block_size       = dc.block_size,
        random_seed      = 12345,                    # fixed for reproducibility
    )


def evaluate_cross_survey(model, device, config, sgy_path, streamer_length,
                            masking_ratio, min_n_valid=20):
    """
    Evaluate DL on all gathers of one cross-survey dataset at one ratio.
    Returns per-gather metric lists.
    """
    dataset = build_dataset_for_cross_survey(config, sgy_path,
                                                streamer_length, masking_ratio)

    dt_seconds = config.data.dt_target_ms * 1e-3
    dx_meters  = float(config.data.dx_target_m)

    snr_miss_list = []
    snr_full_list = []
    ssim_list     = []
    fk_list       = []
    excluded      = 0

    model.eval()
    for idx in range(len(dataset)):
        sample = dataset[idx]
        target     = sample["target"].squeeze().numpy()
        masked     = sample["input"][0].numpy()
        trace_mask = sample["trace_mask"].numpy()
        validity   = sample["validity"].numpy()
        geometry   = sample["geometry"].item()

        # Skip gathers with too few valid traces (acquisition issues)
        n_valid = int(validity.sum())
        if n_valid < min_n_valid:
            excluded += 1
            continue

        # DL inference
        x = sample["input"].unsqueeze(0).to(device)
        g = torch.tensor([geometry], device=device)
        with torch.no_grad():
            dl_pred = model(x, g).cpu().squeeze().numpy()

        # Trace replacement
        dl_recon = dl_pred * (1 - trace_mask)[np.newaxis, :] + \
                   masked  * trace_mask[np.newaxis, :]

        # Metrics
        t_torch  = torch.from_numpy(target[None, None]).float()
        m_torch  = torch.from_numpy(trace_mask[None]).float()
        v_torch  = torch.from_numpy(validity[None]).float()
        dl_torch = torch.from_numpy(dl_recon[None, None]).float()

        snr_miss_list.append(compute_snr(dl_torch, t_torch, m_torch, v_torch,
                                            missing_only=True).item())
        snr_full_list.append(compute_snr(dl_torch, t_torch, m_torch, v_torch,
                                            missing_only=False).item())
        ssim_list.append(compute_ssim_metric(dl_torch, t_torch).item())
        fk_list.append(compute_fk_energy_ratio(dl_torch, t_torch,
                                                  dt=dt_seconds,
                                                  dx=dx_meters).item())

    return {
        "n":         len(snr_miss_list),
        "excluded":  excluded,
        "snr_miss":  snr_miss_list,
        "snr_full":  snr_full_list,
        "ssim":      ssim_list,
        "fk":        fk_list,
    }


def main(checkpoint, output_dir, ratios):
    config = get_config()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"CROSS-SURVEY PER-GATHER EVALUATION (DL only)")
    print(f"{'='*70}")
    print(f"Checkpoint:    {checkpoint}")
    print(f"Output:        {out_dir}")
    print(f"Ratios:        {[f'{int(r*100)}%' for r in ratios]}")
    print(f"Datasets:")
    for ds in CROSS_SURVEY_DATASETS:
        print(f"  {ds['name']:10s}  ({ds['length']:.0f}m, {ds['setting']})")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"  val_snr = {ckpt.get('val_snr', '?')}\n")

    all_results = {}
    t_global = time.time()

    for ds in CROSS_SURVEY_DATASETS:
        ds_name = ds["name"]
        all_results[ds_name] = {}
        for ratio in ratios:
            ratio_key = f"{int(ratio*100)}pct"
            t_exp = time.time()
            print(f"--- {ds_name} ({ds['length']:.0f}m) | {ratio_key} ---")
            results = evaluate_cross_survey(
                model, device, config,
                ds["path"], ds["length"], ratio,
            )
            all_results[ds_name][ratio_key] = results
            print(f"  n={results['n']}, excluded={results['excluded']}, "
                  f"time={time.time()-t_exp:.0f}s")
            m  = np.mean(results["snr_full"])
            ss = np.mean(results["ssim"])
            fk = np.mean(results["fk"])
            print(f"  SNR_full = {m:+.2f} dB, SSIM = {ss:.4f}, FK = {fk:.4f}\n")

    t_total = time.time() - t_global
    print(f"{'='*70}")
    print(f"DONE. Total time: {t_total/60:.1f} min")
    print(f"{'='*70}\n")

    output_file = out_dir / "cross_survey_per_gather.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default=str(CKPT_ROOT / "v6_grid25m/within_geology/best.pt"))
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT / "statistics"))
    parser.add_argument("--ratios", nargs="+", type=float,
                        default=[0.3, 0.4, 0.5])
    args = parser.parse_args()

    main(checkpoint=args.checkpoint,
         output_dir=args.output_dir,
         ratios=args.ratios)
