"""
run_cross_survey_test_v6_final.py — Cross-survey test (Table 7): the model is
evaluated on three surveys not seen in training (CG03, IT245, CROP_M18) at
30, 40 and 50% random missing traces.

  - Gathers with n_valid < 20 recorded traces are excluded (clearly defective)
  - MEDIAN and quartiles are reported besides mean ± std (more robust to outliers)
  - The number of excluded gathers is printed and saved

The model is loaded with model_loader.load_checkpoint, which recognises the
architecture from the weights (compact U-Net v6 and its ablations, v9
variants, Chai et al. 2020 U-Net and its conditioned version).
Use a different --output-dir for each model: the default one holds the
reference file of the paper (cross_survey_v6_final.json).

Usage:
    python scripts/evaluate/run_cross_survey_test_v6_final.py

Outputs (RESULTS_ROOT/cross_survey_v6):
    cross_survey_v6_final.json, fig_cross_survey_v6.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from config import get_config
from jtr_marine.data.dataset import SeismicGatherDataset, SharedGatherDataset
from jtr_marine.model_loader import load_checkpoint
from jtr_marine.metrics import compute_snr, compute_mse, compute_ssim_metric, \
                          compute_fk_energy_ratio


DATA_ROOT = str(DATA_ROOT)

CROSS_SURVEY_DATASETS = [
    {
        "name":           "CG03",
        "path":           f"{DATA_ROOT}/CG03_SP150_450.sgy",
        "streamer_m":     1500.0,
        "description":    "Cagliari, Sardinia",
    },
    {
        "name":           "IT245",
        "path":           f"{DATA_ROOT}/IT245_SP_800_1100.sgy",
        "streamer_m":     3000.0,
        "description":    "Drake Passage, shallow water",
    },
    {
        "name":           "CROP_M18",
        "path":           f"{DATA_ROOT}/CROP_M18_SP_500_800.sgy",
        "streamer_m":     4500.0,
        "description":    "Upper Adriatic, shallow water",
    },
]

MASKING_RATIOS = [0.3, 0.4, 0.5]
MIN_VALID_TRACES = 20   # gathers with fewer than 20 valid traces are excluded


def robust_stats(values):
    """Mean, std, median, quartiles, min, max. NaN/inf are discarded."""
    arr = np.array(values, dtype=np.float64)
    finite = np.isfinite(arr)
    n_total   = len(arr)
    n_valid   = int(finite.sum())
    if n_valid == 0:
        return dict(n_total=n_total, n_valid=0, n_invalid=n_total,
                    mean=float('nan'), std=float('nan'),
                    median=float('nan'), q25=float('nan'), q75=float('nan'),
                    min=float('nan'), max=float('nan'))
    a = arr[finite]
    return dict(
        n_total   = n_total,
        n_valid   = n_valid,
        n_invalid = n_total - n_valid,
        mean      = float(a.mean()),
        std       = float(a.std()),
        median    = float(np.median(a)),
        q25       = float(np.percentile(a, 25)),
        q75       = float(np.percentile(a, 75)),
        min       = float(a.min()),
        max       = float(a.max()),
    )


@torch.no_grad()
def evaluate_with_filtering(model, dataset, gathers, device, dt, dx,
                              batch_size=4, min_valid_traces=MIN_VALID_TRACES):
    """
    Evaluate the model, excluding the gathers with too few valid traces.

    Returns:
        dict with the number of evaluated / excluded gathers and the
        statistics of each metric.
    """
    # Find the gathers with enough valid traces
    valid_indices = []
    excluded_low_coverage = []
    for i in range(len(dataset)):
        g = gathers[dataset.indices[i]]
        n_valid = int((np.abs(g).max(axis=0) > 0).sum())
        if n_valid >= min_valid_traces:
            valid_indices.append(i)
        else:
            excluded_low_coverage.append({"idx": i, "n_valid": n_valid})

    if len(excluded_low_coverage) > 0:
        excluded_strs = [f"idx={e['idx']} (n_valid={e['n_valid']})"
                          for e in excluded_low_coverage[:5]]
        more = "" if len(excluded_low_coverage) <= 5 else \
               f" + {len(excluded_low_coverage)-5} more"
        print(f"    Excluded {len(excluded_low_coverage)} gathers with n_valid<"
              f"{min_valid_traces}: {', '.join(excluded_strs)}{more}")

    # Subset
    from torch.utils.data import Subset
    subset = Subset(dataset, valid_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=2)

    snr_missing, snr_full, ssim_list, fk_list = [], [], [], []
    model.eval()

    for batch in loader:
        x          = batch["input"].to(device)
        target     = batch["target"].to(device)
        geometry   = batch["geometry"].to(device)
        trace_mask = batch["trace_mask"].to(device)
        validity   = batch["validity"].to(device)

        pred = model(x, geometry)

        tm_2d  = trace_mask.unsqueeze(1).unsqueeze(2).expand_as(pred)
        masked = x[:, 0:1, :, :]
        pred_replaced = pred * (1 - tm_2d) + masked * tm_2d

        for b in range(pred.shape[0]):
            p_raw = pred[b:b+1]
            p_rep = pred_replaced[b:b+1]
            t     = target[b:b+1]
            m     = trace_mask[b:b+1]
            v     = validity[b:b+1]

            snr_missing.append(compute_snr(p_raw, t, m, v).item())
            snr_full.append(compute_snr(p_rep, t, torch.zeros_like(m), v).item())
            ssim_list.append(compute_ssim_metric(p_rep, t).item())
            fk_list.append(compute_fk_energy_ratio(p_rep, t,
                                                     dt=dt, dx=dx).item())

    return {
        "n_evaluated":        len(snr_missing),
        "n_excluded_low_cov": len(excluded_low_coverage),
        "snr_missing":        robust_stats(snr_missing),
        "snr_full":           robust_stats(snr_full),
        "ssim":               robust_stats(ssim_list),
        "fk":                 robust_stats(fk_list),
    }


def build_test_dataset(config, segy_path, streamer_m, masking_ratio):
    dc = config.data
    n_t = int(round(dc.t_max_s * 1000.0 / dc.dt_target_ms))
    n_x = int(round(dc.x_max_m / dc.dx_target_m)) + 1

    full_ds = SeismicGatherDataset(
        data_path           = segy_path,
        streamer_length     = streamer_m,
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
    n_total = len(all_gathers)
    all_indices = np.arange(n_total, dtype=np.int64)
    geom_code = streamer_m / dc.max_streamer_length

    test_ds = SharedGatherDataset(
        gathers          = all_gathers,
        indices          = all_indices,
        geometry_code    = geom_code,
        n_time           = n_t,
        n_offsets        = n_x,
        masking_ratios   = [masking_ratio],
        masking_strategy = "random",
        block_size       = dc.block_size,
        random_seed      = dc.random_seed + int(masking_ratio * 1000),
    )
    return test_ds, all_gathers, n_total


def main(checkpoint_path, output_dir=str(RESULTS_ROOT / "cross_survey_v6")):
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"{'='*78}")
    print(f"CROSS-SURVEY TEST")
    print(f"{'='*78}")
    print(f"Filtering: gathers with n_valid < {MIN_VALID_TRACES} excluded")
    print(f"{'='*78}\n")

    print(f"Checkpoint: {checkpoint_path}")
    model, ckpt = load_checkpoint(checkpoint_path, config, device)
    print(f"Model loaded: epoch {ckpt.get('epoch', '?')}\n")

    all_results = {}
    for ds_cfg in CROSS_SURVEY_DATASETS:
        ds_name = ds_cfg["name"]
        print(f"--- {ds_name} ({ds_cfg['streamer_m']:.0f}m, "
              f"{ds_cfg['description']}) ---")
        all_results[ds_name] = {
            "streamer_m":  ds_cfg["streamer_m"],
            "description": ds_cfg["description"],
            "path":        ds_cfg["path"],
            "by_ratio":    {},
        }

        for ratio in MASKING_RATIOS:
            print(f"\n  [ratio {int(ratio*100)}%]")
            test_ds, gathers, n_total = build_test_dataset(
                config, ds_cfg["path"], ds_cfg["streamer_m"], ratio,
            )
            print(f"    n_total = {n_total} gathers")

            metrics = evaluate_with_filtering(
                model, test_ds, gathers, device,
                dt=config.data.dt_target_ms * 1e-3,
                dx=config.data.dx_target_m,
            )
            all_results[ds_name]["by_ratio"][f"{int(ratio*100)}pct"] = metrics

            print(f"    n_evaluated = {metrics['n_evaluated']} "
                  f"(excluded {metrics['n_excluded_low_cov']})")
            sf = metrics["snr_full"]
            sm = metrics["snr_missing"]
            print(f"    SNR_full    = {sf['mean']:+.2f} ± {sf['std']:.2f} dB"
                  f" | median {sf['median']:+.2f} [Q25-Q75: {sf['q25']:+.2f} to {sf['q75']:+.2f}]")
            print(f"    SNR_missing = {sm['mean']:+.2f} ± {sm['std']:.2f} dB"
                  f" | median {sm['median']:+.2f}")
            print(f"    SSIM        = {metrics['ssim']['mean']:.4f}")
            print(f"    FK          = {metrics['fk']['mean']:.4f}")

        print("")

    # Save
    summary_file = out_path / "cross_survey_v6_final.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved: {summary_file}")

    # Final table
    print(f"\n{'='*78}")
    print(f"SUMMARY — cross-survey test")
    print(f"{'='*78}")
    print(f"{'Dataset':<11}{'Geom':<8}", end="")
    for ratio in MASKING_RATIOS:
        print(f"  {int(ratio*100):3d}% SNR_full      ", end="")
    print()
    print("-" * 78)
    for ds_cfg in CROSS_SURVEY_DATASETS:
        print(f"{ds_cfg['name']:<11}{ds_cfg['streamer_m']:.0f}m   ", end="")
        for ratio in MASKING_RATIOS:
            m = all_results[ds_cfg["name"]]["by_ratio"][f"{int(ratio*100)}pct"]
            sf = m["snr_full"]
            print(f"  {sf['mean']:+5.2f} ± {sf['std']:4.2f}        ", end="")
        print()
    print("=" * 78)

    # Figure
    make_figure(all_results, out_path / "fig_cross_survey_v6.png",
                title_model=Path(checkpoint_path).parent.parent.name)
    print(f"\nFigure saved: {out_path / 'fig_cross_survey_v6.png'}")


def make_figure(results, output_path, title_model="v6"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))

    datasets = list(results.keys())
    ratios   = list(results[datasets[0]]["by_ratio"].keys())
    n_r      = len(ratios)
    bar_w    = 0.8 / n_r

    x_pos  = np.arange(len(datasets))
    colors = ["#065A82", "#1C7293", "#5BA5B3"]

    for i, ratio_key in enumerate(ratios):
        means = [results[ds]["by_ratio"][ratio_key]["snr_full"]["mean"]
                 for ds in datasets]
        stds  = [results[ds]["by_ratio"][ratio_key]["snr_full"]["std"]
                 for ds in datasets]
        means = [m if np.isfinite(m) else 0 for m in means]
        stds  = [s if np.isfinite(s) else 0 for s in stds]
        ax.bar(x_pos + i * bar_w - 0.4 + bar_w / 2,
               means, bar_w, yerr=stds, capsize=4,
               color=colors[i],
               label=f"{ratio_key.replace('pct', '%')} missing")

    labels = [f"{ds}\n({results[ds]['streamer_m']:.0f}m)" for ds in datasets]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("SNR (dB) — with trace replacement", fontsize=12)
    ax.set_title(
        f"Cross-survey generalization: {title_model} on 3 unseen surveys",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(CKPT_ROOT / "v6_grid25m/within_geology/best.pt"),
    )
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT / "cross_survey_v6"))
    args = parser.parse_args()
    main(args.checkpoint, args.output_dir)
