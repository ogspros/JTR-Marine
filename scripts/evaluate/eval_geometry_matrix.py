#!/usr/bin/env python3
"""
eval_geometry_matrix.py -- one uniform evaluation of the multi-geometry,
single-geometry and leave-one-out models on the internal test gathers
(material for Table 6).

Models (rows)
    multi           compact U-Net of the paper (all six training lines)
    single_1500     single-geometry models (train_single_geometry_all.py)
    single_3000
    single_4500
    logo_1500..4500 leave-one-GEOMETRY-out models (train.py --holdout-geometry L):
                    trained without both lines of that geometry; their column
                    for the held-out geometry is the true LOGO result.
    loo_fold0..2    models of train.py --fold k. NOTE: build_fold_datasets()
                    holds out config.data.dataset_paths[k], i.e. ONE LINE, not
                    one geometry; with the six-line configuration fold 0/1/2 =
                    SISOC-MEPA / CG-1 / IT90-65. The script prints the line
                    held out by each fold.

Test sets (columns)
    the internal test gathers of recompute_with_per_gather_lists.py (last 15 %
    of each line, contiguous split): 46 per line, 92 per geometry.

Metric
    SNR on the missing traces (trace replacement is applied, as in the paper),
    30 % random masking, the SAME masks for every model: each model is run
    through recompute_with_per_gather_lists.evaluate_one_setting(), which
    rebuilds the test set, so the masks are those of
    that script (see --mask-seed-offset for the Table 5 masks). Mask seeds depend on
    hash(strategy): the script re-executes itself with PYTHONHASHSEED=0.

Leakage note: the leave-one-out models were trained on the first 85 % of the
lines they did not hold out; the first internal test gather of each of those
lines (index 255 of 301) is inside that part.

Outputs (RESULTS_ROOT/geometry_matrix):
    matrix_per_gather.json   per-gather SNR_miss of every model
    matrix.txt               mean SNR_miss per geometry and per line

Usage:
    python scripts/evaluate/eval_geometry_matrix.py
    # the masks of Table 5:
    python scripts/evaluate/eval_geometry_matrix.py --mask-seed-offset random=587
    # checkpoint folders named checkpoints_<name> instead of <name>:
    python scripts/evaluate/eval_geometry_matrix.py --ckpt-root /path/to/folder --prefixed-names
"""
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
sys.path.insert(0, str(Path(__file__).resolve().parent))      # sibling scripts

import argparse
import json

import numpy as np
import torch

from config import CKPT_ROOT, RESULTS_ROOT, get_config
from jtr_marine.model_loader import load_checkpoint
import recompute_with_per_gather_lists as RW
from recompute_with_per_gather_lists import build_test_set, evaluate_one_setting

MODELS = [
    ("multi",       "v6_grid25m/within_geology/best.pt"),
    ("single_1500", "single_1500m_all/best.pt"),
    ("single_3000", "single_3000m_all/best.pt"),
    ("single_4500", "single_4500m_all/best.pt"),
    ("loo_fold0",   "v6_grid25m/fold0/best.pt"),
    ("loo_fold1",   "v6_grid25m/fold1/best.pt"),
    ("loo_fold2",   "v6_grid25m/fold2/best.pt"),
    # leave-one-GEOMETRY-out (train.py --holdout-geometry L); skipped if absent
    ("logo_1500",   "logo_1500m/within_geology/best.pt"),
    ("logo_3000",   "logo_3000m/within_geology/best.pt"),
    ("logo_4500",   "logo_4500m/within_geology/best.pt"),
]

# values printed in Table 6 of the paper (for the comparison only)
PAPER = {
    "multi":       {1500: 4.79, 3000: 8.79, 4500: 9.43},
    "single_1500": {1500: 1.76},
    "single_3000": {3000: 2.51},
    "single_4500": {4500: 2.39},
    "loo_fold0":   {1500: 0.98},
    "loo_fold1":   {3000: 2.03},
    "loo_fold2":   {4500: 2.00},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default=str(CKPT_ROOT))
    ap.add_argument("--prefixed-names", action="store_true",
                    help="checkpoint folders named checkpoints_<name> instead of <name>")
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--strategy", default="random")
    ap.add_argument("--output-dir", default=str(RESULTS_ROOT / "geometry_matrix"))
    ap.add_argument("--mask-seed-offset", nargs="+", metavar="STRATEGY=K",
                    help="fixed mask-seed offset (as in recompute_with_per_gather_lists.py); "
                         "random=587 gives the masks of Table 5. Default: hash with "
                         "PYTHONHASHSEED=0")
    args = ap.parse_args()
    RW.MASK_SEED_OFFSET.update(RW.parse_mask_seed_offsets(args.mask_seed_offset))
    mask_note = (f"mask-seed offset {RW.MASK_SEED_OFFSET}" if RW.MASK_SEED_OFFSET
                 else "PYTHONHASHSEED=0")

    config = get_config()
    dc = config.data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # layout of the test set: one block of gathers per line, in config order
    test = build_test_set(config, args.strategy, args.ratio)
    sizes = [len(d) for d in test.datasets]
    lines = [Path(p).stem for p in dc.dataset_paths]
    bounds = np.cumsum([0] + sizes)
    line_slices = {ln: slice(bounds[i], bounds[i + 1]) for i, ln in enumerate(lines)}
    geoms = sorted(set(dc.streamer_lengths))
    geom_idx = {int(g): np.concatenate([np.arange(bounds[i], bounds[i + 1])
                                        for i, L in enumerate(dc.streamer_lengths) if L == g])
                for g in geoms}
    del test

    print("PYTHONHASHSEED =", os.environ.get("PYTHONHASHSEED"), "|", mask_note)
    print("Test gathers per line:", dict(zip(lines, sizes)))
    for k in range(3):
        print(f"loo_fold{k} holds out line {k}: {lines[k]} "
              f"({dc.streamer_lengths[k]:.0f} m)")

    dt_s = dc.dt_target_ms * 1e-3
    results = {"meta": {"strategy": args.strategy, "ratio": args.ratio,
                        "mask_seed_offset": dict(RW.MASK_SEED_OFFSET),
                        "lines": lines, "test_sizes": sizes,
                        "streamer_lengths": list(dc.streamer_lengths)}}
    for name, rel in MODELS:
        rel_path = Path(rel)
        if args.prefixed_names:
            rel_path = Path("checkpoints_" + rel_path.parts[0], *rel_path.parts[1:])
        ck = Path(args.ckpt_root) / rel_path
        if not ck.is_file():
            print(f"\nMISSING {name}: {ck} (skipped)")
            continue
        print(f"\n=== {name}: {ck}")
        model, _ = load_checkpoint(ck, config, device)
        r = evaluate_one_setting(model, device, config, args.strategy, args.ratio,
                                 dt_s, float(dc.dx_target_m), skip_pocs=True)
        results[name] = {"checkpoint": str(rel_path), "snr_miss": r["dl_snr_miss"],
                         "snr_full": r["dl_snr_full"]}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(out / "matrix_per_gather.json", "w") as f:
        json.dump(results, f, indent=1)

    # ---------------------------------------------------------------- table
    rows = []
    hdr = f"{'model':<12}" + "".join(f"{int(g):>9}m" for g in geoms) + \
          "".join(f"{ln[:12]:>14}" for ln in lines)
    rows.append(f"Mean SNR_miss (dB), {args.strategy} {int(args.ratio * 100)}%, "
                f"internal test gathers, {mask_note}")
    rows.append(hdr)
    for name, _ in MODELS:
        if name not in results:
            continue
        v = np.asarray(results[name]["snr_miss"])
        cells = "".join(f"{v[geom_idx[int(g)]].mean():10.2f}" for g in geoms)
        cells += "".join(f"{v[line_slices[ln]].mean():14.2f}" for ln in lines)
        rows.append(f"{name:<12}" + cells)

    rows.append("")
    rows.append("Comparison with Table 6 (paper value vs this evaluation, per geometry):")
    for name, ref in PAPER.items():
        if name not in results:
            continue
        v = np.asarray(results[name]["snr_miss"])
        for g, pv in ref.items():
            mv = v[geom_idx[g]].mean()
            rows.append(f"  {name:<12} {g} m: paper {pv:5.2f}  matrix {mv:5.2f}  "
                        f"diff {mv - pv:+.2f}  {'OK' if abs(mv - pv) <= 0.1 else 'DIFFERENT'}")
    rows.append("")
    rows.append("Leave-one-geometry-out (held-out geometry vs multi on the same gathers):")
    for g in geoms:
        name = f"logo_{int(g)}"
        if name in results and "multi" in results:
            v = np.asarray(results[name]["snr_miss"])[geom_idx[int(g)]].mean()
            m = np.asarray(results["multi"]["snr_miss"])[geom_idx[int(g)]].mean()
            rows.append(f"  {name}: {int(g)} m never seen in training: {v:5.2f}  "
                        f"(multi {m:5.2f}, diff {v - m:+.2f})")
    for k in range(3):
        name = f"loo_fold{k}"
        if name in results:
            v = np.asarray(results[name]["snr_miss"])
            rows.append(f"  {name} on its held-out line {lines[k]}: "
                        f"{v[line_slices[lines[k]]].mean():.2f}")

    text = "\n".join(rows)
    print("\n" + text)
    (out / "matrix.txt").write_text(text + "\n")
    print(f"\nSaved {out / 'matrix_per_gather.json'} and {out / 'matrix.txt'}")


if __name__ == "__main__":
    main()
