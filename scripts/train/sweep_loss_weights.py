#!/usr/bin/env python3
"""
sweep_loss_weights.py — One-at-a-time sensitivity analysis of the loss
weights (lambda_ssim, lambda_fk) of the proposed model.

Design
------
lambda_l1 is fixed at 1 (with AdamW, a uniform rescaling of the loss has
little effect; what matters are the ratios). lambda_ssim and lambda_fk are
varied one at a time around the default (0.3, 0.1), plus two informative
points: the setting used in early versions of the model (0.5, 0.3), and
L1 alone (0, 0).

The default configuration is NOT re-trained: three independent runs of it
already exist, and they are read from disk as the reference.

What is collected
-----------------
For each run, at the best validation epoch (selection criterion of the
paper: validation SNR on missing traces at 30% random masking):
    val SNR_miss, val SSIM, val f-k energy ratio, best epoch.
Selecting on SNR alone would trivially favour the most L1-heavy setting,
since L1 is the term closest to SNR. The SSIM and f-k terms exist to
preserve structure and spectral content, so the justification of the
weights is a trade-off, and the table reports all three metrics.

Parallelism
-----------
One run needs about 8 GB of GPU memory and ~3 GB of RAM for the data
cache, so two or three runs fit side by side on an RTX A6000. Each run is
slower when shared, but total throughput is higher. Check the epoch time
in the first log lines and adjust --parallel if needed.

Usage
-----
    python scripts/train/sweep_loss_weights.py --dry-run
    python scripts/train/sweep_loss_weights.py --parallel 2          # core grid, 6 runs
    python scripts/train/sweep_loss_weights.py --grid extended       # + finer points
    python scripts/train/sweep_loss_weights.py --summarise-only      # re-read results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = str(CKPT_ROOT / "lossweights")

GRIDS = {
    # (tag, lambda_ssim, lambda_fk)
    "core": [
        ("ssim0_fk0p1",   0.0, 0.1),   # is the SSIM term needed?
        ("ssim1_fk0p1",   1.0, 0.1),   # heavy SSIM
        ("ssim0p3_fk0",   0.3, 0.0),   # is the f-k term needed?
        ("ssim0p3_fk0p3", 0.3, 0.3),   # heavy f-k
        ("ssim0p5_fk0p3", 0.5, 0.3),   # setting of early model versions
        ("l1only",        0.0, 0.0),   # is the composite loss needed at all?
    ],
    "extended": [
        ("ssim0p1_fk0p1",  0.1, 0.1),
        ("ssim0p3_fk0p03", 0.3, 0.03),
    ],
}

# Existing runs of the default configuration (lambda_ssim=0.3, lambda_fk=0.1)
DEFAULT_RUNS = [
    str(CKPT_ROOT / "v6_grid25m/within_geology/history.json"),
    str(CKPT_ROOT / "v6_seed43/within_geology/history.json"),
    str(CKPT_ROOT / "v6_seed44/within_geology/history.json"),
]


def best_from_history(path):
    """Return metrics at the best validation-SNR epoch, or None."""
    p = Path(path)
    if not p.exists():
        return None
    h = json.load(open(p))
    best = max(h, key=lambda e: e.get("val_snr_db_mean", float("-inf")))
    return {
        "val_snr": best.get("val_snr_db_mean"),
        "val_ssim": best.get("val_ssim_mean"),
        "val_fk": best.get("val_fk_energy_ratio_mean"),
        "epoch": best.get("epoch"),
        "n_epochs": len(h),
    }


def launch(tag, lam_ssim, lam_fk, seed, epochs):
    cmd = [sys.executable, str(Path(__file__).with_name("train_v6_lossweights.py")),
           "--lambda-ssim", str(lam_ssim), "--lambda-fk", str(lam_fk),
           "--seed", str(seed), "--tag", tag, "--root", ROOT]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    log = open(f"log_lossweights_{tag}.log", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log


def run_grid(configs, parallel, seed, epochs, dry_run):
    queue = list(configs)
    running = []
    t0 = time.time()

    if dry_run:
        for tag, s, f in queue:
            print(f"  [dry-run] {tag:16s} lambda_ssim={s:<5g} lambda_fk={f:<5g}"
                  f"  -> log_lossweights_{tag}.log")
        return

    while queue or running:
        while queue and len(running) < parallel:
            tag, s, f = queue.pop(0)
            proc, log = launch(tag, s, f, seed, epochs)
            running.append((tag, proc, log, time.time()))
            print(f"[{(time.time()-t0)/60:6.1f} min] started  {tag}"
                  f"  (lambda_ssim={s:g}, lambda_fk={f:g})")
        time.sleep(30)
        still = []
        for tag, proc, log, ts in running:
            if proc.poll() is None:
                still.append((tag, proc, log, ts))
            else:
                log.close()
                status = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                print(f"[{(time.time()-t0)/60:6.1f} min] finished {tag}"
                      f"  {status}  ({(time.time()-ts)/60:.0f} min)")
        running = still

    print(f"\nAll runs finished in {(time.time()-t0)/3600:.1f} h")


def summarise(configs):
    rows = []

    # reference: existing default runs
    ref = [r for r in (best_from_history(p) for p in DEFAULT_RUNS) if r]
    if ref:
        import statistics as st
        rows.append({
            "tag": f"DEFAULT ({len(ref)} runs)", "lambda_ssim": 0.3, "lambda_fk": 0.1,
            "val_snr": st.mean(r["val_snr"] for r in ref),
            "val_ssim": st.mean(r["val_ssim"] for r in ref),
            "val_fk": st.mean(r["val_fk"] for r in ref),
            "epoch": None,
            "snr_std": st.stdev([r["val_snr"] for r in ref]) if len(ref) > 1 else None,
        })

    for tag, s, f in configs:
        r = best_from_history(f"{ROOT}/{tag}/within_geology/history.json")
        if r is None:
            print(f"  (no results yet for {tag})")
            continue
        rows.append({"tag": tag, "lambda_ssim": s, "lambda_fk": f, **r})

    if not rows:
        return

    print("\n" + "=" * 78)
    print("LOSS-WEIGHT SENSITIVITY — validation metrics at the best-SNR epoch")
    print("(lambda_l1 = 1 throughout; selection criterion: val SNR_miss @30% random)")
    print("=" * 78)
    print(f"{'config':20s} {'l_ssim':>7} {'l_fk':>6} {'SNR (dB)':>9} "
          f"{'SSIM':>7} {'R_fk':>7} {'epoch':>6}")
    print("-" * 78)
    for r in rows:
        ep = "" if r["epoch"] is None else r["epoch"]
        print(f"{r['tag']:20s} {r['lambda_ssim']:>7g} {r['lambda_fk']:>6g} "
              f"{r['val_snr']:>9.2f} {r['val_ssim']:>7.4f} {r['val_fk']:>7.4f} {ep:>6}")

    if rows[0]["tag"].startswith("DEFAULT") and rows[0].get("snr_std") is not None:
        print(f"\nRun-to-run std of the default (validation SNR): "
              f"{rows[0]['snr_std']:.3f} dB")
        print("Differences in SNR smaller than ~3x this value should not be "
              "interpreted as real effects.")

    out = Path(ROOT) / "sensitivity_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["core", "extended", "all"], default="core")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None,
                    help="default: keep config (150). Do not reduce for the paper.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args()

    if args.grid == "all":
        configs = GRIDS["core"] + GRIDS["extended"]
    else:
        configs = GRIDS[args.grid]

    print(f"Loss-weight sensitivity: grid={args.grid} ({len(configs)} runs), "
          f"parallel={args.parallel}, seed={args.seed}")

    if not args.summarise_only:
        run_grid(configs, args.parallel, args.seed, args.epochs, args.dry_run)
    if not args.dry_run:
        summarise(configs)


if __name__ == "__main__":
    main()
