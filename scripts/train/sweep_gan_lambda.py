#!/usr/bin/env python3
"""
sweep_gan_lambda.py — Tune the adversarial weight of the conditional-GAN
baseline (Table 9).

Why
---
The paper states that both classical baselines (POCS and MSSA) are tuned by
exhaustive sweep on the same training pool, "so that the comparison is not
biased by under-tuned references". The same standard must be applied to the
deep learning baseline: reporting a GAN trained with an arbitrary
lambda_adv would be inconsistent with that claim.

This script sweeps lambda_adv, selecting on the SAME criterion used for model
selection of the proposed model: validation SNR on missing traces at 30%
random masking. The test set is never used for selection.

Control run
-----------
lambda_adv = 0 is included on purpose. With no adversarial term the generator
reduces exactly to the proposed model, so this run must reproduce the
baseline (~6.9 dB validation). If it does not, the GAN pipeline itself has a
problem, and the whole comparison is invalid. Treat it as a correctness check,
not as a candidate configuration.

Usage
-----
    python scripts/train/sweep_gan_lambda.py                      # full sweep, 150 epochs
    python scripts/train/sweep_gan_lambda.py --epochs 80          # faster ranking pass
    python scripts/train/sweep_gan_lambda.py --lambdas 0.005 0.01 # custom grid
    python scripts/train/sweep_gan_lambda.py --dry-run            # just print the plan

Cost
----
Roughly 100 minutes per configuration at 150 epochs. The default grid of five
values therefore takes about 8 hours.
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

DEFAULT_LAMBDAS = [0.0, 0.001, 0.005, 0.01, 0.05]


def run_one(lam, epochs, seed, outdir_root, dry_run=False):
    tag = f"lam{lam:g}".replace(".", "p")
    ckpt_dir = f"{outdir_root}/gan_{tag}_seed{seed}"
    log_file = f"log_gan_sweep_{tag}_seed{seed}.log"

    cmd = [
        sys.executable, str(Path(__file__).with_name("train_gan.py")),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--lambda-adv", str(lam),
        "--checkpoint-dir", ckpt_dir,
    ]

    print(f"\n{'=' * 66}")
    print(f"lambda_adv = {lam:g}   (seed {seed}, {epochs} epochs)")
    print(f"  checkpoint dir: {ckpt_dir}")
    print(f"  log:            {log_file}")
    print(f"{'=' * 66}")

    if dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return None

    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode}) — see {log_file}")
        return None

    summary_path = Path(ckpt_dir) / "summary.json"
    if not summary_path.exists():
        print(f"  no summary.json in {ckpt_dir}")
        return None

    with open(summary_path) as f:
        s = json.load(f)
    s["lambda_adv"] = lam
    s["epochs"] = epochs
    s["minutes"] = round(elapsed, 1)
    s["checkpoint_dir"] = ckpt_dir

    print(f"  best val SNR = {s['best_val_snr_db']:.2f} dB "
          f"(epoch {s.get('best_epoch', '?')}, {elapsed:.0f} min)")
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=42,
                   help="single seed for the sweep; the selected configuration "
                        "should then be re-run with 3 seeds")
    p.add_argument("--outdir-root", type=str, default=str(CKPT_ROOT / "gan_sweep"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    Path(args.outdir_root).mkdir(parents=True, exist_ok=True)

    print(f"GAN adversarial-weight sweep")
    print(f"  grid:   {args.lambdas}")
    print(f"  epochs: {args.epochs}")
    print(f"  seed:   {args.seed}")
    print(f"  estimated time: ~{len(args.lambdas) * args.epochs * 0.65 / 60:.1f} h")

    results = []
    for lam in args.lambdas:
        r = run_one(lam, args.epochs, args.seed, args.outdir_root, args.dry_run)
        if r is not None:
            results.append(r)

    if args.dry_run or not results:
        return

    results.sort(key=lambda r: -r["best_val_snr_db"])

    print(f"\n{'=' * 66}")
    print("SWEEP SUMMARY — ranked by validation SNR (selection criterion)")
    print(f"{'=' * 66}")
    print(f"{'lambda_adv':>12}  {'val SNR':>9}  {'epoch':>6}  {'min':>6}")
    print("-" * 42)
    for r in results:
        star = "  <- control (= proposed model)" if r["lambda_adv"] == 0 else ""
        print(f"{r['lambda_adv']:>12g}  {r['best_val_snr_db']:>9.2f}  "
              f"{r.get('best_epoch', -1):>6}  {r['minutes']:>6.0f}{star}")

    # best configuration EXCLUDING the control
    candidates = [r for r in results if r["lambda_adv"] > 0]
    if candidates:
        best = candidates[0]
        print(f"\nSelected: lambda_adv = {best['lambda_adv']:g} "
              f"(val SNR {best['best_val_snr_db']:.2f} dB)")
        print("Next step — re-run the selected value with three seeds:")
        for s in (42, 43, 44):
            print(f"  python scripts/train/train_gan.py --epochs {args.epochs} --seed {s} "
                  f"--lambda-adv {best['lambda_adv']:g}")

    control = [r for r in results if r["lambda_adv"] == 0]
    if control:
        v = control[0]["best_val_snr_db"]
        ok = abs(v - 6.91) < 0.15
        print(f"\nControl check (lambda_adv = 0): {v:.2f} dB "
              f"vs proposed model 6.91 dB -> {'consistent' if ok else 'INCONSISTENT'}")
        if not ok:
            print("  The GAN pipeline does not reproduce the baseline with the")
            print("  adversarial term switched off. Investigate before using")
            print("  any of these results.")

    out = Path(args.outdir_root) / "sweep_summary.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
