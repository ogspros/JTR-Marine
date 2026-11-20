#!/usr/bin/env python3
"""
check_against_reference.py -- compare a results JSON with a reference JSON.

Walks the two JSON trees and, on every key they have in common, compares the
numbers (scalars and per-gather lists). Keys present in only one file and
null values (e.g. pocs_* fields of a --skip-pocs run) are listed but not
compared, so a run on a subset of strategies / ratios can be checked against
a complete reference. Top-level run metadata ("config", "meta": paths, list
of ratios, flags) is ignored by default (--ignore).

Exit status 0 if every compared number agrees within --tol, 1 otherwise.

Usage:
    python scripts/stats/check_against_reference.py \
        reference_results/internal_test_compact_unet_seed42.json \
        results/nonregression/internal/per_gather_results.json
"""
import argparse
import json
import math
import sys


def walk(ref, new, path, rows, skipped, ignore=()):
    if isinstance(ref, dict) and isinstance(new, dict):
        for k in sorted(set(ref) | set(new), key=str):
            if path == "" and k in ignore:
                skipped.append(f"/{k}: run metadata, ignored")
                continue
            if k not in ref or k not in new:
                skipped.append(f"{path}/{k}: only in {'reference' if k in ref else 'new'}")
                continue
            walk(ref[k], new[k], f"{path}/{k}", rows, skipped)
    elif ref is None or new is None:
        if ref is not new:
            skipped.append(f"{path}: null in {'reference' if ref is None else 'new'}")
    elif isinstance(ref, list) and isinstance(new, list):
        if len(ref) != len(new):
            rows.append((path, math.inf, f"length {len(ref)} vs {len(new)}"))
            return
        if all(isinstance(v, (int, float)) for v in ref + new):
            diffs = [abs(a - b) for a, b in zip(ref, new)
                     if not (isinstance(a, float) and isinstance(b, float)
                             and math.isnan(a) and math.isnan(b))]
            rows.append((path, max(diffs, default=0.0), f"n={len(ref)}"))
        else:
            for i, (a, b) in enumerate(zip(ref, new)):
                walk(a, b, f"{path}[{i}]", rows, skipped)
    elif isinstance(ref, bool) or isinstance(new, bool) or isinstance(ref, str):
        if ref != new:
            rows.append((path, math.inf, f"{ref!r} vs {new!r}"))
    elif isinstance(ref, (int, float)) and isinstance(new, (int, float)):
        rows.append((path, abs(ref - new), ""))
    else:
        rows.append((path, math.inf, f"type {type(ref).__name__} vs {type(new).__name__}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reference")
    ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="maximum absolute difference accepted (default 1e-6)")
    ap.add_argument("--ignore", nargs="*", default=["config", "meta"],
                    help="top-level keys not compared (default: config meta)")
    ap.add_argument("--show", type=int, default=15,
                    help="number of largest differences to print")
    args = ap.parse_args()

    with open(args.reference) as f:
        ref = json.load(f)
    with open(args.new) as f:
        new = json.load(f)

    rows, skipped = [], []
    walk(ref, new, "", rows, skipped, set(args.ignore))
    if not rows:
        print("Nothing in common to compare.")
        sys.exit(1)

    rows.sort(key=lambda r: -r[1])
    worst = rows[0][1]
    n_bad = sum(r[1] > args.tol for r in rows)
    print(f"Compared {len(rows)} entries; not compared: {len(skipped)}")
    print(f"Largest absolute difference: {worst:.3g}  (tolerance {args.tol:g})")
    for path, d, note in rows[:args.show]:
        print(f"  {d:12.3g}  {path}  {note}")
    if skipped:
        print("Not compared (first 10):")
        for s in skipped[:10]:
            print(f"  {s}")
    print("PASS" if n_bad == 0 else f"FAIL: {n_bad} entries above tolerance")
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
