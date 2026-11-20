#!/usr/bin/env python3
"""
marmousi_to_segy.py — Convert the original Marmousi shot data (Madagascar .hh)
into SEG-Y files that the project's data loader can read like the real surveys.

Source data
-----------
Original Marmousi synthetic data (Brougois et al., 1990, "Marmousi, model and
data", doi:10.3997/2214-4609.201411190), as distributed with Madagascar
(marmrefl.hh). Acoustic, marine, end-on:
    726 samples at 4 ms (2.9 s)
    96 channels, 25 m spacing, offsets 2575 -> 200 m (far first in the file)
    240 shots, 25 m apart, from 3000 to 8975 m
Stored as big-endian 32-bit floats ("xdr_float") after the 3-byte
separator \\x0c\\x0c\\x04 that ends the text header.

Output
------
Two SEG-Y files, both sorted shot by shot, near offset first:
    MARMOUSI_1500.sgy : the 60 nearest channels (200 - 1675 m), to be used
                        with the 1500 m conditioning (a geometry seen in training)
    MARMOUSI_full.sgy : all 96 channels (200 - 2575 m); the streamer length
                        falls between the training geometries

Time axis: native 4 ms sampling is kept, and the traces are zero-padded from
2.9 s to 4.5 s so that the loader finds at least the 2250 samples (at 2 ms) it
expects; the loader then resamples 4 -> 2 ms exactly as it does for IT89-33 and
the CROP lines. Pass --no-pad to disable.

Headers written (standard SEG-Y positions), by default mirroring the
convention of the real surveys (checked on MEPA_SP_200_500.sgy: channel 1 =
nearest, negative offsets in metres, coordinates in decimetres with scalar -10):
    FieldRecord        shot number (1..240)
    TraceNumber        channel within the shot (1 = nearest)
    offset             source-receiver offset in metres, NEGATIVE
    SourceX, GroupX    coordinates x10 (decimetres); receivers trail the source:
                       GroupX = SourceX - |offset|
    CDP_X              midpoint
    SourceGroupScalar  -10
    TRACE_SAMPLE_COUNT / TRACE_SAMPLE_INTERVAL
Use --positive-offsets / --coord-scalar 1 to write the plain convention.

Usage
-----
    python scripts/data_prep/marmousi_to_segy.py data/marmousi/marmrefl.hh --outdir data/raw
    python scripts/data_prep/marmousi_to_segy.py data/marmousi/marmrefl.hh --outdir data/raw --far-first
"""

import argparse
from pathlib import Path

import numpy as np
import segyio

N1, N2, N3 = 726, 96, 240          # samples, channels, shots
DT_S = 0.004
FAR_OFFSET, DX = 2575.0, 25.0      # channel k (file order) has offset 2575 - 25k
SHOT0, DSHOT = 3000.0, 25.0


def read_hh(path):
    raw = Path(path).read_bytes()
    sep = raw.find(b"\x0c\x0c\x04")
    if sep < 0:
        raise ValueError("Madagascar header separator not found")
    data = np.frombuffer(raw[sep + 3:], dtype=">f4")
    if data.size != N1 * N2 * N3:
        raise ValueError(f"expected {N1 * N2 * N3} samples, found {data.size}")
    cube = data.reshape(N3, N2, N1).astype(np.float32)   # (shot, channel, time)
    offsets = FAR_OFFSET - DX * np.arange(N2)             # 2575 ... 200
    # reorder near offset first
    cube = cube[:, ::-1, :]
    offsets = offsets[::-1]                               # 200 ... 2575
    return cube, offsets


def write_segy(path, cube, offsets, pad_to_s, far_first,
               negative_offsets=True, coord_scalar=-10):
    nshot, nch, ns = cube.shape
    if far_first:
        cube = cube[:, ::-1, :]
        offsets = offsets[::-1]

    if pad_to_s is not None:
        ns_out = int(round(pad_to_s / DT_S))
        if ns_out > ns:
            pad = np.zeros((nshot, nch, ns_out - ns), dtype=np.float32)
            cube = np.concatenate([cube, pad], axis=2)
    ns_out = cube.shape[2]

    spec = segyio.spec()
    spec.format = 5                     # IEEE float
    spec.samples = np.arange(ns_out) * DT_S * 1000.0   # ms
    spec.tracecount = nshot * nch

    H = segyio.TraceField
    # SEG-Y scalar convention: negative -> divide stored value by |scalar|
    cmul = abs(coord_scalar) if coord_scalar < 0 else 1.0
    osign = -1.0 if negative_offsets else 1.0
    with segyio.create(str(path), spec) as f:
        f.bin.update(hdt=int(DT_S * 1e6), hns=ns_out, format=5)
        itr = 0
        for s in range(nshot):
            sx = SHOT0 + DSHOT * s
            for c in range(nch):
                off = float(offsets[c])
                gx = sx - off
                f.header[itr] = {
                    H.TRACE_SEQUENCE_LINE: itr + 1,
                    H.TRACE_SEQUENCE_FILE: itr + 1,
                    H.FieldRecord: s + 1,
                    H.TraceNumber: c + 1,
                    H.offset: int(round(osign * off)),
                    H.SourceX: int(round(sx * cmul)),
                    H.GroupX: int(round(gx * cmul)),
                    H.CDP_X: int(round(0.5 * (sx + gx) * cmul)),
                    H.SourceGroupScalar: int(coord_scalar),
                    H.TRACE_SAMPLE_COUNT: ns_out,
                    H.TRACE_SAMPLE_INTERVAL: int(DT_S * 1e6),
                }
                f.trace[itr] = cube[s, c]
                itr += 1
    return nshot, nch, ns_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hh", help="path to marmrefl.hh")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--no-pad", action="store_true")
    ap.add_argument("--pad-to", type=float, default=4.5,
                    help="pad record length to this many seconds (default 4.5)")
    ap.add_argument("--far-first", action="store_true",
                    help="write far offset first within each shot")
    ap.add_argument("--positive-offsets", action="store_true",
                    help="write positive offsets (default: negative, as in MEPA)")
    ap.add_argument("--coord-scalar", type=int, default=-10,
                    help="SourceGroupScalar (default -10, as in MEPA)")
    ap.add_argument("--n-near", type=int, default=60,
                    help="channels kept for the 1500 m version (default 60)")
    args = ap.parse_args()

    cube, offsets = read_hh(args.hh)
    print(f"read {args.hh}: {cube.shape[0]} shots x {cube.shape[1]} channels "
          f"x {cube.shape[2]} samples, offsets {offsets[0]:.0f}-{offsets[-1]:.0f} m")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pad = None if args.no_pad else args.pad_to

    jobs = [
        ("MARMOUSI_1500.sgy", cube[:, :args.n_near, :], offsets[:args.n_near]),
        ("MARMOUSI_full.sgy", cube, offsets),
    ]
    for name, c, o in jobs:
        ns, nc, nt = write_segy(out / name, c, o, pad, args.far_first,
                                not args.positive_offsets, args.coord_scalar)
        print(f"wrote {out / name}: {ns} shots x {nc} channels x {nt} samples "
              f"({nt * DT_S:.2f} s at {DT_S * 1000:.0f} ms), "
              f"offsets {o.min():.0f}-{o.max():.0f} m")


if __name__ == "__main__":
    main()
