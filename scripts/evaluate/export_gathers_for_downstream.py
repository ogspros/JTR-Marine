"""
export_gathers_for_downstream.py — Export the CROP_M18 gathers (ground truth +
DL/POCS/MSSA reconstructions) to SEG-Y for downstream analysis
(stack/migration) in a seismic processing package; input of the stacked
sections of Fig. 8.

Pipeline:
  1. Load the original raw CROP_M18 SEG-Y
  2. For each shot:
     a. Apply the same preprocessing as in training (high-pass + clip + norm)
     b. Generate the mask at the requested ratio
     c. Run DL, POCS and MSSA on the normalized data
     d. Denormalize with the per-trace scale factors
     e. Apply trace replacement
  3. Export:
     - Version A: common grid (181 traces × 25 m)
     - Version B: original CROP_M18 geometry (180 traces × 25 m, offsets -150 → -4625)

Output (in RESULTS_ROOT/output_for_stack/):
  A_30/ A_50/ → common-grid version
  B_30/ B_50/ → original-geometry version
  Each directory holds 5 SEG-Y files: GT, masked, DL, POCS, MSSA

Expected run time: ~40 minutes (301 shots × 2 ratios × ~4 s/shot for MSSA).

Notes:
  - the model is loaded with model_loader.load_checkpoint, which handles the
    compact U-Net, the Chai et al. (2020) U-Net, its conditioned version
    and the other variants;
  - --skip-classical: POCS and MSSA are neither computed nor written (the masks
    depend only on --mask-seed and on the shot, so GT, masked, POCS and MSSA
    are identical to those exported for the compact U-Net). Only GT, masked
    and DL are written.
  Use a different --output-dir for each model: the default one holds the files
  of the paper.

Usage:
    python scripts/evaluate/export_gathers_for_downstream.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from config import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

import argparse
import time
import numpy as np
import segyio
import torch
from pathlib import Path
from scipy.signal import butter, filtfilt, resample_poly

from config import get_config
from jtr_marine.model_loader import load_checkpoint
from jtr_marine.baselines.pocs import pocs_interpolate_validity_aware
from jtr_marine.baselines.mssa import mssa_interpolate_validity_aware


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — same as clip_and_normalize in data/dataset.py
# ─────────────────────────────────────────────────────────────────────────────

def highpass_filter(gather: np.ndarray, f_cut: float, dt_ms: float) -> np.ndarray:
    """Butterworth high-pass filter (same as in data/dataset.py)."""
    fs    = 1000.0 / dt_ms
    nyq   = fs / 2.0
    norm  = f_cut / nyq
    b, a  = butter(4, norm, btype="high")
    out   = filtfilt(b, a, gather, axis=0).astype(np.float32)
    return out


def preprocess_and_save_scales(gather_raw: np.ndarray, dt_ms: float,
                                  apply_filter: bool = True,
                                  filter_f_cut: float = 8.0,
                                  clip_sigma: float = 3.0):
    """
    Same preprocessing as in training, but the per-trace scale factors are
    also returned (needed for denormalization).

    Returns:
        gather_norm:  (n_time, n_offsets) normalized to [-1, 1] trace by trace
        gather_clip:  (n_time, n_offsets) preprocessed but NOT normalized
        scale_factors: (n_offsets,) abs_max of each trace (zero for empty traces)
    """
    gather = gather_raw.copy().astype(np.float32)

    if apply_filter:
        gather = highpass_filter(gather, filter_f_cut, dt_ms)

    n_offsets = gather.shape[1]
    gather_clip = np.zeros_like(gather)
    gather_norm = np.zeros_like(gather)
    scale_factors = np.zeros(n_offsets, dtype=np.float32)

    for i in range(n_offsets):
        trace = gather[:, i]
        if np.all(trace == 0):
            continue
        sigma = np.std(trace)
        if sigma < 1e-10:
            continue
        trace_clip = np.clip(trace, -clip_sigma * sigma, clip_sigma * sigma)
        max_amp    = float(np.max(np.abs(trace_clip)) + 1e-10)
        gather_clip[:, i] = trace_clip
        gather_norm[:, i] = trace_clip / max_amp
        scale_factors[i]  = max_amp

    return gather_norm, gather_clip, scale_factors


def smart_denormalize(pred_norm: np.ndarray,
                       scale_factors: np.ndarray,
                       trace_mask: np.ndarray,
                       window_size: int = 5) -> np.ndarray:
    """
    Denormalize the model output with the per-trace scale factors.

    Known traces (trace_mask = 1): the true scale factor is used.
    Missing traces (trace_mask = 0): the scale factor is estimated as the mean
    over the nearby known traces (within window_size positions).
    """
    n_offsets = pred_norm.shape[1]
    smart_scales = scale_factors.copy()

    for i in range(n_offsets):
        if trace_mask[i] > 0.5 and scale_factors[i] > 0:
            continue                                 # known and valid: keep its scale
        # Look for nearby known traces
        lo = max(0, i - window_size)
        hi = min(n_offsets, i + window_size + 1)
        nearby = scale_factors[lo:hi]
        nearby_known = nearby[(nearby > 0) & (trace_mask[lo:hi] > 0.5)]
        if len(nearby_known) > 0:
            smart_scales[i] = nearby_known.mean()
        else:
            # Fallback: use all the known traces
            all_known = scale_factors[(scale_factors > 0) & (trace_mask > 0.5)]
            if len(all_known) > 0:
                smart_scales[i] = all_known.mean()

    return pred_norm * smart_scales[None, :]


# ─────────────────────────────────────────────────────────────────────────────
# Geometry conversion: original CROP_M18 → common grid
# ─────────────────────────────────────────────────────────────────────────────

def original_to_grid(traces_orig: np.ndarray,
                       offsets_orig: np.ndarray,
                       dx_target: float = 25.0,
                       x_max: float = 4500.0) -> np.ndarray:
    """
    Map the traces from the original geometry to the common grid (181 traces × 25 m).

    Args:
        traces_orig:  (n_time, n_orig)  original traces
        offsets_orig: (n_orig,)         offsets (may be negative)
        dx_target:    25 m
        x_max:        4500 m
    Returns:
        gather_grid: (n_time, 181) on the common grid
    """
    n_time = traces_orig.shape[0]
    n_grid = int(round(x_max / dx_target)) + 1     # 181
    gather_grid = np.zeros((n_time, n_grid), dtype=np.float32)

    # Absolute offsets
    abs_offsets = np.abs(offsets_orig)

    for trace_idx, off in enumerate(abs_offsets):
        if off > x_max:
            continue                                # outside the grid
        grid_idx = int(round(off / dx_target))
        if 0 <= grid_idx < n_grid:
            gather_grid[:, grid_idx] = traces_orig[:, trace_idx]

    return gather_grid


def grid_to_original(gather_grid: np.ndarray,
                       offsets_orig: np.ndarray,
                       dx_target: float = 25.0,
                       x_max: float = 4500.0,
                       traces_orig: np.ndarray = None) -> np.ndarray:
    """
    Map a gather from the common grid back to the original geometry.
    Traces with offset > x_max are taken from traces_orig (if given).
    """
    n_time = gather_grid.shape[0]
    n_orig = len(offsets_orig)
    out = np.zeros((n_time, n_orig), dtype=np.float32)

    abs_offsets = np.abs(offsets_orig)
    for trace_idx, off in enumerate(abs_offsets):
        if off > x_max:
            if traces_orig is not None:
                out[:, trace_idx] = traces_orig[:, trace_idx]
            continue
        grid_idx = int(round(off / dx_target))
        if 0 <= grid_idx < gather_grid.shape[1]:
            out[:, trace_idx] = gather_grid[:, grid_idx]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Load original CROP_M18 SGY
# ─────────────────────────────────────────────────────────────────────────────

def load_original_segy(path: str):
    """
    Load the original CROP_M18 SEG-Y and return:
      - shots: dict {sp_id: {'data': (n_samples, n_traces),
                              'offsets': (n_traces,),
                              'source_x': scalar,
                              'group_x': (n_traces,),
                              'trace_indices': (n_traces,)  -> original trace indices}}
      - dt_us:  sample interval in microseconds
      - n_samples_orig: number of samples per trace in the original file
    """
    shots = {}
    with segyio.open(path, "r", ignore_geometry=True) as f:
        n_traces       = f.tracecount
        n_samples_orig = len(f.trace[0])
        dt_us          = f.bin[segyio.BinField.Interval]

        # Build shot index
        for i in range(n_traces):
            h  = f.header[i]
            sp = h[segyio.TraceField.FieldRecord]
            shots.setdefault(sp, {
                "data":          [],
                "offsets":       [],
                "source_x":      h[segyio.TraceField.SourceX],
                "source_y":      h[segyio.TraceField.SourceY],
                "group_x":       [],
                "trace_indices": [],
            })
            shots[sp]["data"].append(f.trace[i].astype(np.float32))
            shots[sp]["offsets"].append(h[segyio.TraceField.offset])
            shots[sp]["group_x"].append(h[segyio.TraceField.GroupX])
            shots[sp]["trace_indices"].append(i)

        # Convert to arrays
        for sp in shots:
            shots[sp]["data"]          = np.array(shots[sp]["data"]).T    # (n_samples, n_traces)
            shots[sp]["offsets"]       = np.array(shots[sp]["offsets"])
            shots[sp]["group_x"]       = np.array(shots[sp]["group_x"])
            shots[sp]["trace_indices"] = np.array(shots[sp]["trace_indices"])

    return shots, dt_us, n_samples_orig


# ─────────────────────────────────────────────────────────────────────────────
# Write SGY (with trace headers)
# ─────────────────────────────────────────────────────────────────────────────

def write_sgy_grid_version(output_path: str,
                              gather_data_all_shots: np.ndarray,
                              sp_list: list,
                              source_x_per_sp: list,
                              dt_us_out: int,
                              n_grid: int = 181,
                              dx_target: float = 25.0):
    """
    Write the SEG-Y of version A (common grid, 181 traces × 25 m per shot).

    Args:
        gather_data_all_shots: (n_shots, n_time, n_grid)
        sp_list:               (n_shots,) shot point numbers
        source_x_per_sp:       (n_shots,) SourceX per shot
        dt_us_out:             sample interval in microseconds (2000 = 2ms)
        n_grid:                traces per shot (181)
        dx_target:             trace spacing in meters (25)
    """
    n_shots, n_time, n_traces = gather_data_all_shots.shape
    assert n_traces == n_grid

    spec = segyio.spec()
    spec.sorting = 2                                # CDP_X sorting (default)
    spec.format  = 1                                # IBM float
    spec.samples = list(range(n_time))
    spec.tracecount = n_shots * n_grid

    with segyio.create(output_path, spec) as f:
        f.bin[segyio.BinField.Interval]    = dt_us_out
        f.bin[segyio.BinField.Samples]     = n_time
        f.bin[segyio.BinField.Format]      = 1     # IBM float

        trace_count = 0
        for shot_idx, sp in enumerate(sp_list):
            src_x = source_x_per_sp[shot_idx]
            for grid_idx in range(n_grid):
                offset_signed = -int(grid_idx * dx_target)   # negative as in original
                group_x       = src_x + offset_signed

                f.header[trace_count] = {
                    segyio.TraceField.FieldRecord:         int(sp),
                    segyio.TraceField.TraceNumber:         grid_idx + 1,
                    segyio.TraceField.offset:              offset_signed,
                    segyio.TraceField.SourceX:             int(src_x),
                    segyio.TraceField.SourceY:             0,
                    segyio.TraceField.GroupX:              int(group_x),
                    segyio.TraceField.GroupY:              0,
                    segyio.TraceField.SourceGroupScalar:   1,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_us_out,
                    segyio.TraceField.TRACE_SAMPLE_COUNT:    n_time,
                }
                f.trace[trace_count] = gather_data_all_shots[shot_idx, :, grid_idx].astype(np.float32)
                trace_count += 1


def write_sgy_original_geometry(output_path: str,
                                   gather_data_all_shots: np.ndarray,
                                   shots_dict: dict,
                                   sp_list: list,
                                   dt_us_out: int):
    """
    Write the SEG-Y of version B (original CROP_M18 geometry).

    Args:
        gather_data_all_shots: (n_shots, n_time_out, n_orig=180), sampled at 2 ms
        shots_dict:             dict from load_original_segy (header information)
        sp_list:                list of the shot points to write
        dt_us_out:              2000 (2ms)
    """
    n_shots, n_time_out, n_traces_per_shot = gather_data_all_shots.shape

    spec = segyio.spec()
    spec.sorting = 2
    spec.format  = 1
    spec.samples = list(range(n_time_out))
    spec.tracecount = n_shots * n_traces_per_shot

    with segyio.create(output_path, spec) as f:
        f.bin[segyio.BinField.Interval]    = dt_us_out
        f.bin[segyio.BinField.Samples]     = n_time_out
        f.bin[segyio.BinField.Format]      = 1

        trace_count = 0
        for shot_idx, sp in enumerate(sp_list):
            shot_info = shots_dict[sp]
            n_orig = len(shot_info["offsets"])
            for trace_idx in range(n_orig):
                f.header[trace_count] = {
                    segyio.TraceField.FieldRecord:         int(sp),
                    segyio.TraceField.TraceNumber:         trace_idx + 1,
                    segyio.TraceField.offset:              int(shot_info["offsets"][trace_idx]),
                    segyio.TraceField.SourceX:             int(shot_info["source_x"]),
                    segyio.TraceField.SourceY:             int(shot_info["source_y"]),
                    segyio.TraceField.GroupX:              int(shot_info["group_x"][trace_idx]),
                    segyio.TraceField.GroupY:              0,
                    segyio.TraceField.SourceGroupScalar:   1,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_us_out,
                    segyio.TraceField.TRACE_SAMPLE_COUNT:    n_time_out,
                }
                f.trace[trace_count] = gather_data_all_shots[shot_idx, :, trace_idx].astype(np.float32)
                trace_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgy",
                        default=str(DATA_ROOT / "CROP_M18_SP_500_800.sgy"))
    parser.add_argument("--checkpoint",
                        default=str(CKPT_ROOT / "v6_grid25m/within_geology/best.pt"))
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT / "output_for_stack"))
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.3, 0.5])
    parser.add_argument("--mssa-rank", type=int, default=30)
    parser.add_argument("--mssa-iter", type=int, default=2)
    parser.add_argument("--n-iter-pocs", type=int, default=50)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--skip-classical", action="store_true",
                        help="skip POCS and MSSA (DL only; masks unchanged)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_config()
    dc = config.data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pipeline parameters (from config)
    dt_target_ms     = dc.dt_target_ms          # 2.0
    dx_target_m      = dc.dx_target_m            # 25.0
    x_max_m          = dc.x_max_m                # 4500.0
    t_max_s          = dc.t_max_s                # 4.5
    n_time_target    = int(round(t_max_s * 1000.0 / dt_target_ms))  # 2250
    n_grid           = int(round(x_max_m / dx_target_m)) + 1         # 181
    streamer_length  = 4500.0
    geometry_code    = streamer_length / dc.max_streamer_length

    print(f"\n{'='*70}")
    print(f"GATHER EXPORT FOR DOWNSTREAM (CROP_M18)")
    print(f"{'='*70}")
    print(f"Input SGY:     {args.sgy}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Output dir:    {output_dir}")
    print(f"Ratios:        {args.ratios}")
    print(f"MSSA params:   rank={args.mssa_rank}, n_iter={args.mssa_iter}")
    print(f"POCS params:   n_iter={args.n_iter_pocs}")
    print(f"Grid:          {n_grid} traces × {dx_target_m}m")
    print(f"Time:          {n_time_target} samples × {dt_target_ms}ms")
    print(f"Device:        {device}")
    print(f"{'='*70}\n")

    # ─── Step 1: Load original SGY ────────────────────────────────────────
    print("Loading original SGY...")
    shots, dt_orig_us, n_samples_orig = load_original_segy(args.sgy)
    dt_orig_ms = dt_orig_us / 1000.0
    sp_list = sorted(shots.keys())
    print(f"  Found {len(sp_list)} shots, dt_orig={dt_orig_ms}ms, n_samples_orig={n_samples_orig}")

    # ─── Step 2: Load DL model ────────────────────────────────────────────
    print("\nLoading DL model...")
    model, ckpt = load_checkpoint(args.checkpoint, config, device)
    model.eval()
    print(f"  Loaded checkpoint, val_snr = {ckpt.get('val_snr_db', '?')}\n")

    # ─── Step 3: Process each shot, for each ratio ────────────────────────
    n_shots = len(sp_list)
    rng_global = np.random.default_rng(args.mask_seed)

    # Pre-allocate output arrays per ratio
    output_arrays = {}
    for ratio in args.ratios:
        key = f"{int(ratio*100)}"
        output_arrays[key] = {
            # Version A: common grid
            "A_GT":     np.zeros((n_shots, n_time_target, n_grid), dtype=np.float32),
            "A_masked": np.zeros((n_shots, n_time_target, n_grid), dtype=np.float32),
            "A_DL":     np.zeros((n_shots, n_time_target, n_grid), dtype=np.float32),
            "A_POCS":   np.zeros((n_shots, n_time_target, n_grid), dtype=np.float32),
            "A_MSSA":   np.zeros((n_shots, n_time_target, n_grid), dtype=np.float32),
        }

    source_x_per_sp = [shots[sp]["source_x"] for sp in sp_list]

    print("Processing shots...")
    t_global = time.time()
    for shot_idx, sp in enumerate(sp_list):
        shot = shots[sp]
        traces_orig = shot["data"]             # (n_samples_orig, n_orig)
        offsets_orig = shot["offsets"]          # negative offsets

        # Resample to dt_target_ms = 2ms (from dt_orig_ms = 4ms typically)
        if abs(dt_orig_ms - dt_target_ms) > 1e-3:
            up   = int(round(dt_orig_ms / dt_target_ms))
            down = 1
            traces_resampled = resample_poly(traces_orig, up, down, axis=0)
        else:
            traces_resampled = traces_orig

        # Trim/pad to n_time_target
        if traces_resampled.shape[0] >= n_time_target:
            traces_resampled = traces_resampled[:n_time_target, :]
        else:
            pad_n = n_time_target - traces_resampled.shape[0]
            traces_resampled = np.pad(traces_resampled,
                                         ((0, pad_n), (0, 0)),
                                         mode="constant")

        # Map original geometry → common grid (181 × 25m)
        gather_grid = original_to_grid(traces_resampled, offsets_orig,
                                          dx_target=dx_target_m, x_max=x_max_m)

        # Preprocess + save scale factors
        gather_norm, gather_clip, scale_factors = preprocess_and_save_scales(
            gather_grid, dt_ms=dt_target_ms,
            apply_filter=dc.apply_filter,
            filter_f_cut=dc.filter_f_cut,
            clip_sigma=dc.clip_sigma,
        )

        # Determine validity (where there are physical receivers)
        validity = (scale_factors > 0).astype(np.float32)
        n_valid = int(validity.sum())

        if n_valid < 10:
            print(f"  Shot {sp}: too few valid traces ({n_valid}), skipping")
            continue

        # GT (version A): preprocessed and clipped data, NOT normalized
        # (keeps the relative amplitudes between traces)
        gt_grid = gather_clip.copy()

        # For each ratio
        for ratio in args.ratios:
            key = f"{int(ratio*100)}"

            # Generate mask (shot-specific seed for reproducibility)
            shot_seed = args.mask_seed + sp
            rng_shot  = np.random.default_rng(shot_seed)
            valid_idx_pos = np.where(validity > 0.5)[0]
            n_remove = int(len(valid_idx_pos) * ratio)
            mask_positions = rng_shot.choice(valid_idx_pos, size=n_remove,
                                                replace=False)
            trace_mask = np.ones(n_grid, dtype=np.float32)
            trace_mask[mask_positions] = 0.0

            # Apply mask to normalized version (for the model input)
            gather_norm_masked = gather_norm * trace_mask[None, :]

            # Run DL inference
            x_in = np.stack([gather_norm_masked, trace_mask[None, :].repeat(n_time_target, axis=0)], axis=0)
            x_torch = torch.from_numpy(x_in[None]).float().to(device)
            g_torch = torch.tensor([geometry_code], device=device).float()
            with torch.no_grad():
                dl_pred_norm = model(x_torch, g_torch).cpu().squeeze().numpy()

            if args.skip_classical:
                pocs_pred_norm = np.zeros_like(dl_pred_norm)
                mssa_pred_norm = np.zeros_like(dl_pred_norm)
            else:
                # Run POCS
                pocs_pred_norm = pocs_interpolate_validity_aware(
                    gather_norm_masked, trace_mask, validity,
                    n_iter=args.n_iter_pocs,
                )

                # Run MSSA
                mssa_pred_norm = mssa_interpolate_validity_aware(
                    gather_norm_masked, trace_mask, validity,
                    rank=args.mssa_rank, n_iter=args.mssa_iter,
                )

            # Denormalize predictions using smart scale factors
            dl_pred_denorm   = smart_denormalize(dl_pred_norm,   scale_factors, trace_mask)
            pocs_pred_denorm = smart_denormalize(pocs_pred_norm, scale_factors, trace_mask)
            mssa_pred_denorm = smart_denormalize(mssa_pred_norm, scale_factors, trace_mask)

            # Trace replacement: known traces from gather_clip,
            # missing traces from prediction (denormalized)
            dl_recon   = trace_mask[None, :] * gather_clip + (1 - trace_mask)[None, :] * dl_pred_denorm
            pocs_recon = trace_mask[None, :] * gather_clip + (1 - trace_mask)[None, :] * pocs_pred_denorm
            mssa_recon = trace_mask[None, :] * gather_clip + (1 - trace_mask)[None, :] * mssa_pred_denorm

            # "masked" version (the model input, denormalized, zeros on the missing traces)
            masked_grid = trace_mask[None, :] * gather_clip

            # Store
            output_arrays[key]["A_GT"][shot_idx]     = gt_grid
            output_arrays[key]["A_masked"][shot_idx] = masked_grid
            output_arrays[key]["A_DL"][shot_idx]     = dl_recon
            output_arrays[key]["A_POCS"][shot_idx]   = pocs_recon
            output_arrays[key]["A_MSSA"][shot_idx]   = mssa_recon

        if (shot_idx + 1) % 30 == 0:
            elapsed = time.time() - t_global
            eta = elapsed / (shot_idx + 1) * (n_shots - shot_idx - 1)
            print(f"  [{shot_idx+1}/{n_shots}] elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    print(f"\nProcessing done in {(time.time()-t_global)/60:.1f} min\n")

    # ─── Step 4: Export SGY (Version A: common grid) ──────────────────────
    names_out = (["GT", "masked", "DL"] if args.skip_classical
                 else ["GT", "masked", "DL", "POCS", "MSSA"])
    print("Exporting Version A (common grid 181 × 25m)...")
    dt_us_out = int(round(dt_target_ms * 1000))     # 2000 us

    for ratio in args.ratios:
        key = f"{int(ratio*100)}"
        sub_dir = output_dir / f"A_{key}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        for name in names_out:
            data = output_arrays[key][f"A_{name}"]
            out_path = sub_dir / f"CROP_M18_{name}.sgy"
            write_sgy_grid_version(
                str(out_path), data, sp_list, source_x_per_sp,
                dt_us_out=dt_us_out, n_grid=n_grid, dx_target=dx_target_m,
            )
            print(f"  Written: {out_path}  ({data.shape})")

    # ─── Step 5: Export SGY (Version B: original geometry) ─────────────────
    print("\nExporting Version B (original geometry, 180 traces)...")
    for ratio in args.ratios:
        key = f"{int(ratio*100)}"
        sub_dir = output_dir / f"B_{key}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Convert from grid to original geometry for each method
        n_orig_per_shot = len(shots[sp_list[0]]["offsets"])
        for name in names_out:
            grid_data = output_arrays[key][f"A_{name}"]
            orig_data = np.zeros((n_shots, n_time_target, n_orig_per_shot),
                                    dtype=np.float32)
            for shot_idx, sp in enumerate(sp_list):
                offsets_this = shots[sp]["offsets"]
                # Original traces (to keep the traces outside the common grid)
                traces_orig_resampled = None
                if name == "GT":
                    # For GT, use the original raw resampled data for off-grid traces
                    traces_orig_raw = shots[sp]["data"]
                    if abs(dt_orig_ms - dt_target_ms) > 1e-3:
                        up   = int(round(dt_orig_ms / dt_target_ms))
                        traces_orig_resampled = resample_poly(
                            traces_orig_raw, up, 1, axis=0,
                        )[:n_time_target, :]
                    else:
                        traces_orig_resampled = traces_orig_raw[:n_time_target, :]

                orig_data[shot_idx] = grid_to_original(
                    grid_data[shot_idx], offsets_this,
                    dx_target=dx_target_m, x_max=x_max_m,
                    traces_orig=traces_orig_resampled,
                )

            out_path = sub_dir / f"CROP_M18_{name}.sgy"
            write_sgy_original_geometry(
                str(out_path), orig_data, shots, sp_list, dt_us_out=dt_us_out,
            )
            print(f"  Written: {out_path}  ({orig_data.shape})")

    print(f"\n{'='*70}")
    print(f"DONE. All gathers exported to {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
