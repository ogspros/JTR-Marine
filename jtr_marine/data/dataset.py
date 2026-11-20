"""
jtr_marine/data/dataset.py — PyTorch Dataset for seismic gather reconstruction.

Handles:
- Loading real seismic data (SEG-Y via segyio or numpy .npy)
- Resampling to common physical grid (dt_target, dx_target, t_max, x_max)
- Offset sign correction (marine data: negative offsets -> absolute value)
- Preprocessing: clip, normalize per trace
- Three masking strategies: random, block, far-offset systematic
- Mixed masking for robust training

The field lines and their streamer lengths (1500, 3000, 4500 m) are listed in
config.py (DataConfig.dataset_paths / streamer_lengths).

Common physical grid after resampling (config.py values):
  dt=2ms, T=4.5s    -> 2250 time samples
  dx=25m, X=4500m   -> 181 traces (offset 0-4500m)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict
from math import gcd
from scipy.signal import butter, sosfilt


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing utilities
# ─────────────────────────────────────────────────────────────────────────────

def highpass_filter(gather: np.ndarray,
                    f_cut: float = 8.0,
                    dt_ms: float = 2.0) -> np.ndarray:
    """
    4th-order Butterworth high-pass filter applied along the time axis.
    Removes swell noise and other low-frequency components.

    Args:
        gather: (n_time, n_offsets)
        f_cut:  cut-off frequency in Hz (e.g. 8.0)
        dt_ms:  sampling interval in ms (e.g. 2.0)

    Returns:
        Filtered gather, same shape as the input.
    """
    fs  = 1000.0 / dt_ms                        # sampling frequency in Hz
    sos = butter(4, f_cut, btype="highpass",
                 fs=fs, output="sos")
    filtered = sosfilt(sos, gather, axis=0)      # filter along the time axis
    return filtered.astype(np.float32)


def clip_and_normalize(gather: np.ndarray,
                       clip_sigma: float = 3.0,
                       apply_filter: bool = False,
                       filter_f_cut: float = 8.0,
                       dt_ms: float = 2.0) -> np.ndarray:
    """
    Full preprocessing of one seismic gather:
      1. (optional) high-pass filter to remove swell noise
      2. amplitude clipping at +/- N sigma per trace
      3. per-trace normalization to [-1, 1]

    Args:
        gather:        (n_time, n_offsets) raw amplitudes
        clip_sigma:    clipping threshold in units of sigma
        apply_filter:  if True, apply the high-pass filter before clipping
        filter_f_cut:  filter cut-off frequency in Hz
        dt_ms:         sampling interval in ms

    Returns:
        Preprocessed gather in [-1, 1].
        All-zero traces (padding or dead channels) stay zero.
    """
    gather = gather.copy().astype(np.float32)

    # Step 1: high-pass filter (swell noise removal)
    if apply_filter:
        gather = highpass_filter(gather, filter_f_cut, dt_ms)

    # Steps 2-3: per-trace clipping and normalization
    for i in range(gather.shape[1]):
        trace = gather[:, i]
        if np.all(trace == 0):
            continue
        sigma = np.std(trace)
        if sigma < 1e-10:
            continue
        trace        = np.clip(trace, -clip_sigma * sigma, clip_sigma * sigma)
        max_amp      = np.max(np.abs(trace)) + 1e-10
        gather[:, i] = trace / max_amp
    return gather


# ─────────────────────────────────────────────────────────────────────────────
# Masking strategies
# ─────────────────────────────────────────────────────────────────────────────

def mask_random(n_offsets: int, valid_offsets: int,
                ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly remove a fraction of valid offset traces."""
    mask     = np.ones(n_offsets, dtype=np.float32)
    n_remove = int(valid_offsets * ratio)
    indices  = rng.choice(valid_offsets, size=n_remove, replace=False)
    mask[indices] = 0.0
    return mask


def mask_block(n_offsets: int, valid_offsets: int, ratio: float,
               block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Remove contiguous blocks of offset traces (simulates acquisition gaps)."""
    mask     = np.ones(n_offsets, dtype=np.float32)
    n_remove = int(valid_offsets * ratio)
    n_blocks = max(1, n_remove // block_size)
    for _ in range(n_blocks):
        start = rng.integers(0, max(1, valid_offsets - block_size))
        mask[start:start + block_size] = 0.0
    return mask


def mask_far_offsets(n_offsets: int, valid_offsets: int,
                     ratio: float) -> np.ndarray:
    """
    Remove far (last) offsets systematically.
    Physically most meaningful: simulates a shorter streamer.
    Used as the primary test-time masking strategy.
    """
    mask   = np.ones(n_offsets, dtype=np.float32)
    cutoff = valid_offsets - int(valid_offsets * ratio)
    mask[cutoff:valid_offsets] = 0.0
    return mask


def apply_mask(gather: np.ndarray, validity: np.ndarray,
               masking_strategy: str, ratio: float,
               block_size: int, rng: np.random.Generator
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply chosen masking strategy to a preprocessed gather.

    Returns:
        masked_gather: gather with removed traces set to zero
        trace_mask:    1D (n_offsets,) -- 1=present, 0=removed
    """
    n_offsets     = gather.shape[1]
    valid_offsets = int(validity[0].sum())

    if masking_strategy == "random":
        trace_mask = mask_random(n_offsets, valid_offsets, ratio, rng)
    elif masking_strategy == "block":
        trace_mask = mask_block(n_offsets, valid_offsets, ratio, block_size, rng)
    elif masking_strategy == "far":
        trace_mask = mask_far_offsets(n_offsets, valid_offsets, ratio)
    elif masking_strategy == "mixed":
        choice = rng.integers(0, 3)
        if choice == 0:
            trace_mask = mask_random(n_offsets, valid_offsets, ratio, rng)
        elif choice == 1:
            trace_mask = mask_block(n_offsets, valid_offsets, ratio, block_size, rng)
        else:
            trace_mask = mask_far_offsets(n_offsets, valid_offsets, ratio)
    else:
        raise ValueError(f"Unknown masking strategy: {masking_strategy}")

    full_mask     = np.tile(trace_mask[np.newaxis, :], (gather.shape[0], 1))
    masked_gather = gather * full_mask
    return masked_gather, trace_mask


# ─────────────────────────────────────────────────────────────────────────────
# SEG-Y loading with physical grid resampling
# ─────────────────────────────────────────────────────────────────────────────

def load_segy(path: Path,
              dt_target_ms: float,
              t_max_s: float,
              x_max_m: float,
              dx_target_m: float) -> np.ndarray:
    """
    Load a marine SEG-Y file and resample every shot gather onto the common
    physical grid.

    Handles:
      - Negative offsets (marine source->receiver convention) -> absolute value
      - Different dt per dataset -> resample_poly (built-in anti-alias filter)
      - Different dx per dataset -> linear interpolation onto the target grid
      - Different record length -> crop (or zero-pad) to t_max_s

    Args:
        path:          path to the .segy / .sgy file
        dt_target_ms:  target sampling interval in ms (2.0 in config.py)
        t_max_s:       target record length in seconds (4.5 in config.py)
        x_max_m:       target maximum offset in metres (4500.0 in config.py)
        dx_target_m:   target receiver spacing in metres (25.0 in config.py)

    Returns:
        gathers: (n_shots, n_time, n_offsets) float32
                 n_time    = round(t_max_s * 1000 / dt_target_ms)  (2250 with config.py)
                 n_offsets = round(x_max_m / dx_target_m) + 1      (181 with config.py)
    """
    try:
        import segyio
    except ImportError:
        raise ImportError(
            "segyio is required to load SEG-Y files.\n"
            "Install it with: pip install segyio"
        )
    from scipy.signal import resample_poly
    from scipy.interpolate import interp1d

    # Accept both str and Path
    path = Path(path)

    # Target grid size
    n_t         = int(round(t_max_s * 1000.0 / dt_target_ms))    # 2250 with config.py
    n_x         = int(round(x_max_m / dx_target_m)) + 1           # 181 with config.py
    offset_grid = np.linspace(0.0, x_max_m, n_x, dtype=np.float32)

    with segyio.open(str(path), ignore_geometry=True) as f:
        dt_ms   = segyio.dt(f) / 1000.0
        n_samp  = f.samples.size
        n_trace = f.tracecount

        print(f"\n  {path.name}")
        print(f"  dt={dt_ms:.2f}ms | {n_samp} samples ({n_samp*dt_ms/1000:.2f}s)"
              f" | {n_trace} traces")

        # Headers: FFID (shot id) and absolute offset
        ffids   = np.array([f.header[i][segyio.TraceField.FieldRecord]
                            for i in range(n_trace)])
        offsets = np.abs(np.array([f.header[i][segyio.TraceField.offset]
                                   for i in range(n_trace)], dtype=np.float32))

        # All traces: (n_trace, n_samp)
        all_traces = np.array([f.trace[i] for i in range(n_trace)],
                               dtype=np.float32)

    # ── 1. Time resampling ────────────────────────────────────────────────
    if abs(dt_ms - dt_target_ms) > 0.01:
        ratio_num  = int(round(dt_target_ms * 1000))
        ratio_den  = int(round(dt_ms        * 1000))
        g          = gcd(ratio_num, ratio_den)
        up, down   = ratio_den // g, ratio_num // g
        all_traces = resample_poly(all_traces, up, down, axis=1).astype(np.float32)
        print(f"  Time resampling: {dt_ms}ms -> {dt_target_ms}ms "
              f"(up={up}, down={down})")

    # ── 2. Crop to t_max_s ────────────────────────────────────────────────
    n_t_file = all_traces.shape[1]
    if n_t_file >= n_t:
        all_traces = all_traces[:, :n_t]
    else:
        pad        = np.zeros((all_traces.shape[0], n_t - n_t_file), dtype=np.float32)
        all_traces = np.concatenate([all_traces, pad], axis=1)
        print(f"  WARNING: file shorter than {t_max_s}s, zero-padded")

    # ── 3. Group by shot and resample along the offset axis ───────────────
    shot_ids = np.unique(ffids)
    n_shots  = len(shot_ids)
    gathers  = np.zeros((n_shots, n_t, n_x), dtype=np.float32)

    for j, sid in enumerate(shot_ids):
        mask         = ffids == sid
        shot_traces  = all_traces[mask]      # (n_rec, n_t)
        shot_offsets = offsets[mask]

        # Sort by increasing offset (near -> far)
        sort_idx     = np.argsort(shot_offsets)
        shot_traces  = shot_traces[sort_idx]
        shot_offsets = shot_offsets[sort_idx]

        # Remove duplicate offsets
        _, unique_idx = np.unique(shot_offsets, return_index=True)
        shot_traces   = shot_traces[unique_idx]
        shot_offsets  = shot_offsets[unique_idx]

        # Interpolate onto the target grid within the range covered by the streamer
        near      = shot_offsets.min()
        far       = shot_offsets.max()
        valid_idx = (offset_grid >= near) & (offset_grid <= far)

        if valid_idx.sum() > 1 and len(shot_offsets) >= 2:
            interp_fn = interp1d(
                shot_offsets, shot_traces,
                axis=0, kind="linear",
                bounds_error=False, fill_value=0.0,
            )
            # interp_fn returns (n_valid, n_t); the mixed basic/boolean index
            # on the left-hand side expects exactly that layout
            gathers[j, :, valid_idx] = interp_fn(offset_grid[valid_idx])

    n_valid_mean = int(((offset_grid >= offsets.min()) &
                        (offset_grid <= offsets.max())).sum())
    print(f"  -> {n_shots} shot gathers | final shape: {gathers.shape}")
    print(f"  -> Valid traces/shot: ~{n_valid_mean} of {n_x} "
          f"({100*n_valid_mean/n_x:.0f}% of the grid)")

    return gathers   # (n_shots, n_t, n_x)


def load_numpy(path: Path) -> np.ndarray:
    """
    Load a pre-processed numpy array.
    Expected layout: (n_shots, n_time, n_offsets).
    """
    data = np.load(path)
    if data.ndim == 2:
        data = data[np.newaxis]
    return data.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SeismicGatherDataset(Dataset):
    """
    Dataset for a single line (one geological setting / streamer length).

    Each sample returns:
        input:      (2, T, O) -- ch0: masked gather, ch1: binary mask
        target:     (1, T, O) -- complete gather (ground truth)
        geometry:   scalar    -- streamer length normalized to [0, 1]
        trace_mask: (O,)      -- 1=present, 0=removed by the mask
        validity:   (O,)      -- 1=real data, 0=padding (short streamer)
        ratio:      scalar    -- fraction of traces removed in this sample

    With config.py:
    T = n_time_samples = 2250  (4.5s at 2ms)
    O = n_max_offsets  = 181   (0-4500m at 25m)
    """

    def __init__(self,
                 data_path: str,
                 streamer_length: float,
                 max_streamer_length: float,
                 # Common physical grid
                 dt_target_ms: float,
                 t_max_s: float,
                 x_max_m: float,
                 dx_target_m: float,
                 # Masking
                 masking_ratios: List[float],
                 masking_strategy: str = "mixed",
                 block_size: int = 8,
                 # Preprocessing
                 clip_sigma: float = 3.0,
                 normalize_per_trace: bool = True,
                 apply_filter: bool = True,
                 filter_f_cut: float = 8.0,
                 # Split
                 split: str = "train",
                 val_fraction: float = 0.15,
                 random_seed: int = 42):

        self.streamer_length     = streamer_length
        self.geometry_code       = streamer_length / max_streamer_length
        self.masking_ratios      = masking_ratios
        self.masking_strategy    = masking_strategy
        self.block_size          = block_size
        self.clip_sigma          = clip_sigma
        self.normalize_per_trace = normalize_per_trace
        self.apply_filter        = apply_filter
        self.filter_f_cut        = filter_f_cut
        self.split               = split

        # Target physical grid (also used by load_segy)
        self.dt_target_ms = dt_target_ms
        self.t_max_s      = t_max_s
        self.x_max_m      = x_max_m
        self.dx_target_m  = dx_target_m
        self.n_time       = int(round(t_max_s * 1000.0 / dt_target_ms))  # 2250 with config.py
        self.n_offsets    = int(round(x_max_m / dx_target_m)) + 1         # 181 with config.py

        # Load the data (kept in RAM to avoid re-reading from disk)
        self.gathers = self._load_data(data_path)

        # Preprocess and cache all gathers in RAM
        # (n_shots x n_time x n_offsets x 4 bytes per line).
        # This removes the I/O bottleneck during training.
        print(f"  Preprocessing and caching {len(self.gathers)} gathers in RAM...")
        self.gathers = [
            clip_and_normalize(
                g,
                clip_sigma    = self.clip_sigma,
                apply_filter  = self.apply_filter,
                filter_f_cut  = self.filter_f_cut,
                dt_ms         = self.dt_target_ms,
            ) if self.normalize_per_trace else g.astype(np.float32)
            for g in self.gathers
        ]
        print(f"  Cache OK — {len(self.gathers)} gathers ready")

        # CONTIGUOUS train/val split (val = last gathers, no leakage)


        n_total = len(self.gathers)


        n_val   = max(1, int(n_total * val_fraction))


        all_idx = np.arange(n_total, dtype=np.int64)


        # val = the last n_val gathers, train = everything else


        self.indices = all_idx[-n_val:] if split == "val" else all_idx[:-n_val]
        self._base_rng = np.random.default_rng(random_seed)

    def _load_data(self, data_path: str) -> List[np.ndarray]:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {data_path}\n"
                f"Supported formats: .segy, .sgy, .npy, .npz"
            )
        if path.suffix in (".sgy", ".segy"):
            data = load_segy(
                path,
                dt_target_ms = self.dt_target_ms,
                t_max_s      = self.t_max_s,
                x_max_m      = self.x_max_m,
                dx_target_m  = self.dx_target_m,
            )
        elif path.suffix == ".npy":
            data = load_numpy(path)
        elif path.suffix == ".npz":
            data = np.load(path)["gathers"].astype(np.float32)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        return [data[i] for i in range(data.shape[0])]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gather_raw = self.gathers[self.indices[idx]]   # (n_time, n_offsets)

        # Preprocessing was already applied at load time (RAM cache)
        gather = gather_raw

        # Validity mask: 1 where there is real data (not short-streamer padding)
        validity    = (np.abs(gather).max(axis=0) > 0).astype(np.float32)  # (O,)
        validity_2d = np.tile(validity[np.newaxis, :], (self.n_time, 1))    # (T, O)

        # Random masking for this sample
        rng   = np.random.default_rng(self._base_rng.integers(0, 2**31) + idx)
        ratio = rng.choice(self.masking_ratios)

        masked_gather, trace_mask = apply_mask(
            gather, validity_2d,
            self.masking_strategy, ratio,
            self.block_size, rng,
        )

        # 2-channel input: [masked gather, 2D binary mask]
        trace_mask_2d = np.tile(trace_mask[np.newaxis, :], (self.n_time, 1))
        model_input   = np.stack([masked_gather, trace_mask_2d], axis=0)  # (2, T, O)
        target        = gather[np.newaxis, :, :]                           # (1, T, O)

        return {
            "input":      torch.from_numpy(model_input),
            "target":     torch.from_numpy(target),
            "geometry":   torch.tensor(self.geometry_code, dtype=torch.float32),
            "trace_mask": torch.from_numpy(trace_mask),
            "validity":   torch.from_numpy(validity),
            "ratio":      torch.tensor(ratio, dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight dataset sharing data already loaded in RAM
# ─────────────────────────────────────────────────────────────────────────────

class SharedGatherDataset(Dataset):
    """
    Dataset operating on a subset of indices of an already loaded array.
    Avoids re-loading the SEG-Y file for separate train and val splits.
    Receives gathers that are already preprocessed (filtered and normalized).
    """
    def __init__(self,
                 gathers: list,
                 indices: np.ndarray,
                 geometry_code: float,
                 n_time: int,
                 n_offsets: int,
                 masking_ratios: List[float],
                 masking_strategy: str = "mixed",
                 block_size: int = 8,
                 random_seed: int = 42,
                 training: bool = False,
                 enable_flip: bool = False,
                 enable_noise: bool = False,
                 noise_snr_min: float = 25.0,
                 noise_snr_max: float = 25.0):
        self.gathers          = gathers
        self.indices          = indices
        self.geometry_code    = geometry_code
        self.n_time           = n_time
        self.n_offsets        = n_offsets
        self.masking_ratios   = masking_ratios
        self.masking_strategy = masking_strategy
        self.block_size       = block_size
        self._base_rng        = np.random.default_rng(random_seed)
        self.training         = training      # True for train, False for val/test
        self.enable_flip      = enable_flip   # enable horizontal flip in training
        # Gaussian noise injection (training only)
        self.enable_noise     = enable_noise
        self.noise_snr_min    = noise_snr_min
        self.noise_snr_max    = noise_snr_max

    def set_masking_ratios(self, masking_ratios):
        """Change the masking ratios during training (curriculum learning)."""
        self.masking_ratios = list(masking_ratios)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gather = self.gathers[self.indices[idx]]   # already preprocessed

        # rng for this specific idx
        rng = np.random.default_rng(self._base_rng.integers(0, 2**31) + idx)

        # Horizontal flip (50%) — training only, before everything else
        if self.training and self.enable_flip and rng.random() < 0.5:
            gather = gather[:, ::-1].copy()  # flip along the offset axis

        # Gaussian noise injection (training only).
        # Noise is added to the CLEAN gather, so it enters both input and target
        # (data augmentation: same task, many noisy versions of the same data)
        if self.training and self.enable_noise:
            snr_db = rng.uniform(self.noise_snr_min, self.noise_snr_max)
            # RMS computed on the valid traces only (padding excluded)
            valid_traces = np.abs(gather).max(axis=0) > 0   # (O,)
            if valid_traces.any():
                signal_rms = np.sqrt((gather[:, valid_traces] ** 2).mean()) + 1e-9
                noise_std = signal_rms * (10 ** (-snr_db / 20))
                noise = rng.normal(0, noise_std, gather.shape).astype(gather.dtype)
                # Keep zero where it was zero (padding zones)
                noise = noise * valid_traces[np.newaxis, :].astype(gather.dtype)
                gather = (gather + noise).copy()

        # Validity mask (after the optional flip)
        validity    = (np.abs(gather).max(axis=0) > 0).astype(np.float32)
        validity_2d = np.tile(validity[np.newaxis, :], (self.n_time, 1))

        # Masking
        ratio = rng.choice(self.masking_ratios)

        masked_gather, trace_mask = apply_mask(
            gather, validity_2d,
            self.masking_strategy, ratio,
            self.block_size, rng,
        )

        trace_mask_2d = np.tile(trace_mask[np.newaxis, :], (self.n_time, 1))
        model_input   = np.stack([masked_gather, trace_mask_2d], axis=0)
        target        = gather[np.newaxis, :, :]

        return {
            "input":      torch.from_numpy(model_input),
            "target":     torch.from_numpy(target),
            "geometry":   torch.tensor(self.geometry_code, dtype=torch.float32),
            "trace_mask": torch.from_numpy(trace_mask),
            "validity":   torch.from_numpy(validity),
            "ratio":      torch.tensor(ratio, dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-geometry leave-one-out fold builder
# ─────────────────────────────────────────────────────────────────────────────

def build_fold_datasets(config, test_fold_idx: int):
    """
    Train / val / test datasets for one leave-one-LINE-out fold.

    test_fold_idx is an index into config.data.dataset_paths: the fold holds
    out ONE LINE, not one geometry. With the six-line configuration of the
    paper and n_folds = 3 (train.py), the folds hold out
        0 -> SISOC-MEPA (1500 m), 1 -> CG-1 (1500 m), 2 -> IT90-65 (3000 m),
    and the other line of the same geometry stays in training.

    Splits: for every training line, train = first 85 %, val = last 15 %
    (so the first internal-test gather, index 255 of 301, is in training);
    test = last 15 % of the held-out line, with "mixed" masking.

    For a leave-one-GEOMETRY-out experiment use
    build_within_geology_datasets(config, exclude_streamer_lengths=[L])
    (train.py --holdout-geometry L).
    """
    from torch.utils.data import ConcatDataset

    dc = config.data

    n_t = int(round(dc.t_max_s * 1000.0 / dc.dt_target_ms))
    n_x = int(round(dc.x_max_m / dc.dx_target_m)) + 1

    shared_kwargs = dict(
        n_time         = n_t,
        n_offsets      = n_x,
        masking_ratios = dc.masking_ratios,
        block_size     = dc.block_size,
        random_seed    = dc.random_seed,
    )

    load_kwargs = dict(
        max_streamer_length = dc.max_streamer_length,
        dt_target_ms        = dc.dt_target_ms,
        t_max_s             = dc.t_max_s,
        x_max_m             = dc.x_max_m,
        dx_target_m         = dc.dx_target_m,
        masking_ratios      = dc.masking_ratios,
        masking_strategy    = dc.masking_strategy,
        block_size          = dc.block_size,
        clip_sigma          = dc.clip_sigma,
        normalize_per_trace = dc.normalize_per_trace,
        apply_filter        = dc.apply_filter,
        filter_f_cut        = dc.filter_f_cut,
        val_fraction        = dc.val_fraction,
        random_seed         = dc.random_seed,
        split               = "train",
    )

    train_sets, val_sets = [], []
    test_dataset = None

    for i, (path, length) in enumerate(zip(dc.dataset_paths, dc.streamer_lengths)):
        # Load the file ONCE — train and val share the data
        tmp = SeismicGatherDataset(
            data_path       = path,
            streamer_length = length,
            **load_kwargs,
        )
        all_gathers = tmp.gathers
        n_total     = len(all_gathers)
        n_val   = max(1, int(n_total * dc.val_fraction))

        # CONTIGUOUS split: val = last n_val gathers, train = the first ones

        all_idx = np.arange(n_total, dtype=np.int64)
        val_idx   = all_idx[-n_val:]

        train_idx = all_idx[:-n_val]
        geom_code   = length / dc.max_streamer_length

        if i == test_fold_idx:
            test_dataset = SharedGatherDataset(
                gathers          = all_gathers,
                indices          = val_idx,
                geometry_code    = geom_code,
                masking_strategy = "mixed",
                **shared_kwargs,
            )
            continue

        train_sets.append(SharedGatherDataset(
            gathers          = all_gathers,
            indices          = train_idx,
            geometry_code    = geom_code,
            masking_strategy = dc.masking_strategy,
            **shared_kwargs,
        ))
        val_sets.append(SharedGatherDataset(
            gathers          = all_gathers,
            indices          = val_idx,
            geometry_code    = geom_code,
            masking_strategy = dc.masking_strategy,
            **shared_kwargs,
        ))

    return ConcatDataset(train_sets), ConcatDataset(val_sets), test_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Within-geology setup: all lines contribute to train, val and test
# ─────────────────────────────────────────────────────────────────────────────

def build_within_geology_datasets(config, exclude_streamer_lengths=None):
    """
    Within-geology setup: loads ALL lines of config.data.dataset_paths and
    splits each of them contiguously (along the shot axis) into
    train/val/test 70/15/15; the per-line splits are then concatenated.

    Differences from build_fold_datasets (cross-geology):
      - No line is held out for testing
      - Every line contributes to train, val and test
      - The model sees examples of every geometry during training

    Research question: "does the model learn to reconstruct missing offsets
    regardless of the acquisition geometry?"
    (As opposed to cross-geology: "does it generalize to unseen geologies?")

    Each gather keeps its own geometry_code, so the FiLM conditioning still
    learns to modulate according to the streamer length.

    exclude_streamer_lengths (optional, default None = paper setup): lines
        whose streamer length is in this list are skipped entirely (no train,
        val or test gathers). Used for the leave-one-geometry-out experiment
        (train.py --holdout-geometry); the contiguous 70/15/15 split of the
        remaining lines is unchanged, so their internal test gathers are
        never seen in training.

    Returns:
        train_dataset, val_dataset, test_dataset (ConcatDataset)
    """
    from torch.utils.data import ConcatDataset

    dc = config.data
    exclude = {float(x) for x in (exclude_streamer_lengths or [])}

    n_t = int(round(dc.t_max_s * 1000.0 / dc.dt_target_ms))
    n_x = int(round(dc.x_max_m / dc.dx_target_m)) + 1

    # Split fractions: 70% train, 15% val, 15% test
    train_frac = 0.70
    val_frac   = 0.15
    # test_frac = 0.15 (the remainder)

    train_sets, val_sets, test_sets = [], [], []

    for i, (path, length) in enumerate(zip(dc.dataset_paths, dc.streamer_lengths)):
        if float(length) in exclude:
            print(f"  {Path(path).name} ({length:.0f}m): excluded (held-out geometry)")
            continue
        # Load once
        full_ds = SeismicGatherDataset(
            data_path           = path,
            streamer_length     = length,
            max_streamer_length = dc.max_streamer_length,
            dt_target_ms        = dc.dt_target_ms,
            t_max_s             = dc.t_max_s,
            x_max_m             = dc.x_max_m,
            dx_target_m         = dc.dx_target_m,
            masking_ratios      = dc.masking_ratios,
            masking_strategy    = dc.masking_strategy,
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

        # CONTIGUOUS 70/15/15 split (no spatial leakage)


        all_idx = np.arange(n_total, dtype=np.int64)
        n_train = int(n_total * train_frac)
        n_val   = int(n_total * val_frac)

        train_idx = all_idx[:n_train]
        val_idx   = all_idx[n_train:n_train + n_val]
        test_idx  = all_idx[n_train + n_val:]

        geom_code = length / dc.max_streamer_length

        shared_kwargs = dict(
            n_time         = n_t,
            n_offsets      = n_x,
            masking_ratios = dc.masking_ratios,
            block_size     = dc.block_size,
            random_seed    = dc.random_seed,
            geometry_code  = geom_code,
        )

        train_sets.append(SharedGatherDataset(
            gathers          = all_gathers, indices = train_idx,
            masking_strategy = dc.masking_strategy, **shared_kwargs,
        ))
        val_sets.append(SharedGatherDataset(
            gathers          = all_gathers, indices = val_idx,
            masking_strategy = dc.masking_strategy, **shared_kwargs,
        ))
        test_sets.append(SharedGatherDataset(
            gathers          = all_gathers, indices = test_idx,
            masking_strategy = "mixed", **shared_kwargs,
        ))

        print(f"  {Path(path).name} ({length:.0f}m): "
              f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return (
        ConcatDataset(train_sets),
        ConcatDataset(val_sets),
        ConcatDataset(test_sets),
    )
