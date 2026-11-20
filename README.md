# JTR-Marine

**Joint training across streamer geometries for missing-trace reconstruction: A comparative evaluation on real marine data**

Author and maintainer of this repository: **Giuseppe Brancatelli** (OGS, Trieste) — `gbrancatelli@ogs.it`

This repository holds the code, the reference results and the instructions to
download the trained models of the paper

> Brancatelli G., Lipari V., Forlin E., Pignelli R., Busetti M., Geletti R.,
> *Joint training across streamer geometries for missing-trace reconstruction:
> A comparative evaluation on real marine data*, Journal of Applied Geophysics
> (2026), https://doi.org/10.1016/j.jappgeo.2026.106621

A compact U-Net (8.36 M parameters), conditioned on the streamer length with
FiLM, is trained jointly on six real marine lines acquired with 1500, 3000 and
4500 m streamers. It is compared with POCS, MSSA, the U-Net of Chai et al.
(2020) and several deep-learning variants on held-out parts of the training
lines, on three lines never used in training, and on the Marmousi synthetic.

To reproduce a specific table or figure, go to [REPRODUCE.md](REPRODUCE.md).

## Repository layout

```
config.py                 configuration and paths (DATA_ROOT, CKPT_ROOT, RESULTS_ROOT)
jtr_marine/               package
  data/dataset.py           SEG-Y loading, resampling to the common grid, masking, splits
  models/                   compact U-Net + FiLM and the variants compared in the paper
  losses/                   composite loss (L1 + SSIM + f-k) and the Charbonnier variant
  baselines/                POCS and MSSA
  metrics.py                SNR, SSIM, f-k energy ratio
  model_loader.py           builds the right model from a checkpoint
scripts/
  train/                    training of every model of the paper
  evaluate/                 per-gather evaluation, baselines, parameter sweeps
  stats/                    confidence intervals, paired tests, comparison with references
  figures/                  one script per figure (make_fig<N>_*.py)
  data_prep/                Marmousi conversion, public map layers
  download_checkpoints.sh   fetches the trained models from the GitHub release
data/                     instructions only: no seismic data are in the repository
checkpoints/              instructions and file list; the weights are release assets
reference_results/        per-gather results and statistics behind the tables
tests/                    quick tests on CPU, no data needed
```

## Installation

```bash
git clone https://github.com/ogspros/JTR-Marine.git
cd JTR-Marine
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate jtr-marine
python -m pytest tests -q                # 8 tests, CPU, about a minute
```

The paper results were obtained with Python 3.11 and PyTorch 2.4 on Linux,
with one NVIDIA RTX A6000 (48 GB). Scripts are run from the repository root.

## Data

No seismic data are distributed here. [data/README.md](data/README.md) lists
the files the code expects, where to obtain each line, and the SEG-Y headers
that are read. The folder with the SEG-Y files is set with

```bash
export DATA_ROOT=/path/to/segy      # default: ./data/raw
```

## Trained models

The weights are attached to the GitHub release of this repository, not stored
in git. To download them into `./checkpoints`:

```bash
bash scripts/download_checkpoints.sh            # paper model only (about 100 MB)
bash scripts/download_checkpoints.sh --all      # every model of the paper (several GB)
```

[checkpoints/README.md](checkpoints/README.md) describes each model. A
different folder can be used with `export CKPT_ROOT=/path/to/checkpoints`;
outputs go to `./results` or to `$RESULTS_ROOT`.

## Quick start

```python
import torch
from config import CKPT_ROOT, get_config
from jtr_marine.model_loader import load_checkpoint

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = load_checkpoint(CKPT_ROOT / "v6_grid25m/within_geology/best.pt", config, device)

# x: (batch, 2, n_time, n_offsets) = masked gather and trace mask on the common
#    grid (2 ms, 4.5 s, 25 m); g: (batch,) streamer length / 4500 (0.333, 0.667, 1.0)
#    The datasets in jtr_marine/data/dataset.py build both (keys "input", "geometry").
# pred = model(x, g)
```

Evaluate the paper model on the internal test gathers (deep learning only, a
few minutes on a GPU) and compare with the reference values:

```bash
PYTHONHASHSEED=0 python scripts/evaluate/recompute_with_per_gather_lists.py \
    --skip-pocs --output-dir results/check
python scripts/stats/check_against_reference.py \
    reference_results/internal_test_compact_unet_seed42.json \
    results/check/per_gather_results.json
```

Train the paper model from scratch (150 epochs, about 9 h on the GPU above):

```bash
python scripts/train/train.py --within-geology
```

## Known behaviours

These points describe what the code does where it could be misread. They are
kept as they are so that the published numbers remain reproducible.

1. **Mask seeds depend on the Python hash seed.** The evaluation masks use the
   seed `random_seed + abs(hash(strategy)) % 1000`, and Python randomises
   string hashing in every process unless `PYTHONHASHSEED` is set. Two runs
   are paired (same masks) only with the same `PYTHONHASHSEED`, or with the
   same `--mask-seed-offset`. The paired comparisons between deep-learning
   models (Tables 9, S2, S3, S7) use `PYTHONHASHSEED=0`. The run behind
   Table 5 was made without a fixed hash seed; its offsets were recovered
   (`scripts/evaluate/find_mask_seed_offset.py`) and are `random=587
   block=446`. With these offsets the deep-learning and POCS values of
   Table 5 are reproduced exactly. The "far" strategy does not depend on the
   seed. Mean values from different mask realisations differ by about 0.1 dB
   (e.g. 7.67 dB vs 7.58 dB of SNR on the missing traces at 30 % random).
2. **MSSA in Table 5.** `reference_results/statistics/` holds the MSSA run on
   the same masks as the deep-learning model and POCS (offsets above).
3. **POCS iterations.** The paper uses 100 iterations. The default of
   `recompute_with_per_gather_lists.py` and
   `export_gathers_for_downstream.py` is 50: pass `--n-iter-pocs 100`
   (`evaluate_all_models.sh` does it).
4. **The "test SNR" printed at the end of training is an internal metric.**
   `validate()` in `train.py` (also used by `train_single_geometry_all.py`)
   uses mixed masking (random, block and far, 20-50 %) and averages over
   batches. It is not the SNR on the missing traces at 30 % random masking of
   the result tables, and the two must not be compared: for the paper model
   the internal metric is 2.48 dB where Table 5 gives 7.67 dB.
   `scripts/evaluate/eval_geometry_matrix.py` scores the multi-geometry,
   single-geometry, leave-one-line-out and leave-one-geometry-out models with
   one metric and the same masks; its output for the released models is in
   `reference_results/geometry_matrix_table5masks.txt`.
5. **Training seed.** `train.py` sets no random seed unless `--seed` is given;
   the paper model was trained without it. Retraining gives a model of
   equivalent quality (see the seed-43 and seed-44 runs), not the same weights.
   Use the released checkpoint to reproduce the tables exactly.
6. **12.5 m lines.** Lines recorded with 12.5 m group spacing are brought to
   the 25 m grid by keeping one trace every two, without a spatial anti-alias
   filter (`scripts/evaluate/resampling_analysis.py`, Fig. S2).
7. **Fig. 5.** The horizontal reference lines are values written in
   `make_fig5_generalization.py` (explained in its docstring), not read from
   result files.
8. **Numerical precision.** On the same hardware and software the
   deep-learning values are reproduced to the last digit. Other CUDA, BLAS or
   PyTorch versions can change aggregated values by about 0.01 dB.

## Citation

Please cite the paper; see
[CITATION.cff](CITATION.cff).

```bibtex
@article{brancatelli2026joint,
  author  = {Brancatelli, Giuseppe and Lipari, Vincenzo and Forlin, Edy and
             Pignelli, Rogerio and Busetti, Martina and Geletti, Riccardo},
  title   = {Joint training across streamer geometries for missing-trace
             reconstruction: A comparative evaluation on real marine data},
  journal = {Journal of Applied Geophysics},
  year    = {2026},
  doi     = {10.1016/j.jappgeo.2026.106621}
}
```

## Acknowledgements

The authors acknowledge academic grants from Aspen Technology for the Echos
and Geodepth software. The work was carried out at the National Institute of
Oceanography and Applied Geophysics (OGS), Trieste, Italy.

## License

MIT, see [LICENSE](LICENSE).

## Contact

Giuseppe Brancatelli — `gbrancatelli@ogs.it`
National Institute of Oceanography and Applied Geophysics (OGS),
Borgo Grotta Gigante 42/c, 34010 Sgonico (Trieste), Italy
