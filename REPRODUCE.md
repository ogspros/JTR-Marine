# Reproducing the paper

Author: Giuseppe Brancatelli (OGS) — `gbrancatelli@ogs.it`

This file maps every table and figure of the paper to the command that
produces it. Three levels are possible:

| Level | Needs | Gives |
|---|---|---|
| A. statistics only | nothing but this repository | Tables 5 and 7 from the per-gather values in `reference_results/` |
| B. evaluation | SEG-Y data + released checkpoints, one GPU | every table and figure, same numbers |
| C. training | SEG-Y data, one GPU (about 9 h per compact model) | new models of equivalent quality (not the same weights, see README, *Known behaviours* 6) |

All commands are run from the repository root. Read *Known behaviours* in the
[README](README.md) first: points 1 (mask seeds) and 3 (POCS iterations) matter
for exact agreement.

## 0. Setup

```bash
conda env create -f environment.yml && conda activate jtr-marine
python -m pytest tests -q

export DATA_ROOT=/path/to/segy            # see data/README.md
bash scripts/download_checkpoints.sh --all
# optional: export CKPT_ROOT=...  RESULTS_ROOT=...   (defaults ./checkpoints, ./results)
```

Checking a result against its reference:

```bash
python scripts/stats/check_against_reference.py <reference.json> <new.json>
```

It compares every number the two files have in common (per-gather lists
included) and prints the largest difference; `PASS` means agreement within
`--tol` (default 1e-6).

## A. Statistics without data

```bash
R=reference_results/statistics
# Table 5 (and the curves of Fig. 4): means, 95 % CI, Wilcoxon, Cliff's delta
python scripts/stats/compute_statistics_3methods.py \
    --per-gather $R/per_gather_results.json --mssa $R/mssa_per_gather_rank10.json \
    --output results/statistics/statistics_3methods.json
python scripts/stats/check_against_reference.py $R/statistics_3methods.json \
    results/statistics/statistics_3methods.json

# Table 7 (cross-survey confidence intervals)
python scripts/stats/compute_statistics.py \
    --per-gather $R/per_gather_results.json --cross-survey $R/cross_survey_per_gather.json \
    --output results/statistics/statistics_full.json
python scripts/stats/check_against_reference.py $R/statistics_full.json \
    results/statistics/statistics_full.json

# Fig. 4 from the statistics just computed
python scripts/figures/make_fig4_decimation.py
```

## B. Evaluation with the released models

### Table 5 — internal test, deep learning vs POCS vs MSSA

276 test gathers (last 15 % of each of the six lines), three masking
strategies, seven ratios. The offsets select the mask realisation of the paper.

```bash
# deep learning + POCS (about 2 h; add --skip-pocs for deep learning only, minutes)
python scripts/evaluate/recompute_with_per_gather_lists.py \
    --n-iter-pocs 100 --mask-seed-offset random=587 block=446 \
    --output-dir results/statistics
# MSSA, rank 10, 3 iterations (about 3 h)
python scripts/evaluate/recompute_mssa_per_gather.py \
    --rank 10 --n-iter 3 --strategies random block far \
    --mask-seed-offset random=587 block=446 --output-dir results/statistics

python scripts/stats/check_against_reference.py \
    reference_results/statistics/per_gather_results.json results/statistics/per_gather_results.json
python scripts/stats/check_against_reference.py \
    reference_results/statistics/mssa_per_gather_rank10.json results/statistics/mssa_per_gather_rank10.json
```

Then run the statistics of section A on the files in `results/statistics/`.

POCS and MSSA parameters were chosen on training gathers with
`scripts/evaluate/sweep_pocs_full.py` and `scripts/evaluate/sweep_mssa_optimized.py`
(`sweep_mssa_full.py` is the wider MSSA grid); the dependence of the MSSA rank
on the geometry (Section 3.1.2) is in `scripts/evaluate/mssa_rank_geometry.py`.

### Table 6 — multi-geometry, single-geometry and held-out models

```bash
python scripts/evaluate/eval_geometry_matrix.py --mask-seed-offset random=587 \
    --output-dir results/geometry_matrix_table5masks
```

Every model is scored with the SNR on the missing traces at 30 % random
masking, on the same gathers and masks as Table 5 (the `multi` row is the
Table 5 model: 4.79 / 8.79 / 9.43 dB). Rows: `single_*` trained on one
geometry; `logo_*` trained without one geometry (`train.py
--holdout-geometry`); `loo_fold*` trained without one line (`train.py
--fold`). The values printed by the training scripts themselves at the end of
a run use a different, internal metric (README, *Known behaviours* 4 and 5);
the script prints both side by side.

### Table 7 — cross-survey test (CG-3, IT97-245, CROP-M18)

```bash
python scripts/evaluate/run_cross_survey_test_v6_final.py --output-dir results/cross_survey_v6
python scripts/stats/check_against_reference.py \
    reference_results/cross_survey_compact_unet.json results/cross_survey_v6/cross_survey_v6_final.json
# per-gather values for the confidence intervals
python scripts/evaluate/recompute_cross_survey_per_gather.py --output-dir results/statistics
```

### Tables 8, 10, S4 — Marmousi and amplitudes

```bash
python scripts/data_prep/marmousi_to_segy.py data/marmousi/marmrefl.hh --outdir $DATA_ROOT
PYTHONHASHSEED=0 python scripts/evaluate/eval_marmousi.py \
    --datasets MARMOUSI_1500 --ratios 0.3 --output-dir results/marmousi
python scripts/stats/check_against_reference.py \
    reference_results/marmousi_1500_compact_unet.json results/marmousi/marmousi_per_gather.json
python scripts/evaluate/amp_check_real.py          # amplitude check on the field data
```

`eval_marmousi.py --help` lists the other datasets, ratios and the
`--save-examples` option needed by Fig. 6.

### Tables 9, S2, S3, S7 — comparison between deep-learning models

```bash
bash scripts/evaluate/evaluate_all_models.sh              # about 2 h with POCS
SKIP_POCS=1 bash scripts/evaluate/evaluate_all_models.sh  # deep learning only
python scripts/stats/check_against_reference.py \
    reference_results/internal_test_compact_unet_seed42.json results/table9/compact_seed42/per_gather_results.json
python scripts/stats/check_against_reference.py \
    reference_results/table9_gan_0p01_sweep.json results/table9/gan_0p01_sweep/per_gather_results.json
```

The script fixes `PYTHONHASHSEED=0`, so all models see the same masks, and
writes one `per_gather_results.json` per model under `results/table9/` and
`results/tableS7/`, with the paired differences of Table S7
(`scripts/stats/compare_chai_v6.py`). Because the mask realisation differs
from that of Table 5, the compact U-Net scores 7.58 dB here and 7.67 dB in
Table 5.

### Stacked sections (Fig. 8)

```bash
python scripts/evaluate/export_gathers_for_downstream.py \
    --ratios 0.5 --mssa-rank 10 --mssa-iter 3 --n-iter-pocs 100
```

writes SEG-Y gathers (ground truth, masked, deep learning, MSSA, POCS) under
`results/output_for_stack/`. NMO correction and stacking were done with
commercial processing software and are not part of this repository; put the
stacked sections back in that folder and run `make_fig8_stack.py` (file names
in its docstring).

## Figures

| Figure | Command | Needs |
|---|---|---|
| 1 map | `python scripts/data_prep/download_map_data.py` then `python scripts/figures/make_fig1_map.py` | public layers only (`data/README.md`) |
| 2 architecture | `python scripts/figures/make_fig2_architecture.py` | nothing |
| 3 convergence | `python scripts/figures/make_fig3_convergence.py` | `history.json` of the paper model and the training log of the no-FiLM model |
| 4 SNR vs decimation | `python scripts/figures/make_fig4_decimation.py` | `results/statistics/statistics_3methods.json` (section A) |
| 5 generalization | `python scripts/figures/make_fig5_generalization.py` | cross-survey results (Table 7) and the fold summary, see below |
| 6 Marmousi | `python scripts/figures/make_fig6_marmousi.py` | `eval_marmousi.py --save-examples` |
| 7 cross-survey gathers | `python scripts/figures/make_fig7_crosssurvey.py` | data + paper model |
| 8 stacks | `python scripts/figures/make_fig8_stack.py` | stacked sections (above) |
| S1 f-k spectra | `python scripts/figures/make_figS1_fk.py` | data + paper model |
| S2 resampling | `python scripts/evaluate/resampling_analysis.py --figure` | data |
| S4 augmentation | `python scripts/figures/make_figS4_augmentation.py` | `summary.json` of the augmentation runs |

Figures are written to `results/figures/` as `Fig_<N>.*`.

Small training outputs read by Figs. 3, 5 and S4 (`history.json`,
`training_log.json`, `summary.json`, and
`v6_grid25m/cross_geometry_LOO_for_paper.json`, a copy of the fold summary
printed by `train.py`) are written by the training scripts; the ones of the
paper runs are distributed with the checkpoints (`checkpoints/README.md`).

## C. Training

Each script writes `best.pt` under `CKPT_ROOT/<name>/` with the layout expected
by the evaluation scripts. `--help` lists the options.

| Model (paper item) | Command | Checkpoint folder |
|---|---|---|
| compact U-Net, paper model | `python scripts/train/train.py --within-geology` | `v6_grid25m/within_geology` |
| repeated runs | `python scripts/train/train_v6_seeds.py --seed 43` (and 44) | `v6_seed43/within_geology` |
| without FiLM | `python scripts/train/train_no_film.py` | `no_film` |
| single geometry (Table 6) | `python scripts/train/train_single_geometry_all.py --streamer 1500` (3000, 4500) | `single_1500m_all` |
| leave one line out | `python scripts/train/train.py --fold 0` (1, 2) | `v6_grid25m/fold0` |
| leave one geometry out (Table 6) | `python scripts/train/train.py --holdout-geometry 1500 --seed 42` (3000, 4500) | `logo_1500m/within_geology` |
| loss weights (Table 3) | `python scripts/train/sweep_loss_weights.py` (runs `train_v6_lossweights.py`) | `lossweights/<tag>/within_geology` |
| kernel / normalisation (Table S5) | `python scripts/train/train_v6_ablation.py --kernel 5 --norm instance` | `v6_ablation/<tag>` |
| GAN (Tables 9, S1) | `python scripts/train/sweep_gan_lambda.py`; `python scripts/train/train_gan.py --seed 43` (44) | `gan_sweep/gan_lam<λ>_seed42`, `gan_seed43` |
| attention variants (Table 9) | `python scripts/train/train_v9_attn.py --attention channel` (spatial, transformer) | `v9_channel` |
| augmentation (Fig. S4) | `python scripts/train/train_v7b.py [--curriculum]`; `python scripts/train/train_v8.py [--noise-snr-min 15 --noise-snr-max 30]` | `v7b_baseline`, `v7b_full`, `v8_*` |
| Chai et al. (2020) U-Net (Table S7) | `python scripts/train/train_chai2020.py --lr 3e-4` | `chai2020/lr3e-4/within_geology` |
| Chai variants (Table S7) | `python scripts/train/train_chai2020_cond.py` (`--no-film`, `--no-mask`) | `chai2020_cond/<tag>/within_geology` |

GAN λ = 0.01 has three runs in Table 9: the sweep run
`gan_sweep/gan_lam0p01_seed42`, `gan_seed43` and `gan_seed44`.
