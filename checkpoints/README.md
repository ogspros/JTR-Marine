# Trained models

Author: Giuseppe Brancatelli (OGS)

The weights are not stored in git. They are assets of the GitHub release
`v1.0` of this repository and are downloaded with

```bash
bash scripts/download_checkpoints.sh            # paper model only
bash scripts/download_checkpoints.sh --all      # every file of MANIFEST.txt
bash scripts/download_checkpoints.sh v6_seed43  # files whose path contains the word
```

`MANIFEST.txt` lists the files, one per line, with their path under
`CKPT_ROOT` (default: this folder) and their SHA-256. A release asset has the
same path with `/` replaced by `__`
(`v6_grid25m/within_geology/best.pt` -> `v6_grid25m__within_geology__best.pt`).
Lines whose path starts with `results/` are small training outputs read by
the figure scripts; they are placed under `RESULTS_ROOT`.

| Folder | Model | Paper |
|---|---|---|
| `v6_grid25m/within_geology` | compact U-Net + FiLM, the paper model (best epoch 146 of 150); with `history.json` | all tables and figures |
| `v6_seed43`, `v6_seed44` | repeated runs of the paper model | Table 9 |
| `no_film` | compact U-Net without FiLM | Fig. 3, Fig. S4 |
| `single_{1500,3000,4500}m_all` | trained on one geometry | Table 6 |
| `logo_{1500,3000,4500}m/within_geology` | trained without one geometry | Table 6 |
| `v6_grid25m/fold{0,1,2}` | trained without one line (SISOC-MEPA, CG-1, IT90-65) | Fig. 5 |
| `lossweights/<weights>/within_geology` | loss-weight sensitivity | Table 3 |
| `v6_ablation/{k3_none,k5_instance,k5_none}/within_geology` | kernel size / normalisation | Table S5 |
| `gan_sweep/gan_lam0p001_seed42`, `gan_sweep/gan_lam0p01_seed42`, `gan_seed43`, `gan_seed44` | adversarial variants | Tables 9, S1 |
| `v9_channel`, `v9_spatial`, `v9_transformer` | attention variants | Table 9 |
| `v7b_baseline`, `v7b_full`, `v8_baseline_snr25`, `v8_full_snr15_30` | training-time augmentation | Fig. S4 |
| `chai2020/lr3e-4`, `chai2020_cond/{mask_film,mask_only,loss_only}_lr3e-4` | U-Net of Chai et al. (2020) and variants (about 1 GB each) | Table S7 |

Each `best.pt` is a PyTorch checkpoint with the model weights
(`model_state`); `jtr_marine.model_loader.load_checkpoint()` recognises the
architecture and builds the right model.
