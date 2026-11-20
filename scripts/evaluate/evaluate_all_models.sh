#!/usr/bin/env bash
# evaluate_all_models.sh -- re-evaluate the deep-learning models of Table 9 (and
# the Chai et al. 2020 configurations of Table S7) on the SAME test gathers and
# the SAME masks, so that every comparison is paired.
#
# Masks: recompute_with_per_gather_lists.py derives the mask seed from
# abs(hash(strategy)) % 1000. Python randomises string hashing per process,
# so PYTHONHASHSEED is fixed here: all paired comparisons of the paper
# (Tables 9, S2, S3, S7) were obtained with PYTHONHASHSEED=0.
#
# Inputs : checkpoints under CKPT_ROOT (default ./checkpoints, layout of
#          checkpoints/README.md); SEG-Y files in DATA_ROOT.
# Outputs: RESULTS_ROOT/table9/<model>/per_gather_results.json
#          RESULTS_ROOT/tableS7/<model>/per_gather_results.json
#
# The compact U-Net (seed 42) is run with POCS on all strategies (reference
# for the paired comparisons and Table 5); all the other models are run DL-only
# (--skip-pocs) on the random strategy, as in the paper. Missing checkpoints
# are skipped with a message.
#
# Usage (from the repository root; GPU recommended, about 2 h with POCS):
#   bash scripts/evaluate/evaluate_all_models.sh
#   SKIP_POCS=1 bash scripts/evaluate/evaluate_all_models.sh   # DL only, minutes

set -u
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0

CKPT_ROOT="${CKPT_ROOT:-$PWD/checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-$PWD/results}"
export CKPT_ROOT RESULTS_ROOT
RECOMPUTE="python -u scripts/evaluate/recompute_with_per_gather_lists.py"
# POCS iterations of the paper (Table 5, Section 2.7, all paired recomputes):
# 100, passed explicitly; the script default (50) is left unchanged.
N_ITER_POCS="${N_ITER_POCS:-100}"

run () {   # run <table> <name> <checkpoint> [extra args...]
  local table=$1 name=$2 ck=$3; shift 3
  if [[ ! -f "$CKPT_ROOT/$ck" ]]; then
    echo "MISSING checkpoint for $name: $CKPT_ROOT/$ck  (skipped)"
    return
  fi
  local out="$RESULTS_ROOT/$table/$name"
  mkdir -p "$out"
  echo "--- $table / $name  $(date)"
  $RECOMPUTE --checkpoint "$CKPT_ROOT/$ck" --output-dir "$out" "$@" 2>&1 | tee "$out/log.txt"
}

# ------------------------------------------------------------- Table 9
# compact U-Net, seed 42 (paper checkpoint): all strategies, with POCS
if [[ "${SKIP_POCS:-0}" == "1" ]]; then
  run table9 compact_seed42 v6_grid25m/within_geology/best.pt --skip-pocs
else
  run table9 compact_seed42 v6_grid25m/within_geology/best.pt --n-iter-pocs "$N_ITER_POCS"
fi

DL_ONLY=(--strategies random --skip-pocs)
run table9 compact_seed43   v6_seed43/within_geology/best.pt               "${DL_ONLY[@]}"
run table9 compact_seed44   v6_seed44/within_geology/best.pt               "${DL_ONLY[@]}"
run table9 gan_0p001        gan_sweep/gan_lam0p001_seed42/best.pt          "${DL_ONLY[@]}"
# GAN lambda = 0.01, three runs. Run 1 is the sweep run gan_lam0p01_seed42;
# a run in gan_seed42/, if present, is a different run, not used in the paper.
run table9 gan_0p01_sweep   gan_sweep/gan_lam0p01_seed42/best.pt           "${DL_ONLY[@]}"
run table9 gan_0p01_r2      gan_seed43/best.pt                             "${DL_ONLY[@]}"
run table9 gan_0p01_r3      gan_seed44/best.pt                             "${DL_ONLY[@]}"
run table9 attn_channel     v9_channel/best.pt                             "${DL_ONLY[@]}"
run table9 attn_spatial     v9_spatial/best.pt                             "${DL_ONLY[@]}"
run table9 attn_transformer v9_transformer/best.pt                         "${DL_ONLY[@]}"

# ------------------------------------------------------------- Table S7
# Chai et al. (2020) U-Net and its variants with our mask / loss / FiLM,
# all strategies, DL only (POCS on the same masks is in table9/compact_seed42).
CHAI=(--skip-pocs)
run tableS7 chai_mse        chai2020/lr3e-4/within_geology/best.pt                     "${CHAI[@]}"
run tableS7 chai_mask_film  chai2020_cond/mask_film_lr3e-4/within_geology/best.pt      "${CHAI[@]}"
run tableS7 chai_mask_only  chai2020_cond/mask_only_lr3e-4/within_geology/best.pt      "${CHAI[@]}"
run tableS7 chai_loss_only  chai2020_cond/loss_only_lr3e-4/within_geology/best.pt      "${CHAI[@]}"

# paired differences and block-bootstrap CI (Table S7)
REF="$RESULTS_ROOT/table9/compact_seed42/per_gather_results.json"
for m in chai_mse chai_mask_film chai_mask_only chai_loss_only; do
  f="$RESULTS_ROOT/tableS7/$m/per_gather_results.json"
  [[ -f "$f" && -f "$REF" ]] || continue
  echo "=== Table S7: $m vs compact U-Net"
  python scripts/stats/compare_chai_v6.py "$f" "$REF" | tee "$RESULTS_ROOT/tableS7/$m/paired_vs_compact.txt"
done

echo "=== DONE  $(date)"
