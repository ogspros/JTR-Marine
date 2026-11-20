#!/usr/bin/env bash
# download_checkpoints.sh -- fetch the trained models of the paper from the
# GitHub release of this repository (author: Giuseppe Brancatelli).
#
# Files and SHA-256 are listed in checkpoints/MANIFEST.txt ("<sha256>  <path>").
# A path X/Y/best.pt is stored in the release as the asset X__Y__best.pt and is
# written to CKPT_ROOT/X/Y/best.pt (default CKPT_ROOT: ./checkpoints). Paths
# starting with results/ go under RESULTS_ROOT (default ./results).
#
# Usage (from the repository root):
#   bash scripts/download_checkpoints.sh            # paper model only
#   bash scripts/download_checkpoints.sh --all      # everything
#   bash scripts/download_checkpoints.sh PATTERN    # paths containing PATTERN
set -euo pipefail
cd "$(dirname "$0")/.."

RELEASE_URL="${RELEASE_URL:-https://github.com/ogspros/JTR-Marine/releases/download/v1.0}"
CKPT_ROOT="${CKPT_ROOT:-$PWD/checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-$PWD/results}"
MANIFEST=checkpoints/MANIFEST.txt
PATTERN="${1:-v6_grid25m/within_geology/}"
[[ "$PATTERN" == "--all" ]] && PATTERN=""

[[ -f "$MANIFEST" ]] || { echo "ERROR: $MANIFEST not found"; exit 1; }
n=0
while read -r sha path; do
  [[ -z "${sha:-}" || "$sha" == \#* ]] && continue
  [[ "$path" == *"$PATTERN"* ]] || continue
  if [[ "$path" == results/* ]]; then dst="$RESULTS_ROOT/${path#results/}"; else dst="$CKPT_ROOT/$path"; fi
  n=$((n + 1))
  if [[ -f "$dst" ]] && echo "$sha  $dst" | sha256sum --check --status; then
    echo "ok       $path"; continue
  fi
  mkdir -p "$(dirname "$dst")"
  echo "download $path"
  curl -L --fail --retry 3 -o "$dst.part" "$RELEASE_URL/${path//\//__}"
  echo "$sha  $dst.part" | sha256sum --check --status || { echo "ERROR: checksum mismatch for $path"; exit 1; }
  mv "$dst.part" "$dst"
done < "$MANIFEST"
[[ $n -gt 0 ]] || { echo "No file in $MANIFEST matches '$PATTERN'"; exit 1; }
echo "Done: $n file(s)."
