#!/usr/bin/env bash
# Reuse the premixed materializer's paths, mapping identity, and existing run hierarchy.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=GPU-40c637bd-acf5-ea1a-0df8-617138228467
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/tangyuxin/tmp/dctnet MPLCONFIGDIR=/home/tangyuxin/tmp/matplotlib
PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
EXP=galp/experiments/dct_pushdown_inference
DATA=galp/data/system_rgbnomore/e2e_v3
MODE=${1:-probe}
RUN=${2:-rtx4090_native_training_20260914}
CONDITION=${3:-B6}
M=${4:-4}
BLOCKS=${5:-32768}
PHYSICAL=/mnt/nvme2/home/tangyuxin/pls-experiments/physical-layout-full-premix-orgseed-20260810/uniform_premix
mapfile -t INPUTS < <("$PYTHON" - "$PHYSICAL/materialization_contract.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); c=json.loads(p.read_text())
print(p.parent/'dct/manifest.bin')
print(c['ordered_mapping'])
print(c['ordered_mapping_sha256'])
PY
)
EXTRA=()
case "$MODE" in
  probe) EXTRA=(--max-pools 2 --validation-count 1000) ;;
  full) EXTRA=(--epochs 2 --validation-count 50000) ;;
  *) exit 2 ;;
esac
for spec in mobilenet24:dctnet_mobilenet24 mobilenet32:dctnet_mobilenet32 resnet24:dctnet_static24 resnet64:dctnet_static64; do
  IFS=: read -r DCTNET_PROFILE name <<< "$spec"
  export DCTNET_PROFILE
  OUT=$DATA/runs/$name/$RUN/${CONDITION}_${MODE}_m${M}_inplace
  mkdir -p "$OUT"
  if [[ -f "$OUT/training.json" ]]; then continue; fi
  RESUME=()
  for epoch in 1 0; do
    if [[ "$MODE" == full && -f "$OUT/epoch_$epoch.pth" ]]; then
      RESUME=(--resume "$OUT/epoch_$epoch.pth")
      break
    fi
  done
  "$PYTHON" "$EXP/training_pls.py" --manifest "${INPUTS[0]}" --mapping "${INPUTS[1]}" \
    --mapping-sha256 "${INPUTS[2]}" --condition "$CONDITION" --segments-per-pool "$M" \
    --transform-blocks-per-launch "$BLOCKS" --output-dir "$OUT" "${EXTRA[@]}" "${RESUME[@]}" > "$OUT/run.log" 2>&1
  cat "$OUT/run.log"
done
