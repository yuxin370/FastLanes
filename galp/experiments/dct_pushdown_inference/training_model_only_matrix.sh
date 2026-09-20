#!/usr/bin/env bash
# Shared Transformer training calibration; this is not the E2E training matrix.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=GPU-40c637bd-acf5-ea1a-0df8-617138228467
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/tangyuxin/tmp/dctnet MPLCONFIGDIR=/home/tangyuxin/tmp/matplotlib
PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
EXP=galp/experiments/dct_pushdown_inference
RUN_NAME=${1:-rtx4090_training_20260914}
if (( $# > 0 )); then shift; fi
for spec in mobilenet24:dctnet_mobilenet24 mobilenet32:dctnet_mobilenet32 resnet24:dctnet_static24 resnet64:dctnet_static64; do
  IFS=: read -r DCTNET_PROFILE name <<< "$spec"
  export DCTNET_PROFILE
  OUT=galp/data/system_rgbnomore/e2e_v3/runs/$name/$RUN_NAME/model_only
  mkdir -p "$OUT"
  if [[ ! -f "$OUT/dct_model_only.json" ]]; then
    "$PYTHON" "$EXP/training_model_only.py" --domain dct --output-dir "$OUT" "$@" > "$OUT/dct.log" 2>&1
  fi
  if [[ "$DCTNET_PROFILE" == *24 ]]; then
    if [[ ! -f "$OUT/rgb_model_only.json" ]]; then
      "$PYTHON" "$EXP/training_model_only.py" --domain rgb --output-dir "$OUT" "$@" > "$OUT/rgb.log" 2>&1
    fi
  fi
done
