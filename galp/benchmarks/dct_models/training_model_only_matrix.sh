#!/usr/bin/env bash
# Shared Transformer training calibration; this is not the E2E training matrix.
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the requested GPU}"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=${HOME}/tmp/dctnet MPLCONFIGDIR=${HOME}/tmp/matplotlib
PYTHON=${PYTHON:-python3}
EXP=galp/benchmarks/dct_models
RUN_NAME=${1:?Pass a result directory name as the first argument}
if (( $# > 0 )); then shift; fi
for spec in mobilenet24:dctnet_mobilenet24 mobilenet32:dctnet_mobilenet32 resnet24:dctnet_static24 resnet64:dctnet_static64; do
  IFS=: read -r DCTNET_PROFILE name <<< "$spec"
  export DCTNET_PROFILE
  OUT=galp/data/system_rgbnomore/e2e_v3/runs/$name/$RUN_NAME/model_only
  mkdir -p "$OUT"
  if [[ ! -f "$OUT/dct_model_only.json" ]]; then
    "$PYTHON" -m galp.benchmarks.dct_models.training_model_only --domain dct --output-dir "$OUT" "$@" > "$OUT/dct.log" 2>&1
  fi
  if [[ "$DCTNET_PROFILE" == *24 ]]; then
    if [[ ! -f "$OUT/rgb_model_only.json" ]]; then
      "$PYTHON" -m galp.benchmarks.dct_models.training_model_only --domain rgb --output-dir "$OUT" "$@" > "$OUT/rgb.log" 2>&1
    fi
  fi
done
