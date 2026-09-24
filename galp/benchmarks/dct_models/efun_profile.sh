#!/usr/bin/env bash
# Capture every measured eFUN/RGB inference route and epoch-2 training arm.
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${CUDA_VISIBLE_DEVICES:?Select a confirmed idle GPU UUID}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=${HOME}/tmp/dctnet MPLCONFIGDIR=${HOME}/tmp/matplotlib
PYTHON=${PYTHON:-python3}
EXP=galp/benchmarks/dct_models
DATA=galp/data/system_rgbnomore/e2e_v3
OUT=$DATA/runs/efun/${1:-nsys_20260915}
ACTIVE_PIDS=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "$ACTIVE_PIDS" ]]; then
  echo "Selected GPU already has compute processes" >&2
  exit 1
fi
mkdir -p "$OUT"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=timestamp,gpu_uuid,pid,used_gpu_memory \
  --format=csv,noheader,nounits -lms 100 >> "$OUT/gpu_processes.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID"; wait "$MONITOR_PID" || true' EXIT

capture() {
  local trace=$1
  shift
  mkdir -p "$(dirname "$trace")"
  if [[ ! -f "$trace.nsys-rep" ]]; then
    nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop --output "$trace" \
      "$@" > "$(dirname "$trace")/run.log" 2>&1
  fi
  if [[ ! -f "$trace.sqlite" ]]; then
    nsys export --type sqlite --output "$trace.sqlite" "$trace.nsys-rep"
  fi
}

for route in jpeg grid_off projected_off projected_on rgb_pytorch rgb_dali; do
  DIR=$OUT/inference/$route
  case "$route" in
    jpeg)
      NAME=dct_reference
      COMMAND=("$PYTHON" -m galp.benchmarks.dct_models.efun evaluate --route R --device cuda --workers 16) ;;
    grid_off|projected_off|projected_on)
      NAME=galp_projected
      if [[ "$route" == grid_off ]]; then NAME=galp_native; fi
      COMMAND=("$PYTHON" -m galp.benchmarks.dct_models.efun evaluate_shards --new-data "galp/data/compressed/backup/offline_model_input/imagenet512_val_efun28_block_major"
        --output-layout "${route%_*}" --pushdown "${route##*_}") ;;
    rgb_pytorch|rgb_dali)
      NAME=$route
      WORKERS=16
      if [[ "$route" == rgb_dali ]]; then WORKERS=4; fi
      COMMAND=("$PYTHON" -m galp.benchmarks.dct_models.efun evaluate_rgb --route "${route#rgb_}" --workers "$WORKERS")
      if [[ "$route" == rgb_dali ]]; then COMMAND+=(--dali-prefetch-depth 4); fi ;;
  esac
  echo "Capturing inference $route"
  capture "$DIR/$NAME" "${COMMAND[@]}" --count 50000 --batch-size 64 --profile --output-dir "$DIR/run"
  "$PYTHON" -m galp.benchmarks.dct_models.analyze_profiles "$DIR" > "$DIR/analysis.log"
done

PHYSICAL=$PWD/galp/data/compressed/imagenet512_train_block_major_premixed
mapfile -t INPUTS < <("$PYTHON" - "$PHYSICAL/materialization_contract.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); c = json.loads(p.read_text())
print(p.parent / 'dct/manifest.bin')
print(c['ordered_mapping'])
print(c['ordered_mapping_sha256'])
PY
)
for backend in jpeg native rgb_pytorch rgb_d2; do
  DIR=$OUT/training/$backend
  SOURCE=$DATA/runs/efun/training_v1
  CONDITION=RGB; WORKERS=16
  INPUT_EXTRA=()
  case "$backend" in
    jpeg) CONDITION=A0; SOURCE=$DATA/runs/efun/training_v1_jpeg_rerun_20260915 ;;
    native) CONDITION=B6 ;;
    rgb_d2) SOURCE=$DATA/runs/efun/dali_tuning_20260916; INPUT_EXTRA=(--dali-prefetch-depth 4) ;;
  esac
  CHECKPOINT=$SOURCE/${backend}_full/epoch_0.pth
  if [[ "$backend" == rgb_d2 ]]; then CHECKPOINT=$SOURCE/rgb_d2_clean_full/epoch_0.pth; fi
  echo "Capturing training $backend from epoch 1 checkpoint"
  capture "$DIR/capture" "$PYTHON" -m galp.benchmarks.dct_models.efun training_pls \
    --manifest "${INPUTS[0]}" --mapping "${INPUTS[1]}" --mapping-sha256 "${INPUTS[2]}" \
    --condition "$CONDITION" --input-backend "$backend" --workers "$WORKERS" \
    --segments-per-pool 4 --transform-blocks-per-launch 0 --epochs 2 "${INPUT_EXTRA[@]}" \
    --resume "$CHECKPOINT" --profile --output-dir "$DIR/run"
  "$PYTHON" -m galp.benchmarks.dct_models.analyze_training_capture "$DIR/capture.sqlite" \
    --output "$DIR/breakdown.json" --timeline "$DIR/timeline.png" \
    --label "eFUN / $backend / epoch 2"
done
