#!/usr/bin/env bash
# Same inference matrix on the requested GPU, reusing complete-frequency mother data.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=${1:-${CUDA_VISIBLE_DEVICES:?Pass a GPU UUID as the first argument}}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=${HOME}/tmp/dctnet MPLCONFIGDIR=${HOME}/tmp/matplotlib
PYTHON=${PYTHON:-python3}
EXP=galp/benchmarks/dct_models
DATA=galp/data/system_rgbnomore/e2e_v3
RUN_NAME=${2:?Pass a result directory name as the second argument}
PHASE=full
CONTROL=$DATA/runs/dctnet_mobilenet24/$RUN_NAME/monitoring
mkdir -p "$CONTROL"
MONITOR_PID=
HOST_PID=
cleanup() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID"
    wait "$MONITOR_PID" || true
  fi
  if [[ -n "$HOST_PID" ]]; then
    kill "$HOST_PID"
    wait "$HOST_PID" || true
  fi
}
trap cleanup EXIT
run_benchmark() {
  local metric=$1
  shift
  if [[ ${MEASURE_PROCESS_MEMORY:-0} == 1 ]]; then
    if [[ -f "$metric" ]]; then
      "$PYTHON" - "$metric" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
assert json.loads(path.read_text())['returncode'] == 0, path
route = path.stem.removeprefix('memory_')
result = path.parent / route / 'N_50000.json' if route.startswith('native_') else path.parent / f'{route}_50000.json'
assert json.loads(result.read_text())['samples'] == 50000, result
PY
      return
    fi
    "$PYTHON" -m galp.benchmarks.dct_models.measure_process --output "$metric" -- "$@"
  else
    "$@"
  fi
}
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw,temperature.gpu --format=csv -l 1 > "$CONTROL/gpu.csv" &
MONITOR_PID=$!
vmstat -t 1 > "$CONTROL/host_vmstat.log" &
HOST_PID=$!
CONFIGURATIONS=${3:-"mobilenet24:dctnet_mobilenet24:imagenet512_val_mobilenet112_block_major mobilenet32:dctnet_mobilenet32:imagenet512_val_mobilenet112_block_major resnet24:dctnet_static24:imagenet512_val_resnet56_block_major resnet64:dctnet_static64:imagenet512_val_resnet56_block_major"}
for spec in $CONFIGURATIONS; do
  IFS=: read -r DCTNET_PROFILE name mother <<< "$spec"
  export DCTNET_PROFILE
  RUN=$DATA/runs/$name/$RUN_NAME
  OUT=$RUN/$PHASE
  mkdir -p "$RUN/logs" "$OUT"
  for route in pytorch dali; do
    run_benchmark "$OUT/memory_RGB_$route.json" "$PYTHON" -m galp.benchmarks.dct_models.evaluate_rgb --route "$route" --count 50000 --batch-size 64 \
      --workers 64 --output-dir "$OUT" > "$RUN/logs/RGB_$route.log" 2>&1
  done
  for route in R O; do
    run_benchmark "$OUT/memory_$route.json" "$PYTHON" -m galp.benchmarks.dct_models.evaluate --route "$route" --device cuda --count 50000 \
      --batch-size 64 --workers 64 --output-dir "$OUT" > "$RUN/logs/$route.log" 2>&1
  done
  for variant in grid_off grid_on projected_off projected_on; do
    run_benchmark "$OUT/memory_native_$variant.json" "$PYTHON" -m galp.benchmarks.dct_models.evaluate_shards --count 50000 --batch-size 64 \
      --new-data "galp/data/compressed/backup/offline_model_input/$mother" --output-layout "${variant%_*}" --pushdown "${variant##*_}" \
      --baseline-dir "$OUT" --output-dir "$OUT/native_$variant" > "$RUN/logs/native_$variant.log" 2>&1
  done
done
