#!/usr/bin/env bash
# Same inference matrix on the requested GPU, reusing complete-frequency mother data.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=${1:-GPU-e796262d-3449-6af1-586d-8460d8836d1b}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/home/tangyuxin/tmp/dctnet MPLCONFIGDIR=/home/tangyuxin/tmp/matplotlib
PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
EXP=galp/experiments/dct_pushdown_inference
DATA=galp/data/system_rgbnomore/e2e_v3
RUN_NAME=${2:-pro6000_io_pushdown_20260913}
PHASE=full
CONTROL=$DATA/runs/dctnet_mobilenet24/$RUN_NAME/monitoring
if [[ "$CUDA_VISIBLE_DEVICES" == GPU-e796262d-3449-6af1-586d-8460d8836d1b ]]; then
  PHASE=firefox_paused
  CONTROL=$DATA/runs/dctnet_mobilenet24/$RUN_NAME/browser_control
fi
mkdir -p "$CONTROL"
MONITOR_PID=
HOST_PID=
cleanup() {
  if [[ "$PHASE" == firefox_paused ]]; then touch "$CONTROL/resume"; fi
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
check_paused() {
  if [[ "$PHASE" != firefox_paused ]]; then return; fi
  "$PYTHON" - "$CONTROL" <<'PY'
import json, sys
from pathlib import Path
state = json.loads((Path(sys.argv[1]) / 'browser_state.json').read_text())
assert state['state'] == 'paused', state
for pid in state['pids']:
    assert Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] == 'T', pid
PY
}
check_paused
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
    "$PYTHON" "$EXP/measure_process.py" --output "$metric" -- "$@"
  else
    "$@"
  fi
}
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw,temperature.gpu --format=csv -l 1 > "$CONTROL/gpu.csv" &
MONITOR_PID=$!
vmstat -t 1 > "$CONTROL/host_vmstat.log" &
HOST_PID=$!
CONFIGURATIONS=${3:-"mobilenet24:dctnet_mobilenet24:dctnet_mobilenet32 resnet24:dctnet_static24:dctnet_static64"}
for spec in $CONFIGURATIONS; do
  IFS=: read -r DCTNET_PROFILE name mother <<< "$spec"
  export DCTNET_PROFILE
  RUN=$DATA/runs/$name/$RUN_NAME
  OUT=$RUN/$PHASE
  mkdir -p "$RUN/logs" "$OUT"
  for route in pytorch dali; do
    check_paused
    run_benchmark "$OUT/memory_RGB_$route.json" "$PYTHON" "$EXP/evaluate_rgb.py" --route "$route" --count 50000 --batch-size 64 \
      --workers 64 --output-dir "$OUT" > "$RUN/logs/RGB_$route.log" 2>&1
  done
  for route in R O; do
    check_paused
    run_benchmark "$OUT/memory_$route.json" "$PYTHON" "$EXP/evaluate.py" --route "$route" --device cuda --count 50000 \
      --batch-size 64 --workers 64 --output-dir "$OUT" > "$RUN/logs/$route.log" 2>&1
  done
  for variant in grid_off grid_on projected_off projected_on; do
    check_paused
    run_benchmark "$OUT/memory_native_$variant.json" "$PYTHON" "$EXP/evaluate_shards.py" --count 50000 --batch-size 64 \
      --new-data "$DATA/dct_major_$mother" --output-layout "${variant%_*}" --pushdown "${variant##*_}" \
      --baseline-dir "$OUT" --output-dir "$OUT/native_$variant" > "$RUN/logs/native_$variant.log" 2>&1
  done
done
check_paused
