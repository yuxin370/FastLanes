#!/usr/bin/env bash
# Run from the repository root, after pause_firefox.py reports paused.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=GPU-e796262d-3449-6af1-586d-8460d8836d1b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/home/tangyuxin/tmp/dctnet MPLCONFIGDIR=/home/tangyuxin/tmp/matplotlib
PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
EXP=galp/experiments/dct_pushdown_inference
DATA=galp/data/system_rgbnomore/e2e_v3
CONTROL=$DATA/runs/dctnet_mobilenet32/pro6000_all_baselines_20260913/browser_control
MONITOR_PID=
cleanup() {
  touch "$CONTROL/resume"
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID"
    wait "$MONITOR_PID" || true
  fi
}
trap cleanup EXIT
check_paused() {
  "$PYTHON" - "$CONTROL" <<'PY'
import json, sys
from pathlib import Path
state = json.loads((Path(sys.argv[1]) / 'browser_state.json').read_text())
assert state['state'] == 'paused', state
for pid in state['pids']:
    assert Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0] == 'T', pid
PY
}
check_paused
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw,temperature.gpu --format=csv -l 1 > "$CONTROL/paused_gpu.csv" &
MONITOR_PID=$!

# Measure the two native paths first for the Firefox on/off comparison.
for spec in mobilenet32:dctnet_mobilenet32 resnet64:dctnet_static64; do
  export DCTNET_PROFILE=${spec%:*}
  name=${spec#*:}
  RUN=$DATA/runs/$name/pro6000_all_baselines_20260913
  check_paused
  "$PYTHON" "$EXP/evaluate_shards.py" --count 50000 --batch-size 64 \
    --new-data "$DATA/dct_major_$name" --output-layout projected --pushdown on \
    --output-dir "$RUN/firefox_paused/native_projected_on" > "$RUN/logs/paused_native_projected_on.log" 2>&1
done

for spec in mobilenet32:dctnet_mobilenet32 resnet64:dctnet_static64; do
  export DCTNET_PROFILE=${spec%:*}
  name=${spec#*:}
  RUN=$DATA/runs/$name/pro6000_all_baselines_20260913
  OUT=$RUN/firefox_paused
  for route in pytorch dali; do
    check_paused
    "$PYTHON" "$EXP/evaluate_rgb.py" --route "$route" --count 50000 --batch-size 64 \
      --workers 64 --output-dir "$OUT" > "$RUN/logs/paused_RGB_$route.log" 2>&1
  done
  for route in R O; do
    check_paused
    "$PYTHON" "$EXP/evaluate.py" --route "$route" --device cuda --count 50000 \
      --batch-size 64 --workers 64 --output-dir "$OUT" > "$RUN/logs/paused_$route.log" 2>&1
  done
  for variant in grid_off grid_on projected_off; do
    check_paused
    "$PYTHON" "$EXP/evaluate_shards.py" --count 50000 --batch-size 64 \
      --new-data "$DATA/dct_major_$name" --output-layout "${variant%_*}" --pushdown "${variant##*_}" \
      --baseline-dir "$OUT" --output-dir "$OUT/native_$variant" > "$RUN/logs/paused_native_$variant.log" 2>&1
  done
  if [[ "$DCTNET_PROFILE" == resnet64 ]]; then
    check_paused
    "$PYTHON" "$EXP/evaluate.py" --route N --device cuda --count 50000 \
      --batch-size 64 --workers 64 --new-data "$DATA/compact_v3_tiled_z32_dctnet_static64" \
      --output-dir "$OUT/legacy_bridge" > "$RUN/logs/paused_legacy_bridge.log" 2>&1
  fi
  check_paused
  "$PYTHON" "$EXP/evaluate_shards.py" --count 50000 --batch-size 64 \
    --new-data "$DATA/dct_major_$name" --output-layout projected --pushdown on \
    --baseline-dir "$OUT" --output-dir "$OUT/repeat_native_projected_on" > "$RUN/logs/paused_repeat_native.log" 2>&1
done
check_paused
