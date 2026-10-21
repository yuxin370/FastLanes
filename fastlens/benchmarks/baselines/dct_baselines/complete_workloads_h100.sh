#!/usr/bin/env bash
# Wait for the current run/GPU, then complete IC/FS for all seven workloads.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
: "${PYTHON:?Set PYTHON to the fastlanes-cuda Python executable}"
: "${CUDA_VISIBLE_DEVICES:?Select H100 by its full GPU UUID}"
prepared="$(cd "${1:?Usage: complete_workloads_h100.sh PREPARED NEW_OUTPUT PREVIOUS_RUN}" && pwd)"
output="${2:?Usage: complete_workloads_h100.sh PREPARED NEW_OUTPUT PREVIOUS_RUN}"
previous="$(cd "${3:?Usage: complete_workloads_h100.sh PREPARED NEW_OUTPUT PREVIOUS_RUN}" && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$HOME/tmp"
trap 'echo "Stopped: workload command failed; keep its logs and passed results. No later workload was launched." >&2' ERR

"$PYTHON" -B - "$CUDA_VISIBLE_DEVICES" "$previous" <<'PY'
import os, sys, time
from pathlib import Path
import psutil
from fastlens.benchmarks.dct_models.gpu_occupancy import gpu_occupancy

gpu = sys.argv[1]
previous_run = sys.argv[2]
if not gpu.startswith('GPU-') or ',' in gpu:
    raise ValueError('select one GPU by its full UUID')
idle_since = None
last_state = None
while True:
    running = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            args = process.info['cmdline'] or []
            if process.pid != os.getpid() and any(
                arg == previous_run or arg.startswith(previous_run+'/') for arg in args) and any(
                arg.startswith('fastlens.benchmarks.baselines.dct_baselines') or Path(arg).name in
                ('run_inference_pilot.sh', 'continue_50k_h100.sh') for arg in args):
                running.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    occupied = gpu_occupancy(gpu)['blocking_pids']
    state = (sorted(running), occupied)
    if running or occupied:
        idle_since = None
        if state != last_state:
            print(f'Waiting: experiment PIDs={state[0]}, H100 compute PIDs={occupied}', flush=True)
    elif idle_since is None:
        idle_since = time.monotonic()
        print('Previous experiment finished; no blocking H100 clients. Waiting for 60 continuous idle seconds.', flush=True)
    elif time.monotonic()-idle_since >= 60:
        print('H100 remained idle for 60 seconds; starting the serial workload queue.', flush=True)
        break
    last_state = state
    time.sleep(10)
PY

# The previous run may have completed while this listener was waiting.
# Link only valid repeats; the original failed records are never overwritten.
bash fastlens/benchmarks/baselines/dct_baselines/continue_50k_h100.sh "$prepared" "$output" "$previous"

# MobileNet produces 112x112 planes (four times the ResNet24 input area).
# With eight workers, pool256 exceeds the common 32 GiB input bound for
# IC-batched. Use pool128 for all three IC/FS paths of each MobileNet workload.
"$PYTHON" -B - "$prepared/reader_settings.json" "$output/mobile_reader_settings.json" <<'PY'
import json, sys
from pathlib import Path
settings = json.loads(Path(sys.argv[1]).read_text())
for name in ('ic_shared', 'ic_batched', 'fs'):
    settings.setdefault(name, {})['pool_size'] = 128
Path(sys.argv[2]).write_text(json.dumps(settings, indent=2)+'\n')
PY

for model in mobilenet24 mobilenet32 resnet64 efun swinv2; do
  settings="$prepared/reader_settings.json"
  if [[ "$model" == mobilenet* ]]; then settings="$output/mobile_reader_settings.json"; fi
  echo "Starting remaining workload: $model"
  bash fastlens/benchmarks/baselines/dct_baselines/run_inference_pilot.sh "$prepared" "$output" \
    --ic-protocol --expected-images 50000 --settings "$settings" \
    --models "$model" --systems "ic_shared ic_batched fs" --resume-measurements --wait-for-gpu \
    --measurement-cache-mib ic_shared=1024 --measurement-cache-mib ic_batched=1024 \
    2>&1 | tee "$output/$model.driver.log"
done
echo "Passed: seven workloads x three IC/FS systems x three repeats; existing CPU-DCT results retained. Results: $output/summary.json"
