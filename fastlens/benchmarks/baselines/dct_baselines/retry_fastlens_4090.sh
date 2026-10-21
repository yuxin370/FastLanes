#!/usr/bin/env bash
# Retry the interrupted P0 FastLens measurement using its passed numerical gate.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES=GPU-40c637bd-acf5-ea1a-0df8-617138228467
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$HOME/tmp"
export FASTLENS_BLOCK_MAJOR_ACCESS_DIR="$PWD/fastlens/data/compressed/imagenet512_val_block_major/access"

old="$HOME/tmp/dct-p0-pilot-4090-20261012-150553/resnet24"
output="$HOME/tmp/dct-p0-fastlens-retry-4090-$(date +%Y%m%d-%H%M%S)"
contract="$old/fastlens.contract/matched.json"
evidence="$old/fastlens.verified.json"
for path in "$contract" "$evidence"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing input: $path" >&2
    exit 1
  fi
done
cpus="$("$PYTHON" -B -c 'import os; print(",".join(str(c) for c in sorted(os.sched_getaffinity(0))[:8]))')"
"$PYTHON" -B - <<'PY'
import os
import subprocess
uuid = os.environ['CUDA_VISIBLE_DEVICES']
subprocess.run(['nvidia-smi', '--id='+uuid, '--query-gpu=name', '--format=csv,noheader'], check=True)
rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                '--format=csv,noheader,nounits'], text=True)
busy = [row for row in rows.splitlines() if row.split(',')[0].strip() == uuid]
if busy:
    raise SystemExit('GPU busy; wait until idle: '+str(busy))
PY
mkdir "$output"
echo "Retrying ResNet24/FastLens on 4090. Results: $output"
taskset -c "$cpus" "$PYTHON" -B -m fastlens.benchmarks.dct_models.measure_process \
  --output "$output/fastlens.repeat0.process.json" \
  --gpu-uuid "$CUDA_VISIBLE_DEVICES" --rss-limit-mib 32768 -- \
  "$PYTHON" -B -m fastlens.benchmarks.baselines.dct_baselines bench \
  --config "$contract" --backend fastlens \
  --root "$PWD/fastlens/data/compressed/imagenet512_val_block_major/manifest.bin" \
  --module-path "$PWD/build/fastlens/torch" --evidence "$evidence" \
  --device cuda:0 --run-kind inference --repeats 1 \
  --warmup-seconds 60 --min-seconds 180 --observe-cache \
  --output "$output/fastlens.repeat0.jsonl" \
  2>&1 | tee "$output/fastlens.repeat0.log"
echo "Passed: FastLens retry. Results: $output"
