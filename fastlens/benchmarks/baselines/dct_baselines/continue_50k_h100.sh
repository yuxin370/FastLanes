#!/usr/bin/env bash
# Continue IC/FS measurements, retaining existing CPU-DCT and IC/FS results.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
: "${PYTHON:?Set PYTHON to the fastlanes-cuda Python executable}"
: "${CUDA_VISIBLE_DEVICES:?Select the idle H100 by UUID}"
prepared="$(cd "${1:?Usage: continue_50k_h100.sh PREPARED NEW_OUTPUT [PREVIOUS_RUN]}" && pwd)"
output="${2:?Usage: continue_50k_h100.sh PREPARED NEW_OUTPUT [PREVIOUS_RUN]}"
previous="$(cd "${3:-$prepared/h100}" && pwd)"
# Require the completed ResNet checks rather than silently repeating them.
for name in ic_shared ic_batched fs; do
  test -f "$previous/resnet24/$name.contract/matched.json"
  test -f "$previous/resnet24/$name.verified.json"
done
"$PYTHON" -B - "$previous" "$output" <<'PY'
import json, sys
from pathlib import Path
previous, output = map(Path, sys.argv[1:])
output.mkdir()
retained = 0
for model in ('resnet24', 'vitti', 'mobilenet24', 'mobilenet32', 'resnet64', 'efun', 'swinv2'):
    old = previous/model
    if not old.is_dir():
        continue
    current = output/model
    current.mkdir()
    for name in ('cpu_dct', 'ic_shared', 'ic_batched', 'fs'):
        for suffix in ('contract', 'verified.json'):
            path = old/f'{name}.{suffix}'
            if path.exists():
                (current/path.name).symlink_to(path)
        for repeat in range(3):
            prefix = f'{name}.repeat{repeat}'
            process = old/f'{prefix}.process.json'
            row = old/f'{prefix}.jsonl'
            if not process.exists() and not row.exists():
                continue
            valid = process.exists() and row.exists()
            if valid:
                result = json.loads(process.read_text())
                valid = result['valid'] and result['returncode'] == 0
            if not valid:
                print(f'Leaving invalid/incomplete measurement in {old}/{prefix}.')
                continue
            # Keep CPU-DCT results for reporting only. The launcher checks
            # selected IC/FS measurements before skipping them.
            for path in old.glob(prefix+'.*'):
                (current/path.name).symlink_to(path)
            retained += 1
cache = previous/'verification_cache'
if cache.is_dir():
    (output/cache.name).symlink_to(cache)
print(f'Retained {retained} completed measurements; original logs remain in {previous}.')
PY
echo "Continuing 50K: IC/IC-b/FS only; existing CPU-DCT results retained; IC cache 1024 MiB per worker. Results: $output"
bash fastlens/benchmarks/baselines/dct_baselines/run_inference_pilot.sh "$prepared" "$output" \
  --ic-protocol --expected-images 50000 --settings "$prepared/reader_settings.json" \
  --systems "ic_shared ic_batched fs" --resume-measurements --wait-for-gpu \
  --measurement-cache-mib ic_shared=1024 --measurement-cache-mib ic_batched=1024 \
  2>&1 | tee "$output/driver.log"
