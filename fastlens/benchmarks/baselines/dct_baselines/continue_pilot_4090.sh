#!/usr/bin/env bash
# Complete the interrupted 8194-image pilot, retaining its two valid repeats.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES=GPU-40c637bd-acf5-ea1a-0df8-617138228467
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$HOME/tmp"
old="$HOME/tmp/dct-p0-pilot-4090-20261012-150553"
retry="$HOME/tmp/dct-p0-fastlens-retry-4090-20261012-173709"
prepared="$HOME/tmp/dct-round3-prod-h100-20261010-012949"
output="$HOME/tmp/dct-p0-pilot-4090-continued-$(date +%Y%m%d-%H%M%S)"
mkdir "$output"
mkdir "$output/resnet24"
# Links retain the original artifact paths/hashes and leave failed logs intact.
for name in cpu_dct fastlens ic_shared ic_batched fs; do
  for suffix in contract verified.json; do
    path="$old/resnet24/$name.$suffix"
    if [[ ! -e "$path" ]]; then echo "Missing input: $path" >&2; exit 1; fi
    ln -s "$path" "$output/resnet24/$name.$suffix"
  done
done
for path in "$old"/resnet24/cpu_dct.repeat0.* "$retry"/fastlens.repeat0.*; do
  if [[ ! -f "$path" ]]; then echo "Missing repeat artifact: $path" >&2; exit 1; fi
  ln -s "$path" "$output/resnet24/$(basename "$path")"
done
echo "Continuing pilot: 2 valid repeats retained; 28 measurements remain. Results: $output"
bash fastlens/benchmarks/baselines/dct_baselines/run_inference_pilot.sh "$prepared" "$output" \
  --ic-protocol --expected-images 8194 --resume-measurements \
  2>&1 | tee "$output/driver.log"
