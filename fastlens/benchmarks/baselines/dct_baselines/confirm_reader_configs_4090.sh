#!/usr/bin/env bash
# Confirm decoded-cache candidates with real inference under the common RAM cap.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES=GPU-40c637bd-acf5-ea1a-0df8-617138228467
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$HOME/tmp"
prepared="$HOME/tmp/dct-round3-prod-h100-20261010-012949"
prior="$HOME/tmp/dct-p0-pilot-4090-continued-20261012-181559"
output="$HOME/tmp/dct-reader-confirm-4090-$(date +%Y%m%d-%H%M%S)"
cpus="$("$PYTHON" -B -c 'import os; c=sorted(os.sched_getaffinity(0)); assert len(c)>=8; print(",".join(map(str,c[:8])))')"
python=(taskset -c "$cpus" "$PYTHON" -B)
# name:cache MiB per worker:reader MiB per worker. Reader budgets include cache.
# Within each pair only cache capacity changes; all cases share the 32 GiB cap.
cases=(ic_shared:0:2560 ic_shared:2048:2560
       ic_batched:0:2816 ic_batched:2048:2816 fs:0:1536 fs:512:1536)
mkdir "$output"
check_gpu() {
  "${python[@]}" - "$CUDA_VISIBLE_DEVICES" <<'PY'
import subprocess, sys
uuid = sys.argv[1]
subprocess.run(['nvidia-smi', '--id='+uuid, '--query-gpu=name', '--format=csv,noheader'], check=True)
text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                '--format=csv,noheader,nounits'], text=True)
occupied = [row for row in text.splitlines() if row.split(',')[0].strip() == uuid]
if occupied:
    raise SystemExit('GPU busy; wait until idle: '+str(occupied))
PY
}
select_case() {
  IFS=: read -r name cache budget <<< "$1"
  case "$name" in
    ic_shared) backend=ic; root="$prepared/ic_matched_shared" ;;
    ic_batched) backend=ic; root="$prepared/ic_matched_batched" ;;
    fs) backend=fs; root="$prepared/fs_matched_b8" ;;
  esac
  directory="$output/$model/$name.cache$cache"
  contract="$directory/contract/matched.json"
  evidence="$directory/verified.json"
}
echo "Reader confirmation: 2 models x 6 configurations x 3 repeats. Results: $output"
for model in resnet24 vitti; do
  for item in "${cases[@]}"; do
    select_case "$item"
    mkdir -p "$directory"
    if [[ "$model" == resnet24 && "$name" == ic_shared ]]; then
      # These unchanged contracts passed before the IC-b budget failure.
      passed="$HOME/tmp/dct-reader-confirm-4090-20261013-033441/$model/$name.cache$cache"
      ln -s "$passed/contract" "$directory/contract"
      ln -s "$passed/verified.json" "$evidence"
      echo "Reusing passed verification: $passed/verified.json"
      continue
    fi
    "${python[@]}" -m fastlens.benchmarks.baselines.dct_baselines prepare \
      --source-manifest "$prepared/contract/source.json" --model "$model" --upstream "$HOME/RGB-no-more" \
      --batch-size 64 --pool-size 256 --workers 8 --torch-threads 2 \
      --cpu-loader dataloader --prefetch-factor 2 --decoded-cache-mib "$cache" \
      --decode-budget-mib 1024 --reader-budget-mib "$budget" --host-input-budget-mib 32768 \
      --request-order source --seed 11997734 --atol 1e-5 --rtol 1e-5 \
      --output "$directory/contract" > "$directory/prepare.log" 2>&1
    check_gpu
    echo "Verifying $model/$name cache=$cache MiB per worker"
    "${python[@]}" -m fastlens.benchmarks.baselines.dct_baselines verify-inference \
      --config "$contract" --backend "$backend" --root "$root" --device cuda:0 \
      --source-evidence "$prior/$model/$name.verified.json" --output "$evidence" \
      2>&1 | tee "$directory/verify.log"
  done
  for repeat in 0 1 2; do
    for ((position=0; position<${#cases[@]}; position++)); do
      select_case "${cases[$(( (position + repeat) % ${#cases[@]} ))]}"
      check_gpu
      echo "Measuring $model/$name cache=$cache MiB per worker repeat $((repeat + 1))/3"
      "${python[@]}" -m fastlens.benchmarks.dct_models.measure_process \
        --output "$directory/measurement.repeat$repeat.process.json" \
        --gpu-uuid "$CUDA_VISIBLE_DEVICES" --rss-limit-mib 32768 -- \
        "$PYTHON" -B -m fastlens.benchmarks.baselines.dct_baselines bench \
        --config "$contract" --backend "$backend" --root "$root" --evidence "$evidence" \
        --device cuda:0 --run-kind inference --repeats 1 \
        --warmup-seconds 60 --min-seconds 180 --observe-cache \
        --output "$directory/measurement.repeat$repeat.jsonl" \
        2>&1 | tee "$directory/measurement.repeat$repeat.log"
      check_gpu
    done
  done
done
"${python[@]}" -m fastlens.benchmarks.baselines.dct_baselines report "$output"/*/*/*.jsonl --output "$output/summary.json"
echo "Passed: reader configuration inference confirmation. Results: $output/summary.json"
