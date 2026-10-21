#!/usr/bin/env bash
# Bounded CPU tuning: 22 cases, not the Cartesian product of all controls.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
: "${PYTHON:?Set PYTHON to the fastlanes-cuda Python executable}"
prepared="${1:?Usage: run_reader_tuning.sh PREPARED NEW_OUTPUT}"
output="${2:?Usage: run_reader_tuning.sh PREPARED NEW_OUTPUT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$HOME/tmp"
cpus="${CPU_LIST:-$("$PYTHON" -B -c 'import os; c=sorted(os.sched_getaffinity(0)); assert len(c)>=8; print(",".join(map(str,c[:8])))')}"
python=(taskset -c "$cpus" "$PYTHON" -B)
mkdir "$output"
prepare_case() {
  "${python[@]}" -m fastlens.benchmarks.baselines.dct_baselines prepare \
    --source-manifest "$prepared/contract/source.json" --model vitti --upstream "$HOME/RGB-no-more" \
    --batch-size 64 --pool-size "$2" --workers "$3" --torch-threads 1 --cpu-loader dataloader \
    --prefetch-factor 2 --decoded-cache-mib "$4" --decode-budget-mib 1024 --reader-budget-mib 1536 \
    --host-input-budget-mib 32768 --request-order "$5" --seed 11997734 \
    --atol 1e-5 --rtol 1e-5 --output "$output/$1.contract" > "$output/$1.prepare.log"
}
run_case() {
  echo "Reader-only: $1"
  "${python[@]}" -m fastlens.benchmarks.baselines.dct_baselines reader-bench \
    --config "$output/$1.contract/matched.json" --root "$2" --repeats 3 \
    --output "$output/$1.reader.json" > "$output/$1.reader.log" 2>&1
}
for order in source shuffle; do
  for name in ic_shared ic_batched fs; do
    case "$name" in
      ic_shared) root="$prepared/ic_matched_shared" ;;
      ic_batched) root="$prepared/ic_matched_batched" ;;
      fs) root="$prepared/fs_matched_b8" ;;
    esac
    case_name="$name.$order"
    prepare_case "$case_name" 256 8 0 "$order"
    run_case "$case_name" "$root"
  done
done
# Reuse the same source and serializer. Only the IC-b storage group changes.
for group in 32 64; do
  root="$output/ic_group$group"
  "${python[@]}" -m fastlens.benchmarks.baselines.dct_baselines build \
    --config "$output/ic_batched.source.contract/matched.json" \
    --method ic_matched_batched --group-size "$group" --workers 8 --output "$root" \
    > "$output/ic_group$group.build.json" 2>&1
  for order in source shuffle; do
    case_name="ic_batched.group$group.$order"
    prepare_case "$case_name" 256 8 0 "$order"
    run_case "$case_name" "$root"
  done
done
# Compare a cache and a wider preparation pool without an exhaustive grid.
for order in source shuffle; do
  for name in ic_batched fs; do
    if [[ "$name" == fs ]]; then root="$prepared/fs_matched_b8"; else root="$prepared/ic_matched_batched"; fi
    for setting in cache512 pool512 workers4; do
      pool=256; workers=8; cache=0
      case "$setting" in
        cache512) cache=512 ;;
        pool512) pool=512 ;;
        workers4) workers=4 ;;
      esac
      case_name="$name.$setting.$order"
      prepare_case "$case_name" "$pool" "$workers" "$cache" "$order"
      run_case "$case_name" "$root"
    done
  done
done
"${python[@]}" - "$output" <<'PY'
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob('*.reader.json')):
    d = json.loads(path.read_text())
    runs = d['runs']
    rates = [run['images_s'] for run in runs]
    rows.append(dict(case=path.stem, method=d['method'], request_order=d['config']['request_order'],
        group=d['storage_group_images'], pool=d['config']['preparation_pool_size'],
        workers=d['config']['workers'], cache_per_reader=d['config']['decoded_cache_bytes'],
        median_images_s=statistics.median(rates), stdev_images_s=statistics.stdev(rates),
        median_decode_amplification=statistics.median(r['decode_amplification'] for r in runs),
        raw_result=str(path.resolve())))
(root/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
for row in rows:
    print(row['case'], round(row['median_images_s'], 2), 'images/s; decode amplification', round(row['median_decode_amplification'], 2))
PY
echo "Passed: bounded reader tuning. Results: $output/summary.json"
