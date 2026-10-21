#!/usr/bin/env bash
# Freeze the existing full validation JPEGs and build the three matched stores.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export PYTHON="${PYTHON:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$HOME/tmp"
output="${1:-$HOME/tmp/dct-50k-$(date +%Y%m%d-%H%M%S)}"
recipe="$HOME/tmp/dct-round3-prod-h100-20261010-012949/source_recipe.json"
selection="$PWD/fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json"
cpus="${CPU_LIST:-$("$PYTHON" -B -c 'import os; c=sorted(os.sched_getaffinity(0)); assert len(c)>=8; print(",".join(map(str,c[:8])))')}"
python=(taskset -c "$cpus" "$PYTHON" -B)
test -f "$recipe"
test -f "$selection"
mkdir "$output"
output="$(cd "$output" && pwd)"
cp fastlens/benchmarks/baselines/dct_baselines/p0_reader_settings.json "$output/reader_settings.json"
echo "Preparing 50K: CPU affinity $cpus; 8 build workers. Results: $output"
"${python[@]}" -m fastlens.benchmarks.baselines.dct_baselines prepare \
  --selection "$selection" --source-recipe "$recipe" --source-workers 8 \
  --model vitti --upstream "$HOME/RGB-no-more" --batch-size 64 --pool-size 256 \
  --workers 8 --cpu-loader dataloader --prefetch-factor 2 --torch-threads 2 \
  --decode-budget-mib 1024 --reader-budget-mib 1024 --host-input-budget-mib 32768 \
  --request-order source --seed 11997733 --atol 1e-5 --rtol 1e-5 \
  --output "$output/contract" 2>&1 | tee "$output/prepare.log"
"${python[@]}" - "$output/contract/matched.json" <<'PY'
import sys
from fastlens.benchmarks.baselines.dct_baselines.runner import inputs
from fastlens.benchmarks.baselines.dct_baselines.source import validate_population
config, source, profile, records, trace = inputs(sys.argv[1])
print(validate_population(source, trace, expected_images=50000, expected_classes=1000))
PY
# Each store is built by eight processes. Run stores sequentially so their
# construction costs are measured without competition from the other builds.
for method in ic_matched_shared ic_matched_batched fs_matched_b8; do
  group=256
  if [[ "$method" == ic_matched_shared ]]; then group=1; fi
  echo "Building $method: group=$group, workers=8"
  "${python[@]}" -m fastlens.benchmarks.baselines.dct_baselines build \
    --config "$output/contract/matched.json" --method "$method" \
    --group-size "$group" --workers 8 --dtype-policy paper --output "$output/$method" \
    2> >(tee "$output/$method.build.log" >&2) | tee "$output/$method.build.json"
done
echo "Passed: 50000 images, 1000 classes, three parallel-built stores. Results: $output"
echo 'Next, when H100 and the selected CPU/storage resources are idle:'
printf 'export PYTHON=%q\n' "$PYTHON"
echo 'export CUDA_VISIBLE_DEVICES=GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76'
printf 'bash fastlens/benchmarks/baselines/dct_baselines/run_inference_pilot.sh %q %q --ic-protocol --expected-images 50000 --settings %q\n' \
  "$output" "$output/h100" "$output/reader_settings.json"
