#!/usr/bin/env bash
# Host-only numerical checks on the exported synthetic JPEG fixture.
# These are not ImageNet performance measurements or a FastLens CUDA gate.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
python_bin=${PYTHON:-/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76}
evidence="$repo_root/galp/ImageCalulatorRe-implement/round-2-response/evidence-v2"
run_root=${1:?Usage: bash run_cuda_gates.sh NEW_OUTPUT_DIRECTORY}
mkdir "$run_root"
run_root=$(cd "$run_root" && pwd)

for model in mobilenet24 resnet24 vitti; do
    model_dir="$run_root/$model"
    mkdir "$model_dir"
    config="$evidence/models/$model/contract/matched.json"
    snapshot="$model_dir/initial.pt"
    # Recreate on the host to capture the visible CUDA device's RNG state.
    "$python_bin" -B -m galp.benchmarks.dct_baselines snapshot \
        --config "$config" --output "$snapshot" 2>&1 | tee "$model_dir/snapshot.log"
    for method in ic_matched_shared ic_matched_batched fs_matched_b8; do
        backend=ic
        if [[ "$method" == fs_matched_b8 ]]; then backend=fs; fi
        common=(--config "$config" --snapshot "$snapshot" --backend "$backend"
                --root "$evidence/stores/$method" --device cuda:0 --updates 10)
        calibration="$model_dir/$method.calibration.json"
        "$python_bin" -B -m galp.benchmarks.dct_baselines calibrate "${common[@]}" \
            --input-atol 0 --model-atol 0 --state-atol 0 --rtol 0 \
            --output "$calibration" 2>&1 | tee "$model_dir/$method.calibrate.log"
        "$python_bin" -B -m galp.benchmarks.dct_baselines verify-training "${common[@]}" \
            --calibration "$calibration" --output "$model_dir/$method.verified.json" \
            2>&1 | tee "$model_dir/$method.verify.log"
    done
done
printf 'Passed: 3 models x 3 store variants x 10 CUDA updates. Evidence: %s\n' "$run_root"
