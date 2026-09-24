#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=${HOME}/tmp/dctnet MPLCONFIGDIR=${HOME}/tmp/matplotlib
PYTHON=${PYTHON:-python3}
EXPERIMENT=galp/benchmarks/dct_models
DATA=galp/data/system_rgbnomore/e2e_v3
OUT=$DATA/runs/dctnet_static64/rgb_comparison/steady
mkdir -p "$OUT/nsys" "$OUT/profile_runs"
for route in rgb_pytorch rgb_dali dct_reference galp_legacy galp_native; do
  case "$route" in
    rgb_pytorch|rgb_dali)
      command=("$PYTHON" "$EXPERIMENT/evaluate_rgb.py" --route "${route#rgb_}" --workers 64);;
    dct_reference)
      command=("$PYTHON" "$EXPERIMENT/evaluate.py" --route R --device cuda --workers 64);;
    galp_legacy)
      command=("$PYTHON" "$EXPERIMENT/evaluate.py" --route N --device cuda --workers 64
        --new-data "galp/data/compressed/backup/offline_model_input/imagenet512_val_resnet56_compact_v3");;
    galp_native)
      command=("$PYTHON" "$EXPERIMENT/evaluate_shards.py"
        --new-data "galp/data/compressed/backup/offline_model_input/imagenet512_val_resnet56_block_major");;
  esac
  echo "Capturing $route"
  if [[ ! -f "$OUT/nsys/$route.nsys-rep" ]]; then
    nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --output "$OUT/nsys/$route" \
      "${command[@]}" --count 50000 --batch-size 64 --profile \
      --output-dir "$OUT/profile_runs/$route" > "$OUT/nsys/$route.log" 2>&1
  fi
  if [[ ! -f "$OUT/nsys/$route.sqlite" ]]; then
    nsys export --type sqlite --output "$OUT/nsys/$route.sqlite" "$OUT/nsys/$route.nsys-rep"
  fi
  echo "Finished $route"
done
