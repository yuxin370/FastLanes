#!/usr/bin/env bash
# Standard DCT A0 plus the same RGB adapters used by the Transformer suite.
set -euo pipefail
cd "$(dirname "$0")/../../.."
if [[ "${3:-}" == *efun:* && "${1:-probe}" != plan ]]; then
  : "${CUDA_VISIBLE_DEVICES:?Select a confirmed idle GPU UUID before running eFUN training}"
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-GPU-40c637bd-acf5-ea1a-0df8-617138228467}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/tangyuxin/tmp/dctnet MPLCONFIGDIR=/home/tangyuxin/tmp/matplotlib
PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
EXP=galp/experiments/dct_pushdown_inference
DATA=galp/data/system_rgbnomore/e2e_v3
MODE=${1:-probe}
RUN=${2:-rtx4090_cnn_training_baselines_20260914}
PHYSICAL=/mnt/nvme2/home/tangyuxin/pls-experiments/physical-layout-full-premix-orgseed-20260810/uniform_premix
mapfile -t INPUTS < <("$PYTHON" - "$PHYSICAL/materialization_contract.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); c=json.loads(p.read_text())
print(p.parent/'dct/manifest.bin')
print(c['ordered_mapping'])
print(c['ordered_mapping_sha256'])
PY
)
case "$MODE" in
  probe) EXTRA=(--max-pools 2 --validation-count 1000) ;;
  full|plan) EXTRA=(--epochs 2 --validation-count 50000) ;;
  data) EXTRA=(--max-pools 5 --data-only) ;;
  profile) EXTRA=(--profile) ;;
  *) exit 2 ;;
esac
SPECS=${3:-"mobilenet24:jpeg mobilenet32:jpeg resnet24:jpeg resnet64:jpeg mobilenet24:rgb_pytorch mobilenet24:rgb_d2 mobilenet24:rgb_d3 resnet24:rgb_pytorch resnet24:rgb_d2 resnet24:rgb_d3"}
for spec in $SPECS; do
  IFS=: read -r DCTNET_PROFILE BACKEND <<< "$spec"
  export DCTNET_PROFILE
  DRIVER=("$PYTHON" "$EXP/training_pls.py")
  MODEL_EXTRA=()
  BLOCKS=32768
  case "$DCTNET_PROFILE" in
    mobilenet*) NAME=dctnet_$DCTNET_PROFILE ;;
    resnet*) NAME=dctnet_static${DCTNET_PROFILE#resnet} ;;
    efun)
      NAME=efun
      DRIVER=("$PYTHON" "$EXP/efun.py" training_pls)
      BLOCKS=0
      if [[ "$MODE" == full || "$MODE" == plan ]]; then MODEL_EXTRA=(--initial-validation); fi
      ;;
  esac
  CONDITION=A0; WORKERS=${CNN_DATA_WORKERS:-16}
  if [[ "$BACKEND" == native ]]; then CONDITION=B6; fi
  if [[ "$BACKEND" == rgb_* ]]; then CONDITION=RGB; fi
  if [[ "$BACKEND" == rgb_d* ]]; then WORKERS=4; fi
  if [[ "$DCTNET_PROFILE" == efun && "$BACKEND" == rgb_d2 ]]; then
    WORKERS=16
    MODEL_EXTRA+=(--dali-prefetch-depth 4)
  fi
  OUT=$DATA/runs/$NAME/$RUN/${BACKEND}_${MODE/plan/full}
  if [[ "$MODE" != plan ]]; then mkdir -p "$OUT"; fi
  if [[ -f "$OUT/training.json" || ( "$MODE" == profile && -f "$OUT/breakdown.json" ) ]]; then continue; fi
  RESUME=()
  for epoch in 1 0; do
    if [[ ( "$MODE" == full || "$MODE" == plan ) && -f "$OUT/epoch_$epoch.pth" ]]; then
      RESUME=(--resume "$OUT/epoch_$epoch.pth")
      break
    fi
  done
  COMMAND=("${DRIVER[@]}" --manifest "${INPUTS[0]}" --mapping "${INPUTS[1]}"
    --mapping-sha256 "${INPUTS[2]}" --condition "$CONDITION" --input-backend "$BACKEND"
    --workers "$WORKERS" --segments-per-pool 4 --transform-blocks-per-launch "$BLOCKS" --output-dir "$OUT" "${EXTRA[@]}" "${MODEL_EXTRA[@]}" "${RESUME[@]}")
  if [[ "$MODE" == plan ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    continue
  fi
  if [[ "$MODE" == profile ]]; then
    if [[ ! -f "$OUT/capture.nsys-rep" ]]; then
      nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop --output "$OUT/capture" \
        "${COMMAND[@]}" > "$OUT/run.log" 2>&1
    fi
    if [[ ! -f "$OUT/capture.sqlite" ]]; then
      nsys export --type sqlite --output "$OUT/capture.sqlite" "$OUT/capture.nsys-rep"
    fi
    "$PYTHON" "$EXP/analyze_training_capture.py" "$OUT/capture.sqlite" --output "$OUT/breakdown.json" \
      --label "$DCTNET_PROFILE / $BACKEND" --timeline "$OUT/timeline.png"
  else
    "${COMMAND[@]}" > "$OUT/run.log" 2>&1
  fi
  cat "$OUT/run.log"
done
