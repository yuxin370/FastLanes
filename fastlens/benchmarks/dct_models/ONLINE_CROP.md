# CNN inference with online DCT crop

This supporting experiment uses all 50,000 ImageNet-512 validation images and
one full-frequency source store shared by eFUN, MobileNetV2 DCT24/DCT32 and
ResNet-50 DCT24/DCT64. Offline preparation extracts source JPEG coefficients,
compresses them losslessly and builds access indexes. Crop, resize, model-channel
selection and normalization happen online.

## Input and comparison

The source-coordinate center crop is `(32,32,448,448)`. Output grids are 28x28
blocks for eFUN, 56x56 for ResNet and 112x112 for MobileNet; chroma is expanded
from the source 4:2:0 geometry. DCT resampling has its own numerical semantics,
so accuracy is evaluated using the resulting inputs.

| Path | Read/decode scope | Output |
| --- | --- | --- |
| `crop_off/grid` | Full source | Full transformed grid, followed by channel selection |
| `crop_off/projected` | Full source | Fused projected model tensor |
| `crop_on/grid` | Crop support | Same transformation, full grid |
| `crop_on/projected` | Crop support | Same transformation, fused projection |

Resize requires the full source-frequency dependencies. `require_all_coefficients`
therefore keeps all 64 source columns. Output-channel projection and source-column
selection are separate experiments. All four paths use the same model shape and
computation. Full-source decoding still performs the same crop/resize; it only
removes storage-range pruning.

## Run

Prepare the ImageNet inputs and public model dependencies using the
[rebuilding guide](../paper/REBUILDING_REFERENCES.md). Build the JPEG converter,
access-index tool, Torch extension and storage bridge through the root CMake project.

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$HOME/tmp" MPLCONFIGDIR="$HOME/tmp/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
SOURCE=fastlens/data/compressed/imagenet512_val_block_major
python -m fastlens.benchmarks.dct_models.prepare_online_source \
  --output-dir "$SOURCE"
python -m fastlens.benchmarks.dct_models.online_crop_suite \
  --source-data "$SOURCE" --output-dir "$HOME/tmp/fastlens-online-crop"
```

Preparation uses the CPU affinity budget by default; `--threads` sets the total
budget. Use a new source directory when rebuilding. The source verifier must
report exact coefficient agreement.

Each model calibrates supported activation sizes on the same 8,192-image prefix,
using three trials and a common configuration for all four ablations. Full runs
use batch 64, FP32, TF32 disabled and three rotating repeats. Run them on an idle
GPU. The calibration selects among measured candidates, not all possible settings.

## Outputs

- `generation.json` records preparation time and storage size.
- `calibration/selection.json` records candidate timings and the selected configuration.
- `verify/` contains input checks for each crop/output combination.
- Full runs retain predictions, accuracy, timing and source-read counts.
- `profile/` contains separate 4,096-image Nsight captures.

Profiling results are separate from unprofiled throughput. The runner retains
page-cache behavior; its file-range counters are not cold-storage traffic.
For CPU checks, run
`python -m unittest fastlens.benchmarks.dct_models.tests.test_online_crop`.
