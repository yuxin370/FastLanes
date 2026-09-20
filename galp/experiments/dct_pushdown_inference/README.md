# Fixed DCTNet / GALP experiment

Base eFUN inference, author-recipe training and the shared GALP A0/B6 training
entry points are documented in [EFUN.md](EFUN.md).
The [Chinese model/results summary](EFUN_RESULTS_AND_TRAINING_ZH.md) also
provides the configured eFUN/RGB EfficientNet-B0 training matrix.

## End-to-end training comparison: agreed scope

Training uses DCT-domain transforms. GALP is allowed to use its native B6
grouped crop and delayed closed-pool shuffle; equality to upstream pixel-domain
training preprocessing is not required. The convergence comparison is **A0
versus B6**, using the existing `training_pls.matrix` definitions:

| Arm | Crop | Epoch ordering | Purpose |
|---|---|---|---|
| A0 standard DCT training | independent per image | global shuffle | convergence control |
| B6 native GALP training | shared per physical group | existing closed-pool delayed shuffle | target system |

Start with the existing G=1024, M=4 B6 configuration, subject to the DCTNet
training memory check. The native implementation must preserve bounded worksets
and must not retain four complete 192-channel float grids just to implement the
shuffle. The existing configuration is a starting point, not proof of global
optimality for these larger input grids.

The four current model configurations are ResNet-50 DCT-24/64 and MobileNetV2
DCT-24/32. For each pair, hold model initialization, full training population,
class mapping, precision, optimizer/update schedule, augmentation operation
distribution, and validation preprocessing fixed. Crop sharing and ordering are
the intended differences; Mixup partners may differ because ordering differs.
Do not compare different initial checkpoints to claim a scheduling effect.
RGB PyTorch/DALI remain system baselines, not the same-model convergence control.

The earlier SwinV2 `report_convergence_reference.py` requires identical crop and
order contracts in its two arms. Its B6-versus-B6 reference does not establish
A0-versus-B6 convergence. Do not weaken those checks or reuse that conclusion
for this comparison.

The existing full-frequency, full-source training GALP and its premixed B6
layout can be used as the training mother representation, with DCTNet geometry
constructed online. This is a training-specific source-DCT path, distinct from
the previous inference N version that precomputed target grids. It avoids
requiring a new expanded full-training dataset merely to begin this comparison.
The raw-source-to-DCTNet training profile still needs implementation: do not
run the full RGB-no-more 28/14-block normalized profile and enlarge it afterward.
Keep shared DCT transform/augmentation semantics between A0 and B6. Apply
coefficient pushdown only after accounting for transform and augmentation
dependencies, then select and normalize the model's output channels.

Two complete epochs remain the initial performance and early-convergence
observation. At effective batch 1024 they contain 2504 optimizer updates, below
the existing 10000-update warmup. Report validation Top-1/Top-5/CE at epoch 0,
1 and 2, curves against processed images and training wall time, and the B6-A0
difference. This is not proof of final convergence or statistical
non-inferiority. Keep the long recipe and epoch-boundary checkpoints so a
longer paired observation can resume without changing the schedule. Do not
extend the run, add seeds, select a favorable endpoint, or declare
non-inferiority merely because two early curves are close. A formal
non-inferiority claim requires a prespecified horizon, margin and adequate
uncertainty evidence; the historical 0.3 pp margin is only a candidate until
chosen for this experiment.

This section specifies the comparison. It does not mean the native DCTNet
training profile or the full A0/B6 runs have been implemented or completed.

## Training model calibration

`training_model_only.py` reuses the Transformer experiment's
`run_model_only_calibration`, published optimizer/scheduler and audit policy.
It enables training and gradients on a strictly loaded official checkpoint,
uses BF16 autocast, Inductor, microbatch 64 and accumulation 16, and checks
that parameters changed. The default is 5 warmup and 120 measured optimizer
updates. Calibration updates are discarded; source checkpoints are unchanged.

The fixed inputs are 64 real training-split images prepared with the reference
center crop. They reside on the GPU throughout timing. This excludes reading,
decoding, augmentation and H2D, and is **not an end-to-end training benchmark**
or a completed training epoch. The separate full training matrix still needs
DCTNet-compatible training geometry and augmentation: the Transformer B6
28/14-block contract cannot be passed directly to these models, and existing
precomputed DCTNet GALP versions contain validation images only.

Run all four checkpoint configurations (ResNet-50 DCT-24/64, MobileNetV2
DCT-24/32) plus one RGB calibration per backbone on RTX 4090:

```bash
bash galp/experiments/dct_pushdown_inference/training_model_only_matrix.sh
```

Outputs use existing `e2e_v3/runs/{model}/rtx4090_training_20260914/model_only/`
directories. A JSON result is written only after calibration completes. A log
or started process does not establish completion.

The original 5-warmup/120-measured window includes 95 updates with strict
auditing and 25 after the shared audit transition. Measure a separate window
entirely after that transition without disabling checks:

```bash
bash galp/experiments/dct_pushdown_inference/training_model_only_matrix.sh \
  rtx4090_training_steady_20260914 --warmup-updates 105 --measured-updates 120
```

The matrix skips completed JSON results in the selected run directory. Use a
different run name when changing measurement arguments. The existing
`audit_seconds_including_warmup` field includes warmup and must not be divided
by measured wall time to report an audit percentage.

`training_input_probe.py` separately checks eight real training images through
the generic native reader at several source crop extents, including flips.
It compares full-grid and projected inputs to an independent source-coefficient
DCT upsampling reference, then runs one backward/update with native-owned input.
All source frequencies are retained because resizing mixes frequencies. This
is an input correctness probe, not an implementation or timing of the B6 PLS
training loop, and must run separately from performance calibration.

The first unrounded-reference probe failed: `finalize_dct_grid_float` rounds
and saturates to int16 range even for float32 output. The current probe reports
the difference from that unrounded reference and checks the explicit rounding
contract (including FP32 half-integer boundary uncertainty), while requiring
grid/projected outputs to match exactly. It does not claim equality with the
unrounded reference. Completed calibration and probe results, the preserved
initial failure, and remaining E2E work are summarized in
`e2e_v3/runs/dctnet_mobilenet24/rtx4090_training_steady_20260914/REPORT.md`.

## DCT-24 and file I/O pushdown

`DCTNET_PROFILE=resnet24` and `DCTNET_PROFILE=mobilenet24` strictly load the
official 24-channel checkpoints from the existing checkpoint cache. Both use
16 Y + 4 Cb + 4 Cr channels, at 56×56 and 112×112 respectively. They reuse
`dct_major_dctnet_static64` and `dct_major_dctnet_mobilenet32`: the mother data
contains every frequency before selection and normalization. The reader checks
the complete stored numerical/geometry contract independently of the model
profile, and records both profiles in its result. No data or manifest is rewritten.

Standard FLS whole-rowgroup reads now propagate explicit column selections to
file I/O. The shared reader resolves alias/dictionary dependencies and coalesces
selected segment entrypoint/data ranges directly from the footer. It does not
build a per-vector index or read the omitted columns first. Shard activation and
the bounded native pipeline are unchanged; `--pushdown off` retains full reads.
The current coefficient API selects the union across components (16 of 64
frequency columns for DCT-24); final projected output contains exactly 24 channels.

`complete_dct24.sh` runs the two complete 50K baseline matrices on PRO 6000,
using the existing browser pause helper before formal timing. Results are under
`e2e_v3/runs/{dctnet_static24,dctnet_mobilenet24}/pro6000_io_pushdown_20260913/`.

The same runner accepts a GPU UUID and run directory name. The RTX 4090 run is:

```bash
bash galp/experiments/dct_pushdown_inference/complete_dct24.sh \
  GPU-40c637bd-acf5-ea1a-0df8-617138228467 rtx4090_io_pushdown_20260914
```

It runs the same 16 frozen-checkpoint inference evaluations, with 50K images,
batch 64, FP32 and TF32 disabled. Results use `full/` and monitors use
`monitoring/`; the browser on PRO 6000 is not paused for a 4090 run.

## Official MobileNetV2 DCT-32

Set `DCTNET_PROFILE=mobilenet32` for the official 32-channel checkpoint;
the default remains the existing ResNet-50 DCT-64 contract. DCT-32 is
MobileNetV2, not a reduced-channel ResNet-50. Its official preprocessing is
Resize(1024), CenterCrop(896), Upscale(2), Q100 4:2:0 JPEG extraction, and
the existing official per-channel normalization. Y comes from the 896px
branch and Cb/Cr from the 1792px branch. All three stored grids are 112×112,
with all 64 quantized frequencies retained per component.

The official checkpoint is cached at
`e2e_v2/checkpoints/mobilenetv2dct_upscaled_static_32/model_best.pth.tar`.
Its metadata identifies `mobilenetv2dct_subset_woinp`, epoch 146. The wrapper
bypasses only the constructor's mandatory RGB initialization load and strictly
restores the entire official DCT state; the upstream network and forward are
unchanged. `profile.json` records the precise 22 Y + 5 Cb + 5 Cr indices and
normalization parameters.

The same block-major writer now accepts the target grid size as an optional
final CLI argument (default 56); the generator passes the selected profile's
grid. The same native reader and projected-output kernels are reused, without
model-specific CUDA changes. Coefficient pushdown uses the 22-frequency union
across the three components. The original measurements read full rowgroups;
the file-I/O extension above also reduces the selected column file reads. Use projected off/on
to isolate coefficient selection while keeping the final output layout fixed.
The O adapter supports both 56×56 and 112×112 target grids.

PRO 6000 commands, from the project root:

```bash
export DCTNET_PROFILE=mobilenet32
export CUDA_VISIBLE_DEVICES=GPU-e796262d-3449-6af1-586d-8460d8836d1b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/home/tangyuxin/tmp/dctnet MPLCONFIGDIR=/home/tangyuxin/tmp/matplotlib
PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
EXP=galp/experiments/dct_pushdown_inference
DATA=galp/data/system_rgbnomore/e2e_v3
OUT=$DATA/runs/dctnet_mobilenet32/pro6000/full
$PYTHON "$EXP/generate.py" --count 50000 --layout block-major --shard-images 1024 \
  --workers 16 --shard-workers 4 --encoding-threads 8 \
  --output-dir "$DATA/dct_major_dctnet_mobilenet32"
# The same generation command resumes completed shards.
build/galp/tools/jpeg_dct/galp_block_major_access_tool \
  "$DATA/dct_major_dctnet_mobilenet32/manifest.bin" \
  --output-dir "$DATA/dct_major_dctnet_mobilenet32/access" \
  --output-json "$DATA/dct_major_dctnet_mobilenet32/access.json"
$PYTHON "$EXP/evaluate.py" --route R --device cuda --count 50000 \
  --batch-size 64 --workers 64 --output-dir "$OUT"
for variant in grid_off projected_off projected_on; do
  layout=${variant%_*}; mode=${variant##*_}
  $PYTHON "$EXP/evaluate_shards.py" --count 50000 --batch-size 64 \
    --new-data "$DATA/dct_major_dctnet_mobilenet32" \
    --output-layout "$layout" --pushdown "$mode" \
    --baseline-dir "$OUT" --output-dir "$OUT/$variant"
done
```

## Existing ResNet-50 DCT-64

This experiment uses the official `ResNetDCT_Upscaled_Static` 64-channel checkpoint,
strictly restored into the unchanged official network. No training is performed.
The upstream checkout is `DCTNet/` at revision
`bd7c669b478e47fde230119045133d10e135de97`. Only Python ABC and libturbojpeg path
compatibility is handled in the wrapper. The input contract, actual means/stds,
and channel indices are recorded in each output's `profile.json`.

`backend.py` imports existing benchmark path defaults. Selection uses the existing
validation population and seed 11997733 shuffle. Original `label`, image IDs, and
paths are preserved; `model_label` explicitly maps sorted WNIDs to official
checkpoint class indices. The two mappings are different (e.g. n01440764 is 448
in the old index and 0 for this checkpoint).

R invokes official resize/crop/upscale, JPEG extraction, channel selection and
normalization code. N stores all 64 frequencies of each of Y/Cb/Cr before model
normalization. It uses quantized int16 plus natural-order quantization tables;
the bridge maps frequencies to GALP zigzag columns and back. All target component
grids are 56×56: Y is from the 448px branch and chroma from the 896px branch.
These are target component grids, not the original source 4:2:0 geometry.

O reads original GALP coefficients, dequantizes once, crops Y64→56 and chroma32→28,
then invokes the existing DCT-domain 2× chroma upsampling mathematics in float32.
It applies no clamp, rounding, RGB-no-more normalization or patch basis transform.
O is an adapter, not an equivalent implementation of official pixel resizing.

All commands below run from the existing project root. Multiprocessing IPC and
the HOME/tmp directory require an environment with normal local process access.

## Recommended N execution: native block-major whole-shard

The earlier `compact_v3_tiled_z32_dctnet_static64` version is actually
image-major-vector-rowgroups, and its per-image CPU bridge is a correctness
prototype, not the B6 execution baseline. It remains intact for historical
comparison. The corrected N version is `dct_major_dctnet_static64`:
spatial-major/image-minor, 1024 images per shard (last 848), with B6's
128-vector, block-group-aligned rowgroups and native access sidecars.

`evaluate_shards.py` uses the existing advanced binding and native planless
GPU decode. Each shard is requested exactly once, with one active and one
preparing shard. Its full 56×56 component grids are dequantized once on GPU;
the DCTNet channels are organized/normalized for the whole shard, then consumed
as batch views. Predictions are restored to the existing evaluation ordinal.
The native grid transform uses identity geometry, explicit 1:1 chroma sampling,
float32 output, no affine scaling, and full int16 clamp bounds. The block-major
writer checks all Q100 tables are 1, so these bounds cannot clip stored values.
No RGB-no-more resize, normalization, patch transform, or training profile runs.

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
$PYTHON galp/experiments/dct_pushdown_inference/build_storage.py
# Repeat generation to resume; completed shards are reused.
$PYTHON galp/experiments/dct_pushdown_inference/generate.py \
  --count 50000 --layout block-major --shard-images 1024 \
  --workers 16 --shard-workers 16 --encoding-threads 4 \
  --output-dir galp/data/system_rgbnomore/e2e_v3/dct_major_dctnet_static64
build/galp/tools/jpeg_dct/galp_block_major_access_tool \
  galp/data/system_rgbnomore/e2e_v3/dct_major_dctnet_static64/manifest.bin \
  --output-dir galp/data/system_rgbnomore/e2e_v3/dct_major_dctnet_static64/access \
  --output-json galp/data/system_rgbnomore/e2e_v3/dct_major_dctnet_static64/access.json
$PYTHON galp/experiments/dct_pushdown_inference/validate.py \
  galp/data/system_rgbnomore/e2e_v3/dct_major_dctnet_static64
$PYTHON galp/experiments/dct_pushdown_inference/evaluate_shards.py \
  --new-data galp/data/system_rgbnomore/e2e_v3/dct_major_dctnet_static64 \
  --count 50000 --batch-size 64 \
  --output-dir galp/data/system_rgbnomore/e2e_v3/runs/dctnet_static64/block_major_gpu
```

Use `--verify --count 8` with a separate output directory for exact sampled
component/input/logit checks; that invocation is not a performance measurement.
Sparse sample selection still activates whole containing shards, and reports
both evaluated and decoded image counts. Stage timings overlap; native read
bytes describe issued storage requests, not uncached device traffic. CUDA
allocator statistics exclude allocations owned by the native GALP runtime.

### Native identity optimization and fixed coefficient selection

The shared planless CUDA kernel now directly stores identity-geometry values
after the existing source lookup/dequantization/clamp, bypassing separable
transform scratch traffic and intervening barriers. Dispatch depends only on
unit scaling factors, not model names, channel counts or normalization constants.
The existing sparse zero-fill, flip and partial-workset accumulation semantics
remain intact. Non-identity geometry follows the existing path.

`evaluate_shards.py` now defaults to `--pushdown on`. It reuses the reader's
`list:` coefficient selection and the existing zigzag mapping. The union of this
checkpoint's required natural frequencies is 44 columns, shared across all
three components. `--pushdown off` requests all 64 columns per component.
The helper `native_options()` without arguments still returns the full-grid
contract used by the storage validator. Model input and normalization are
unchanged; `--verify` checks zero-filled omitted frequencies and exact final
input/logit equality with the reference.

For this block-major dataset, `sparse_read_supported` is false: coefficient
selection reduces GPU workset upload and selects decode expressions, but storage
still reads complete rowgroups. This is not physical-I/O pushdown. The legacy
`decoded_coefficient_bytes` counter assumes 64 coefficients per vector and does
not quantify selected-column decode savings. Full output grids and the final
round/gather/concatenate/normalize passes remain; no new projected-output API or
scheduling policy was introduced.

Actual results are in `runs/dctnet_static64/native_optimization/REPORT.md` under
the existing e2e_v3 data root. The earlier `h100_off`/`h100_on` files actually identify RTX PRO 6000 Blackwell
(`CUDA_VISIBLE_DEVICES=1` did not match the nvidia-smi index). They are not H100
measurements. For H100 use its verified UUID; corrected full results are in
`runs/dctnet_static64/h100_uuid_retest/`. Commands:

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
export CUDA_VISIBLE_DEVICES=GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/home/tangyuxin/tmp/dctnet MPLCONFIGDIR=/home/tangyuxin/tmp/matplotlib
PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
for mode in off on; do
  $PYTHON galp/experiments/dct_pushdown_inference/evaluate_shards.py \
    --new-data galp/data/system_rgbnomore/e2e_v3/dct_major_dctnet_static64 \
    --count 50000 --batch-size 64 --pushdown "$mode" \
    --output-dir "galp/data/system_rgbnomore/e2e_v3/runs/dctnet_static64/h100_uuid_retest/native_$mode"
done
```

## Historical compact CPU-bridge commands

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/home/tangyuxin/tmp/matplotlib
export TMPDIR=/home/tangyuxin/tmp/dctnet
PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
$PYTHON galp/experiments/dct_pushdown_inference/build_storage.py

# Full validation compression; repeat this exact command to resume completed shards.
$PYTHON galp/experiments/dct_pushdown_inference/generate.py \
  --count 50000 --workers 16 --shard-workers 32 --encoding-threads 1 \
  --shard-images 512 \
  --output-dir galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32_dctnet_static64

$PYTHON galp/experiments/dct_pushdown_inference/validate.py \
  galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32_dctnet_static64

$PYTHON galp/experiments/dct_pushdown_inference/evaluate.py --route R --count 1000 --batch-size 16 --workers 8
$PYTHON galp/experiments/dct_pushdown_inference/evaluate.py --route O --count 1000 --batch-size 16 --workers 8
$PYTHON galp/experiments/dct_pushdown_inference/evaluate.py --route N --count 1000 --batch-size 16 --workers 8 \
  --new-data galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32_dctnet_static64 \
  --output-dir galp/data/system_rgbnomore/e2e_v3/runs/dctnet_static64/full
```

For the actual first 1000-image dataset command, use generation `--count 1000
--shard-images 32` and output suffix `_1000`, with the same worker allocation.
The initial N measurement uses that dataset; the final N measurement uses full
validation shards. The commands above retain CPU float32 results, model threads
8; H2D is not applicable to those results. CUDA was unavailable in the sandbox,
but initialization outside it succeeded on RTX 4090. The actual GPU runs use:

```bash
for route in R N O; do
  $PYTHON galp/experiments/dct_pushdown_inference/evaluate.py \
    --route "$route" --device cuda --count 50000 --batch-size 64 --workers 64 \
    --new-data galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32_dctnet_static64 \
    --output-dir galp/data/system_rgbnomore/e2e_v3/runs/dctnet_static64/gpu || exit
done
```

The GPU 1000-image screening used `--count 1000 --workers 16`, otherwise the same
command. Both GPU stages use float32 with TF32 disabled, the same checkpoint and
batch size for all routes. H2D and model timings synchronize CUDA; E2E includes
the complete DataLoader path and metric collection. Source-file reads are real
online paths with normal OS page caching,
not cached model input tensors. `input_worker_seconds` is summed worker work and
must not be added to overlapping E2E wall time. Native reads and decode are
coupled; their combined timing is reported. Requested payload bytes are not
physical device I/O bytes. Pushdown on/off is not implemented in this wrapper.

The generator bounds each active shard to two four-image producer batches,
limits BLAS/OpenCV/PyTorch threads, and uses separate native processes to overlap
shards. A shard contains many images with the existing compact v3 independent
vector rowgroup layout (64 DCT columns), not individual image files. Final
manifests commit only contiguous completed shard prefixes. Codec preparation is
serial inside each encoder; shard concurrency is needed to parallelize it.
Intermediate standard FLS files reside in HOME/tmp and are removed on success.
No intermediate JPEG/NPY dataset is written. Process RSS statistics are per-process
high-water marks, not the summed peak of the whole pipeline.

Source paths are not serialized by existing GALP metadata. As with the existing
dataset, the sample/label sidecar provides identity; final validation checks its
mapping against the old manifest and verifies actual shard ranges and sampled
coefficients. Stored data and old experiment files remain unchanged.

To restore missing external resources, the actual acquisition commands used were:

```bash
git clone --depth 1 https://github.com/kaix-nv/DCTNet.git galp/experiments/dct_pushdown_inference/DCTNet
LIBRARY_PATH=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/lib \
  $PYTHON -m pip install opencv-python-headless jpeg2dct 'PyTurboJPEG<2' gdown matplotlib scikit-learn scikit-image
$PYTHON -m gdown --folder https://drive.google.com/drive/folders/1-A7XdSAYsfD_liZsK1hQv-UaaMNhA4Sm \
  -O galp/data/system_rgbnomore/e2e_v2/checkpoints/resnet50dct_upscaled_static_64
```

See the experiment run directory for actual results; a started process is not a
completed dataset. `generation.json` and `validation.json` are written on completion.

## RGB baseline and pipeline profiling

`DCTNET_PROFILE=mobilenet32` also selects MobileNetV2 for both RGB baselines.
The weight is DCTNet's specified `mobilenetv2_1.0-0c6065bc.pth`, retrieved from
`https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2_1.0-0c6065bc.pth`
and stored in `e2e_v2/checkpoints/mobilenetv2_rgb_official/`. The experiment uses
the official DCTNet base class with `upscale=True` to retain canonical RGB
strides, supplies its missing RGB forward, and strictly restores the entire
RGB state. At 224px, its output matched the original RGB implementation within
1.91e-6 absolute error on a real image. RGB and DCT checkpoints remain distinct.

The old-data adapter uses `backend.GRID`: for MobileNetV2, first perform the
existing Y64-to56 / chroma32-to28 crop and chroma-to56 DCT upsample, then apply
the same DCT-domain 2x upsample to all three 56x56 planes. This gives 112x112
planes without changing old GALP data, rounding, clamping, or re-quantizing.
This adapter is not the official pixel resize/re-encode reference.

The upstream `models/imagenet/resnet.py` references the official RGB checkpoint
`https://download.pytorch.org/models/resnet50-19c8e357.pth`. Its `ResNet` constructor
is modified for DCT: it has no RGB `forward` and uses layer2 stride=1. Therefore
`rgb.py` subclasses that constructor, supplies the canonical RGB forward, and
restores only the RGB layer2 stride=2. A real-image CPU forward matched the same
weights loaded into torchvision ResNet-50 exactly. The DCT network is unchanged.
This is an upstream-referenced RGB checkpoint with the canonical RGB graph;
the repository does not provide an independent runnable RGB evaluation entry.

RGB PyTorch uses a standard Dataset/DataLoader, PIL JPEG decode, bilinear
resize256/center224, ToTensor and ImageNet normalization. RGB DALI uses file reader,
mixed JPEG decoder, GPU resize and crop/normalize, with the same checkpoint and
class/sample mapping. DALI and PIL preprocessing are not numerically identical.
Both keep the last partial batch, use float32 with TF32 disabled, and include
reader initialization, input, forward and metric collection in E2E timing.

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/home/tangyuxin/tmp/dctnet MPLCONFIGDIR=/home/tangyuxin/tmp/matplotlib
PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
OUT=galp/data/system_rgbnomore/e2e_v3/runs/dctnet_static64/rgb_comparison
for route in pytorch dali; do
  $PYTHON galp/experiments/dct_pushdown_inference/evaluate_rgb.py \
    --route "$route" --count 50000 --batch-size 64 --workers 64 --output-dir "$OUT"
done
$PYTHON galp/experiments/dct_pushdown_inference/diagnose_models.py --output "$OUT/models.json"
$PYTHON galp/experiments/dct_pushdown_inference/diagnose_reference.py --output "$OUT/reference_cpu.json"
bash galp/experiments/dct_pushdown_inference/profile.sh
$PYTHON galp/experiments/dct_pushdown_inference/analyze_profiles.py "$OUT/steady/nsys"
```

The RGB checkpoint is cached at
`galp/data/system_rgbnomore/e2e_v2/checkpoints/resnet50_rgb_official/resnet50-19c8e357.pth`.
It is a trusted upstream legacy tar checkpoint, requiring `weights_only=False`;
all parameters are loaded strictly.

Profiling is separate from formal throughput. `--profile` processes 16384 warmup
images (beyond the 8192-image DataLoader prefetch capacity), captures the next
4096 images, and exits. Native N uses 16 warmup shards and four captured shards.
All traces contain 64 model forwards; native shard order differs from shuffled
reference order, while full 50K predictions use the same evaluation ordinals.
The earlier 1024-warmup traces remain under `rgb_comparison/nsys`, but sustained
pipeline conclusions use `rgb_comparison/steady/nsys`. Capture startup and
profiling perturb timing; use unprofiled 50K results for throughput.

`analyze_profiles.py` reuses existing interval helpers. CUDA runtime correlation
IDs link kernel launches to NVTX model/input scopes; DALI/native background
launches are classified as input. It reports interval unions, actual overlap,
copy bytes and kernel counts, not SM utilization or cold-disk physical reads.
CPU worker/stage sums overlap each other and GPU execution; do not add them to
wall time. `reference_cpu.json` is a separate 128-image single-worker diagnostic.

## Projected native output

`evaluate_shards.py` defaults to `--output-layout projected`. Use `--output-layout grid`
for the previous full-grid comparison, with the same `--pushdown on` source selection.
The native default remains unchanged: projection is requested explicitly by this wrapper.

The advanced `grid_transform.output_channels` option is a list of
`[component, natural_frequency, subtract, divide]` entries in output-channel order.
The native result is exposed as `batch.projected`, a contiguous FP32 NCHW tensor.
This uses the existing native allocation owner, producer event and consumer-stream lifetime.
It requires equal component output grids, uncached planless execution, unique component/
frequency pairs, finite normalization parameters and nonzero divisors. Geometry and source
coefficient selection remain separate: projecting a frequency-mixing transform does not
permit dropping its source dependencies. No checkpoint-specific constants occur in CUDA.

Only the selected channels are allocated. General transforms accumulate into this compact
buffer and finalize once with the original round/clamp/affine followed by subtract/divide.
Block-major identity geometry with a clamp containing zero has a single physical source
rowgroup per output block; it initializes normalized zero for absent locations and writes
final normalized coefficients directly from each resident source block. Identity threads
own output blocks and reuse source lookup across selected channels. The wrapper no longer
runs frequency gather, concatenation, subtraction or division on the projected tensor.

Full-grid accessors reject projected batches; `batch.projected` is the output interface.
The existing six-dimensional native descriptor carries trailing singleton dimensions;
PyTorch exposes NCHW through zero-copy squeeze views. The compressed format and the sample
order are unchanged. Both routes activate each shard once, then run microbatch views.

For block-major requests sharing one shard's identity geometry, the native planner
stores only one image's spatial ownership indices per workset. CUDA repeats the
slice using the logical image stride, preserving request order without expanding
or uploading a per-image index list. This applies to full-grid and projected
output independently of coefficient selection or checkpoint. Different crops,
scales or component geometry retain the general schedule. Existing optional
interval sidecars retain their expanded representation; no data format changes.
