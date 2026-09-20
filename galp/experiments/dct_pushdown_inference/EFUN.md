# eFUN inference and training

See [家族模型、完整结果与下一阶段训练配置](EFUN_RESULTS_AND_TRAINING_ZH.md)
for the consolidated model/result tables and the four-arm training matrix.

`efun.py` selects `efun_backend.py` and runs the existing CNN drivers. It reuses
sample selection, labels, GALP encoding, native grid/projected reads, timing,
PLS A0/B6, checkpointing, and model-only calibration. No eFUN-specific CUDA
kernel is introduced. Run commands below from the repository root.

## Model and input contract

The upstream source is [FUN](https://github.com/kfirgoldberg/FUN), revision
`6c2b5f4a43a2b514163ff1f3f114d4feeb174d3c`. The base model has 4,233,448
parameters and input `[N, 192, 28, 28]`: all 64 natural-order frequencies of
Y, Cb and Cr. The published checkpoint is loaded strictly from `state_dict`,
matching upstream validation's default without `--use-ema`.

Inference uses the author's bicubic Resize(256), CenterCrop(224), ToTensor,
ToPILImage, Q100 JPEG with subsampling=0, then `jpeg2dct(normalized=False)`.
The installed jpeg2dct transcodes this JPEG to 4:2:0; the author's
`DCT._upsample_and_concat` expands chroma back to 28×28. The adapter invokes
that exact function. There is no mean/std normalization. GALP stores the
resulting raw integer model values and identity quantization tables; these
tables are **not** the quantization tables of the transcoded source JPEG.

eFUN keeps all 192 channels, so frequency pushdown on/off both request all
64 frequency columns. It provides a full-frequency control for data-path and
output-layout comparisons; it cannot establish gains from dropping DCT
frequencies or reducing model work. Changing the channel count requires a
different model/checkpoint. The old DCTNet `O` adapter is not an eFUN baseline.

## Setup and CPU checks

```bash
PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
EXP=galp/experiments/dct_pushdown_inference
DATA=galp/data/system_rgbnomore/e2e_v3
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export TMPDIR="$HOME/tmp/dctnet" MPLCONFIGDIR="$HOME/tmp/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
# Already downloaded in this workspace. For a fresh checkout:
git clone https://github.com/kfirgoldberg/FUN.git "$EXP/FUN"
git -C "$EXP/FUN" checkout 6c2b5f4a43a2b514163ff1f3f114d4feeb174d3c
$PYTHON -m gdown 17GRexna2lMaqELTK-PYgIlwEUk9Hponp \
  -O galp/data/system_rgbnomore/e2e_v2/checkpoints/efun.pth
$PYTHON - <<'PY'
from pathlib import Path
import torch
from torchvision.models import EfficientNet_B0_Weights
w = EfficientNet_B0_Weights.IMAGENET1K_V1
root = Path('galp/data/system_rgbnomore/e2e_v2/checkpoints/efficientnet_b0_rgb_official')
root.mkdir(parents=True, exist_ok=True)
torch.hub.download_url_to_file(w.url, str(root / w.url.rsplit('/', 1)[-1]), hash_prefix='7f5810bc')
PY
$PYTHON "$EXP/test_efun.py" -v
$PYTHON "$EXP/efun.py" generate --count 8 --workers 1 --encoding-threads 1 \
  --shard-images 4 --layout compact --output-dir "$DATA/dct_efun_smoke_cpu"
$PYTHON "$EXP/efun.py" validate "$DATA/dct_efun_smoke_cpu"
for route in R N; do
  $PYTHON "$EXP/efun.py" evaluate --route "$route" --device cpu --count 8 \
    --batch-size 2 --workers 0 --model-threads 2 --new-data "$DATA/dct_efun_smoke_cpu" \
    --output-dir "$DATA/runs/efun/cpu_smoke_20260915"
done
```

Use the existing `fastlanes-cuda` environment and compiled `dctnet_storage`
bridge documented in `README.md`. Python dependencies are torch, torchvision,
numpy, scipy, jpeg2dct, einops, PyYAML and gdown. The wrapper imports the
author's local timm fork, supplying its removed `torch._six.container_abcs`
alias without editing upstream code. `EFUN_ROOT` and `EFUN_CHECKPOINT` override
the checkout and checkpoint paths. Each invocation needs a separate process
from programs importing a different timm version.

## GPU inference through GALP

Check utilization **and** active processes before selecting a device. Prefer
an idle PRO 6000, then an idle H100. Do not use a busy RTX 4090 or stop another
workload. The commands below assume the PRO 6000 has been confirmed idle;
replace the UUID with the H100 UUID if that is the idle device.

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=GPU-e796262d-3449-6af1-586d-8460d8836d1b
# H100: GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76
$PYTHON "$EXP/efun.py" generate --count 50000 --layout block-major \
  --shard-images 1024 --workers 8 --shard-workers 2 --encoding-threads 4 \
  --output-dir "$DATA/dct_major_efun"
build/galp/tools/jpeg_dct/galp_block_major_access_tool \
  "$DATA/dct_major_efun/manifest.bin" --output-dir "$DATA/dct_major_efun/access" \
  --output-json "$DATA/dct_major_efun/access.json"
$PYTHON "$EXP/efun.py" validate "$DATA/dct_major_efun"
OUT="$DATA/runs/efun/pro6000/full"
$PYTHON "$EXP/efun.py" evaluate --route R --device cuda --count 50000 \
  --batch-size 64 --workers 16 --output-dir "$OUT"
for variant in grid_off projected_off projected_on; do
  layout=${variant%_*}; mode=${variant##*_}
  $PYTHON "$EXP/efun.py" evaluate_shards --count 50000 --batch-size 64 \
    --new-data "$DATA/dct_major_efun" --output-layout "$layout" --pushdown "$mode" \
    --baseline-dir "$OUT" --output-dir "$OUT/$variant"
done
```

`evaluate_shards --verify` enables additional input/logit comparisons and must
be run separately from performance measurements. Generation preserves the
existing validation population and supports resuming completed shards.
The workspace's source is ImageNet-512; reproducing the paper's reported
accuracy requires the original ImageNet dataset, not treating these reencoded
images as the original validation inputs.

## Nsight Systems captures

The complete inference/training Nsight Systems matrix can be collected with
`efun_profile.sh` after selecting an idle GPU:

```bash
export CUDA_VISIBLE_DEVICES=GPU-40c637bd-acf5-ea1a-0df8-617138228467
bash galp/experiments/dct_pushdown_inference/efun_profile.sh nsys_20260915
```

The script checks that the selected GPU has no compute processes and records
its processes every 100 ms. It captures six inference paths (JPEG reference,
GALP grid off, projected off/on, RGB PyTorch/DALI), each over 4,096 images
after 16,384 warmup images. Four training paths (JPEG, GALP B6, RGB PyTorch,
RGB DALI D2) resume the completed epoch-1 checkpoints, warm up one pool, and
capture four pools / 16,384 images in epoch 2. Thus training captures exclude
the first compile and use the audit policy after update 100. They do not
repeat full training or replace unprofiled throughput measurements.

Reports are written under `e2e_v3/runs/efun/nsys_20260915/`, with per-path
`.nsys-rep`, SQLite, breakdown JSON and charts; training also exports PNG/PDF
timelines. Capture logs must report `profile_complete: true`. Training uses
the JPEG checkpoint from `training_v1_jpeg_rerun_20260915` and the other three
checkpoints from `training_v1`. Inference uses the official pretrained weights.

## RGB EfficientNet-B0 with PyTorch and DALI

`efun.py evaluate_rgb` reuses `evaluate_rgb.py` and `rgb.py` with Torchvision's
`EfficientNet_B0_Weights.IMAGENET1K_V1`. The 5,288,548-parameter RGB model is
strictly restored from `efficientnet_b0_rwightman-7f5810bc.pth`; override its
path with `EFUN_RGB_CHECKPOINT`. PyTorch uses the official weight transform:
bicubic Resize(256), CenterCrop(224), RGB `[0,1]`, ImageNet mean/std. DALI uses
mixed JPEG decoding, GPU antialiased cubic resizing, crop and normalization.
Its decoder/resampler is not assumed bit-identical to PIL; results include
prediction agreement and accuracy difference against the same RGB checkpoint.
The existing ResNet/MobileNet RGB profiles retain their bilinear preprocessing.

With an idle GPU selected as above:

```bash
EFUN_RGB_TEST_CUDA=1 $PYTHON "$EXP/test_efun.py" -v
RGB_OUT="$DATA/runs/efun/rgb_efficientnet_b0_reproduction"
for route in pytorch dali; do
  workers=16; depth=2
  if [[ "$route" == dali ]]; then workers=4; depth=4; fi
  $PYTHON "$EXP/efun.py" evaluate_rgb --route "$route" --count 50000 \
    --batch-size 64 --workers "$workers" --dali-prefetch-depth "$depth" --output-dir "$RGB_OUT"
done
```

Both RGB routes use the same checkpoint, samples and labels. Their comparison
isolates the input pipeline. Comparing either with eFUN/GALP also changes the
model architecture and input representation, so report both accuracy and
throughput rather than attributing the entire difference to storage pushdown.
This entry point evaluates pretrained RGB inference. `efun.py training_pls`
also supports RGB EfficientNet-B0 with `--condition RGB --input-backend
rgb_pytorch` or `rgb_d2`, using the existing system training/validation recipe
(bilinear, mean/std .5) and scratch initialization, separately from these
pretrained inference transforms. `--initial-validation` records the scratch
baseline before training; the configured full matrix enables it. The eFUN DALI
training entrypoint now uses 16 workers and `--dali-prefetch-depth 4`, selected
from repeated training probes. See [the results report](EFUN_RESULTS_AND_TRAINING_ZH.md#6-dali-配置调优2026-09-16)
for the completed full-training results and their concurrent-GPU-process flags.
The selected configuration completed two full epochs in 949.099 / 938.800 s
with unchanged training/validation metrics, but brief extra GPU processes
were observed in both runs; these timings are not uncontended measurements.

The RTX 4090 results use 50K images, FP32, TF32 disabled and batch 64.
PyTorch uses the original 16-worker run. DALI uses 4 threads and prefetch depth 4,
with the median of three interleaved confirmation runs on 2026-09-16 under
`dali_tuning_20260916/inference_confirmation/`:

| RGB input path | Online seconds | Images/s | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| PyTorch/PIL | 22.338 | 2,238.30 | 76.774% | 93.238% |
| DALI (4 threads / depth 4) | 11.733 | 4,261.62 | 76.770% | 93.262% |

DALI is 1.90× faster than the recorded PyTorch result. Interleaved confirmation
medians were 11.929 s for the old 16-thread/depth-2 configuration and 11.733 s
for 4-thread/depth-4, only a 1.64% elapsed-time reduction. All six confirmation
runs have identical DALI predictions. Prediction agreement with PyTorch was
99.244%; the Top-1 difference was -0.004 percentage points. Results preserve
both prediction arrays and the preprocessing contract. Six integration tests
passed, including strict RGB checkpoint loading, official PyTorch input
equality and DALI's 17-image `8+8+1` tail-batch/order check; existing ResNet and
MobileNet RGB forward checks also passed.

For context, the earlier eFUN/GALP projected run on the same GPU and sample
population took 10.730 seconds with Top-1 75.428%. Against this stronger RGB
DALI baseline, its measured speed advantage is about 1.09× with 1.342
percentage points lower Top-1. This is an architecture/data-path tradeoff,
not a same-model pushdown ablation. Neither measurement establishes a
statistical performance bound or forces a cold filesystem cache.

## Training: two distinct experiments

**Author recipe.** This delegates to unchanged upstream `train.py`, with the
README defaults: 450 epochs, batch 128, RMSpropTF, learning rate .048, step
decay .97 every 2.4 epochs, dropout .2, drop-path .2 and EMA .9999. Additional
arguments override these defaults. Use an ImageNet root containing `train/`
and `val/`; no pretrained checkpoint is required to train from scratch.

```bash
$PYTHON "$EXP/efun.py" official_train /path/to/original/imagenet \
  --workers 16 --output "$DATA/runs/efun/official_training"
# Resume with the upstream --resume /path/to/checkpoint.pth.tar option.
```

The author performs random crops/flips/color jitter in the pixel domain before
DCT extraction. Its DCT transform branch does not apply RandomErasing even
though the README passes `--reprob .2`. The wrapper preserves that behavior.

**GALP system comparison.** This reuses the current CNN `training_pls.py`
recipe: scratch initialization, BF16, compile, microbatch 64, accumulation 16,
shared optimizer/schedule, DCT crop/resize/flip, RandAugment and Mixup. It uses
the existing full-frequency training storage and constructs 28×28 grids online;
no new expanded training dataset is needed. JPEG A0 and native A0/B6 share the
DCT-domain contract. It is not a reproduction of the author's training recipe.
Validation uses the eFUN target storage generated above.

```bash
PHYSICAL=/mnt/nvme2/home/tangyuxin/pls-experiments/physical-layout-full-premix-orgseed-20260810/uniform_premix
mapfile -t INPUTS < <("$PYTHON" - "$PHYSICAL/materialization_contract.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); c = json.loads(p.read_text())
print(p.parent / 'dct/manifest.bin')
print(c['ordered_mapping'])
print(c['ordered_mapping_sha256'])
PY
)
$PYTHON "$EXP/efun.py" training_input_probe --manifest "${INPUTS[0]}" \
  --mapping "${INPUTS[1]}" --output-dir "$DATA/runs/efun/input_probe"
for spec in jpeg:A0 native:A0 native:B6; do
  backend=${spec%:*}; condition=${spec#*:}
  $PYTHON "$EXP/efun.py" training_pls --manifest "${INPUTS[0]}" \
    --mapping "${INPUTS[1]}" --mapping-sha256 "${INPUTS[2]}" \
    --condition "$condition" --input-backend "$backend" --segments-per-pool 4 \
    --epochs 2 --validation-data "$DATA/dct_major_efun" \
    --output-dir "$DATA/runs/efun/training/${backend}_${condition}"
done
$PYTHON "$EXP/efun.py" training_model_only \
  --output-dir "$DATA/runs/efun/model_only"
```

Begin with `--max-pools 2 --no-compile` for a bounded PLS diagnostic; it does
not establish convergence. Full runs save epoch checkpoints and accept
`--resume`. Model-only calibration measures resident-input updates and excludes
input processing. Use `efun.py TASK --help` for each existing driver's options.

## Validation performed

On 2026-09-15, all four CPU integration tests passed: exact author input,
strict checkpoint inference and a finite RMSpropTF training update, unchanged
existing CNN upsampling, and constant-signal preservation during 28-grid
downsampling. Eight real validation images were encoded into two GALP shards
and evaluated through R and N: inputs matched exactly, predictions matched
100%, and sampled logits differed by at most 1.44e-6. These are correctness
checks, not an accuracy or speedup experiment.

After the user confirmed the RTX 4090 was idle, GPU checks and measurements
completed under `e2e_v3/runs/efun/rtx4090_20260915/`. PRO 6000 and H100 were
busy and were not used. GPU block-major roundtrips and the separate projected
verification passed. The training input probe passed crop/resize/flip checks
and a backward/update; grid and projected outputs matched exactly. The
maximum difference from the unrounded reference was 0.50005, consistent with
the existing integer-rounding contract.

The complete inference comparison used 50,000 ImageNet-512 validation images,
FP32, TF32 disabled and batch 64. The reference used 16 CPU workers. All four
runs achieved Top-1 75.428%, Top-5 92.622%, and all GALP predictions matched R.

| Online inference path | Seconds | Speedup over R |
|---|---:|---:|
| R: author JPEG preprocessing | 41.873 | 1.00× |
| GALP grid, pushdown off | 11.800 | 3.55× |
| GALP projected, pushdown off | 10.730 | 3.90× |
| GALP projected, pushdown on | 10.738 | 3.90× |

These are single-run online timings after offline target preparation; cache
state was not forced cold. Generation took 533.47 seconds and produced about
5.35 GB before access sidecars. All GALP variants requested the same
5,274,114,488 compressed payload bytes and retained every frequency. Therefore
this comparison supports faster online execution through the GALP data path,
not a benefit from frequency selection: projected off/on are effectively
equal. The model, input dimensions and arithmetic workload are unchanged;
different measured forward times do not establish reduced model work.
Raw results are in `full/R_50000.json` and
`full/{grid_off,projected_off,projected_on}/N_50000.json` beneath the run root.

Model-only calibration completed 5 warmup plus 120 measured updates using
BF16/compile and accumulation 16: 122,880 image instances in 42.609 seconds
(2,883.87 images/s), with parameter updates verified. The calibration reuses
64 resident training images; it does not measure data loading.

The uncompiled BF16 training diagnostics completed two pools each for native
B6 and JPEG A0: 8,192 unique images, 8 optimizer updates, finite checks and
8-image validation passed. Their training windows were 7.984 and 13.630
seconds respectively. They ran alongside CPU target generation, use different
ordering/crop policies, and are correctness diagnostics rather than a matched
training performance or convergence study.

Native A0 was interrupted after its first pool finally completed: 4,096
images and 4 updates in 423.951 seconds. Its two-pool diagnostic did not finish,
and no successful `training.json` was emitted. The preserved
`training_smoke/native_A0/run.log` includes the first-pool record and corrected
interruption status. Full-epoch convergence and the author's 450-epoch
training reproduction have not been run.
