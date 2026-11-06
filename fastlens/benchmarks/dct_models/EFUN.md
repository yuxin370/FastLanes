# eFUN inference and training

> Offline target-grid copies are used for the target-representation controls. For online source-DCT preparation, see the [rebuilding guide](../paper/REBUILDING_REFERENCES.md).

See the [published result tables](../paper/results/README.md) for the
configurations adopted by the final paper.

`efun.py` selects `efun_backend.py` and runs the existing CNN drivers. It reuses
sample selection, labels, FASTLENS encoding, native grid/projected reads, timing,
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
that exact function. There is no mean/std normalization. FASTLENS stores the
resulting raw integer model values and identity quantization tables; these
tables are **not** the quantization tables of the transcoded source JPEG.

eFUN keeps all 192 channels, so frequency pushdown on/off both request all
64 frequency columns. It provides a full-frequency control for data-path and
output-layout comparisons; it cannot establish gains from dropping DCT
frequencies or reducing model work. Changing the channel count requires a
different model/checkpoint. The old DCTNet `O` adapter is not an eFUN baseline.

## Setup and CPU checks

```bash
PYTHON=python3
EXP=fastlens/benchmarks/dct_models
DATA=fastlens/data/system_rgbnomore/e2e_v3
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export TMPDIR="$HOME/tmp/dctnet" MPLCONFIGDIR="$HOME/tmp/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
# Download the model source and checkpoint for a fresh checkout.
git clone https://github.com/kfirgoldberg/FUN.git "$EXP/FUN"
git -C "$EXP/FUN" checkout 6c2b5f4a43a2b514163ff1f3f114d4feeb174d3c
$PYTHON -m gdown 17GRexna2lMaqELTK-PYgIlwEUk9Hponp \
  -O fastlens/data/system_rgbnomore/e2e_v2/checkpoints/efun.pth
$PYTHON - <<'PY'
from pathlib import Path
import torch
from torchvision.models import EfficientNet_B0_Weights
w = EfficientNet_B0_Weights.IMAGENET1K_V1
root = Path('fastlens/data/system_rgbnomore/e2e_v2/checkpoints/efficientnet_b0_rgb_official')
root.mkdir(parents=True, exist_ok=True)
torch.hub.download_url_to_file(w.url, str(root / w.url.rsplit('/', 1)[-1]), hash_prefix='7f5810bc')
PY
$PYTHON -m unittest -v fastlens.benchmarks.dct_models.tests.test_efun
$PYTHON "$EXP/efun.py" generate --count 8 --workers 1 --encoding-threads 1 \
  --shard-images 4 --layout compact --output-dir "$DATA/dct_efun_smoke_cpu"
$PYTHON "$EXP/efun.py" validate "$DATA/dct_efun_smoke_cpu"
for route in R N; do
  $PYTHON "$EXP/efun.py" evaluate --route "$route" --device cpu --count 8 \
    --batch-size 2 --workers 0 --model-threads 2 --new-data "$DATA/dct_efun_smoke_cpu" \
    --output-dir "$DATA/runs/efun/cpu_smoke"
done
```

Use the existing `fastlanes-cuda` environment and compiled `dctnet_storage`
bridge documented in `README.md`. Python dependencies are torch, torchvision,
numpy, scipy, jpeg2dct, einops, PyYAML and gdown. The wrapper imports the
author's local timm fork, supplying its removed `torch._six.container_abcs`
alias without editing upstream code. `EFUN_ROOT` and `EFUN_CHECKPOINT` override
the checkout and checkpoint paths. Each invocation needs a separate process
from programs importing a different timm version.

## GPU inference through FASTLENS

Select an idle GPU after checking utilization and active processes. Obtain its
UUID with `nvidia-smi -L` and replace `GPU-SELECTED-UUID` below.

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=GPU-SELECTED-UUID
$PYTHON "$EXP/efun.py" generate --count 50000 --layout block-major \
  --shard-images 1024 --workers 8 --shard-workers 2 --encoding-threads 4 \
  --output-dir "fastlens/data/compressed/backup/offline_model_input/imagenet512_val_efun28_block_major"
build/fastlens/tools/jpeg_dct/fastlens_block_major_access_tool \
  "fastlens/data/compressed/backup/offline_model_input/imagenet512_val_efun28_block_major/manifest.bin" --output-dir "fastlens/data/compressed/backup/offline_model_input/imagenet512_val_efun28_block_major/geometry" \
  --output-json "fastlens/data/compressed/backup/offline_model_input/imagenet512_val_efun28_block_major/access.json"
$PYTHON "$EXP/efun.py" validate "fastlens/data/compressed/backup/offline_model_input/imagenet512_val_efun28_block_major"
OUT="$DATA/runs/efun/inference"
$PYTHON "$EXP/efun.py" evaluate --route R --device cuda --count 50000 \
  --batch-size 64 --workers 16 --output-dir "$OUT"
for variant in grid_off projected_off projected_on; do
  layout=${variant%_*}; mode=${variant##*_}
  $PYTHON "$EXP/efun.py" evaluate_shards --count 50000 --batch-size 64 \
    --new-data "fastlens/data/compressed/backup/offline_model_input/imagenet512_val_efun28_block_major" --output-layout "$layout" --pushdown "$mode" \
    --baseline-dir "$OUT" --output-dir "$OUT/$variant"
done
```

`evaluate_shards --verify` enables additional input/logit comparisons and must
be run separately from performance measurements. Generation preserves the
existing validation population and supports resuming completed shards.
The workspace's source is ImageNet-512; reproducing the paper's reported
accuracy requires the original ImageNet dataset, not treating these reencoded
images as the original validation inputs.

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
EFUN_RGB_TEST_CUDA=1 $PYTHON -m unittest -v fastlens.benchmarks.dct_models.tests.test_efun
RGB_OUT="$DATA/runs/efun/rgb_efficientnet_b0_reproduction"
for route in pytorch dali; do
  workers=16; depth=2
  if [[ "$route" == dali ]]; then workers=4; depth=4; fi
  $PYTHON "$EXP/efun.py" evaluate_rgb --route "$route" --count 50000 \
    --batch-size 64 --workers "$workers" --dali-prefetch-depth "$depth" --output-dir "$RGB_OUT"
done
```

Both RGB routes use the same checkpoint, samples and labels. Their comparison
isolates the input pipeline. Comparing either with eFUN/FASTLENS also changes the
model architecture and input representation, so report both accuracy and
throughput rather than attributing the entire difference to storage pushdown.
This entry point evaluates pretrained RGB inference. `efun.py training_pls`
also supports RGB EfficientNet-B0 with `--condition RGB --input-backend
rgb_pytorch` or `rgb_d2`, using the existing system training/validation recipe
(bilinear, mean/std .5) and scratch initialization, separately from these
pretrained inference transforms. `--initial-validation` records the scratch
baseline before training; the configured full matrix enables it. The eFUN DALI
training entry point uses 16 workers and `--dali-prefetch-depth 4`.

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

**FASTLENS system comparison.** This reuses the current CNN `training_pls.py`
recipe: scratch initialization, BF16, compile, microbatch 64, accumulation 16,
shared optimizer/schedule, DCT crop/resize/flip, RandAugment and Mixup. It uses
the existing full-frequency training storage and constructs 28×28 grids online;
no new expanded training dataset is needed. JPEG A0 and native A0/B6 share the
DCT-domain contract. It is not a reproduction of the author's training recipe.
Validation uses the eFUN target storage generated above.

```bash
PHYSICAL=./fastlens/data/compressed/imagenet512_train_block_major_premixed
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
    --epochs 2 --validation-data "fastlens/data/compressed/backup/offline_model_input/imagenet512_val_efun28_block_major" \
    --output-dir "$DATA/runs/efun/training/${backend}_${condition}"
done
$PYTHON "$EXP/efun.py" training_model_only \
  --output-dir "$DATA/runs/efun/model_only"
```

Begin with `--max-pools 2 --no-compile` for a bounded PLS diagnostic; it does
not establish convergence. Full runs save epoch checkpoints and accept
`--resume`. Model-only calibration measures resident-input updates and excludes
input processing. Use `efun.py TASK --help` for each existing driver's options.
