# L3 RGB baseline

This is a repaired and ported baseline from
[SNU-ARC/L3](https://github.com/SNU-ARC/L3/tree/3b8e52c226ac8749025e14858d5c5f260b562085),
commit `3b8e52c226ac8749025e14858d5c5f260b562085` (MIT license in `upstream/LICENSE`).
Cite Bae et al., *L3: Accelerator-Friendly Lossless Image Format for
High-Resolution, High-Throughput DNN Training*, ECCV 2022.

## Scope and repairs

The original Paeth/base-delta/bit-packing algorithm is retained. For the fixed
512x512 inputs, patches are **32x32**, as specified for sub-HD images in
[the L3 paper, §4.3](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710171.pdf).
The decoder only replaces its hard-coded 64-pixel row length with the patch size;
its reconstruction algorithm and patch/pixel parallelism are unchanged.
The encoder repairs are bounded to observed correctness defects:

- Compute bit width with integer arithmetic, including constant rows.
- Emit both header bytes for zero-bit rows and implement the missing eight-bit case.
- Repair two six-bit masks and one shift-precedence error.

The standalone prototype's file I/O and old DALI 1.1 patch are replaced by a small
current-DALI mixed operator. It computes 256 patch prefixes per channel and uses
DALI-managed buffers. The author patch's 32 nonblocking decode streams are retained,
with sample i assigned to stream i modulo 32 so batches larger than 32 are valid.
Events order the input copies before decoding and join the decoder streams before
resize/normalization or buffer reuse. Decode streams have the lowest priority;
RGB model execution has the highest priority, following paper §4.4.
The operator decodes directly into the output GPU tensor. This avoids the old
patch's prefix-array overrun, fixed 32-stream indexing, erroneous device-to-host output copy, leaked offset buffers,
and per-image device synchronization. No new decoder algorithm, selective decode,
compression optimization, or model change is introduced. Report it as
**L3 (repaired codec, ported to DALI 2.x)**, not the unmodified author artifact.

This baseline accepts locally converted **512x512 RGB** files. It supports the
paper's ViT-Ti, SwinV2-T, MobileNetV2, ResNet-50 and EfficientNet-B0 RGB references,
with their existing 224/256 inference transforms and planned training crop/flip.
It does not provide the seven matched-DCT workloads or arbitrary image sizes.
EfficientNet-B0 remains an RGB deployment reference for eFUN, not the same model.

## Build and verify

Run from the repository root, in the existing `fastlanes-cuda` environment:

```bash
export TMPDIR="$HOME/tmp"
python -m galp.benchmarks.l3.build \
  --nvcc "$HOME/tmp/coordl-cuda-12.8/bin/nvcc" \
  --output "$HOME/tmp/l3-baseline/build/libl3-patch32.so"
L3_LIBRARY="$HOME/tmp/l3-baseline/build/libl3-patch32.so" \
  python -m unittest galp.benchmarks.l3.test_codec -v
```

The GPU tests cover 0–8-bit data, constant pixels, batches of 64 and partial
batches. The conversion command additionally checks every image against Pillow's
RGB decode of the source JPEG. L3 preserves those pixels exactly; it does not
recover pre-JPEG originals. Native JPEG decoders can differ slightly in pixels,
so the inference comparison reports measured accuracy rather than assuming equal
predictions between JPEG/DALI and L3/DALI.

```bash
python -m galp.benchmarks.l3.prepare --split val \
  --library "$HOME/tmp/l3-baseline/build/libl3-patch32.so" --output "$HOME/tmp/l3-baseline/data-patch32"
python -m galp.benchmarks.l3.prepare --split train \
  --library "$HOME/tmp/l3-baseline/build/libl3-patch32.so" --output "$HOME/tmp/l3-baseline/data-patch32"
```

Validation contains all 50,000 images. Training contains the existing seeded
98,560-image selection: 32,768 excluded warmup images, 65,536 measured images,
and the reader's extra tail. The conversion JSON records actual encoded bytes
and total preparation time **including exhaustive round-trip validation**.

For the storage table, each 512x512 image has 1,549 metadata bytes: a 13-byte
file header and 768 two-byte patch lengths. Payload is the remaining encoded
stream, including row control bits and row padding. There is no separate
persistent access index; shared JPEG/L3 benchmark manifests are excluded.

To measure all training images without retaining another full encoded dataset,
run the same encoder and exhaustive pixel verification, accumulating the exact
serialized lengths in memory:

```bash
python -m galp.benchmarks.l3.storage \
  --library "$HOME/tmp/l3-baseline/build/libl3-patch32.so" \
  --output results/l3_patch32_20261006/storage_train_full.json
```

This measures encoded file bytes, not filesystem allocation or conversion
throughput. The output is written only after the full manifest is processed.

The [full training measurement](../../../results/l3_patch32_20261006/storage_train_full.json)
completed on 2026-10-07: all 1,281,167 images passed pixel comparison.
The encoded total is 737,018,660,283 bytes: 735,034,132,600 payload bytes and
1,984,527,683 metadata bytes, with no separate persistent index. The corresponding
JPEG files total 50,600,614,992 bytes, giving 14.5654x storage amplification.
Encoding and exhaustive verification took 16,136.3 seconds; encoded files were
not retained. The training-window performance runs still use their original
98,560-image converted subset.

## Paper-scale comparison

`run.py` reuses the accepted DALI commands referenced by
`results/p0/deployment_inference_summary.csv` and `training_window_summary.csv`.
These local experiment artifacts, `experiments/training_window_rgb.py`, the
original model repositories, checkpoints, and ImageNet-512 are required.
L3 training is exposed as `--pipeline l3 --l3-root ... --l3-library ...` in that
existing training runner. Model constructors, checkpoint selection, batch size,
precision, augmentations, warmup and measurement boundaries stay with each
workload's established recipe.

```bash
python -m galp.benchmarks.l3.run --gpu H100 --phase inference \
  --library "$HOME/tmp/l3-baseline/build/libl3-patch32.so" \
  --data "$HOME/tmp/l3-baseline/data-patch32" --output results/l3_patch32_20261006
python -m galp.benchmarks.l3.run --gpu H100 --phase training \
  --library "$HOME/tmp/l3-baseline/build/libl3-patch32.so" \
  --data "$HOME/tmp/l3-baseline/data-patch32" --output results/l3_patch32_20261006
```

Repeat for `--gpu 4090`. Run measurements serially, after conversion finishes.
Each model runs three L3 and three JPEG/DALI trials in alternating order.
Inference consumes all 50,000 images; each training trial measures 64 updates
with microbatch 64 and accumulation 16. L3 training first executes the existing
input/finite-loss/parameter-update correctness check. The existing resource guard
and warm-cache protocol apply. Outputs retain the original raw results and a
`summary.csv`; no manuscript numbers are updated automatically.

## Verification of the corrected configuration (2026-10-06)

The GPU codec and training regressions passed 13 tests; the existing training and
inference adapter suites passed 86 tests. All 50,000 validation images were
converted with 32x32 patches and checked pixel-for-pixel against the Pillow RGB
source. They occupy 29,250,077,944 bytes; preparation including exhaustive
validation took 603.3 seconds.
The 98,560-image training selection also passed exhaustive pixel comparison;
it occupies 56,706,085,476 bytes and took 1,174.8 seconds including validation.

All five RGB models passed 100-image inference smoke checks on the RTX PRO 6000
and the existing training correctness checks (RGB input comparison, finite loss,
and parameter updates) on the RTX PRO 6000, RTX 4090, and H100. These checks are
separate from performance trials.

A separate 4,096-image Nsight Systems diagnostic on the RTX PRO 6000 recorded
32 active L3 decoder streams, a peak of 32 concurrent decoder kernels, and
57.27 ms of overlap between L3 decoding and GPU kernels inside the model.forward
NVTX ranges. Each image uses a 16x16 grid with 32 threads per block. This verifies
scheduling behavior, not 4090/H100 throughput. The
[trace and diagnostic summary](../../../results/l3_patch32_20261006/profile/PRO6000_mobilenet24)
are retained separately from performance trials.

The initial 4090 profiling attempt was stopped by the existing contention guard
when another paper experiment acquired the GPU. No performance result from that
attempt is used. All 30 corrected 4090 inference trials completed (50,000 images
per trial, three trials per model and pipeline). Medians in images/s:

| Model | JPEG/DALI | L3/DALI | L3 / JPEG-DALI |
|---|---:|---:|---:|
| ViT-Ti | 4,802 | 4,496 | 0.936× |
| SwinV2-T | 1,059 | 1,041 | 0.983× |
| MobileNetV2 | 5,256 | 5,248 | 0.998× |
| ResNet-50 | 2,046 | 2,025 | 0.990× |
| EfficientNet-B0 (eFUN RGB reference) | 4,234 | 4,137 | 0.977× |

L3 Top-1 is unchanged from the initial port for all five models and is identical
across repeats. The corrected L3 inference throughput is 1.046–1.928× that of
the initial port. These measurements restore the paper's configuration and
the author's stream arrangement; they do not isolate the contributions of patch
size and scheduling. The [summary and raw results](../../../results/l3_patch32_20261006/summary.csv)
are separate from the historical results below.

All 30 corrected 4090 training trials also completed: each consumed 65,536
measured images and performed 64 updates after 32 excluded warmup updates.
Every trial passed the existing correctness checks and matched the original
sample order and initial model hash. Medians in images/s, with minimum–maximum
across the three trials in brackets:

| Model | JPEG/DALI | L3/DALI | L3 / JPEG-DALI |
|---|---:|---:|---:|
| ViT-Ti | 2,214 [2,031–2,230] | 2,194 [2,175–2,209] | 0.991× |
| SwinV2-T | 1,389 [1,387–1,390] | 1,371 [1,362–1,373] | 0.987× |
| MobileNetV2 | 2,231 [1,982–2,247] | 2,084 [2,063–2,211] | 0.934× |
| ResNet-50 | 1,731 [1,729–1,732] | 1,711 [1,705–1,716] | 0.989× |
| EfficientNet-B0 (eFUN RGB reference) | 1,848 [1,737–1,910] | 1,754 [1,741–1,765] | 0.949× |

The host is shared. ViT-Ti, MobileNetV2 and EfficientNet-B0 training ranges overlap
between pipelines, and several JPEG/DALI trials vary appreciably. Small median
differences must not be presented as a demonstrated stable advantage. These are
training-window throughput measurements, not convergence results.

### H100 measurements (2026-10-06)

All 60 H100 trials completed using the same corrected plugin and converted data:
30 full-validation inference trials and 30 training-window trials. The existing
sample-order, checkpoint/initial-model, finite-loss and parameter-update checks
passed. Together with the 4090 results, the summary contains 120 accepted trials.

An initial SwinV2-T JPEG/DALI attempt was stopped when a separate spatial
experiment acquired H100. Its [contention record](../../../results/l3_patch32_20261006/interrupted/H100/swinv2_dali_trial_0_contention)
is retained separately and excluded from the summary. Measurement resumed after
that entire experiment exited. No accepted H100 trial recorded target-GPU
contention; the host remains shared.

Inference medians in images/s, with the minimum–maximum of three trials:

| Model | JPEG/DALI | L3/DALI | L3 / JPEG-DALI |
|---|---:|---:|---:|
| ViT-Ti | 5,249 [5,035–5,279] | 4,893 [4,854–5,093] | 0.932× |
| SwinV2-T | 1,179 [1,174–1,181] | 1,168 [1,157–1,172] | 0.991× |
| MobileNetV2 | 6,170 [6,156–6,170] | 6,366 [6,356–6,379] | 1.032× |
| ResNet-50 | 2,210 [2,205–2,213] | 2,217 [2,213–2,219] | 1.003× |
| EfficientNet-B0 (eFUN RGB reference) | 4,891 [4,884–4,924] | 5,075 [5,074–5,091] | 1.038× |

Training medians in images/s, with the minimum–maximum of three trials:

| Model | JPEG/DALI | L3/DALI | L3 / JPEG-DALI |
|---|---:|---:|---:|
| ViT-Ti | 2,092 [2,087–2,190] | 2,135 [2,130–2,136] | 1.020× |
| SwinV2-T | 1,807 [1,796–1,837] | 1,827 [1,781–1,840] | 1.011× |
| MobileNetV2 | 2,117 [2,045–2,143] | 2,073 [2,034–2,155] | 0.979× |
| ResNet-50 | 2,084 [2,048–2,085] | 2,109 [2,060–2,147] | 1.012× |
| EfficientNet-B0 (eFUN RGB reference) | 1,778 [1,768–1,944] | 1,752 [1,744–1,754] | 0.985× |

Training medians differ by about 2% or less. The first four models have overlapping
trial ranges; the EfficientNet-B0 JPEG/DALI trials vary by about 10%. These shared-host
results do not establish a consistent training advantage for either pipeline.

H100 validation Top-1 is identical across all three repeats of each pipeline:

| Model | JPEG/DALI (%) | L3/DALI (%) | Difference (percentage points) |
|---|---:|---:|---:|
| ViT-Ti | 74.094 | 74.084 | -0.010 |
| SwinV2-T | 78.964 | 78.992 | +0.028 |
| MobileNetV2 | 70.782 | 70.718 | -0.064 |
| ResNet-50 | 74.850 | 74.740 | -0.110 |
| EfficientNet-B0 (eFUN RGB reference) | 76.776 | 76.750 | -0.026 |

No manuscript tables have been changed.

## Historical initial-port results (2026-10-06; superseded configuration)

**These results used 64x64 patches and one decode stream. They do not represent
the corrected paper configuration above and must not be inserted as final L3
numbers in the manuscript.** The old files and plugin remain available for
reproducing this historical run. New 32x32 measurements use a separate output
directory and require freshly converted data; the formats are not interchangeable.

All 60 formal trials completed: five models, two input pipelines, three trials each,
for both full-validation inference and training windows. H100 performance measurement
is deferred because another user still occupies that GPU; all five H100 L3 training
correctness checks passed. Software: PyTorch 2.11.0+cu128, DALI 2.2.0; plugin built
with CUDA 12.8. The measured plugin is `$HOME/tmp/l3-baseline/build/libl3-paper.so`.

Inference uses all 50,000 validation images with the original checkpoints and FP32
recipes. Training measures 65,536 images after 32,768 excluded warmup images, with
microbatch 64 and accumulation 16; ViT-Ti uses FP32 and the other models BF16 autocast.
Input pages are warm before timing. Rates below are medians in images/s; brackets
give the minimum and maximum of all three retained trials. Ratio is L3 / JPEG-DALI.

### Inference

| Model | JPEG/DALI | L3/DALI | Ratio |
|---|---:|---:|---:|
| ViT-Ti | 4,745 [4,709–4,838] | 2,833 [2,832–2,841] | 0.597× |
| SwinV2-T | 1,057 [1,055–1,060] | 995 [989–996] | 0.942× |
| MobileNetV2 | 5,223 [5,172–5,229] | 2,721 [2,716–2,733] | 0.521× |
| ResNet-50 | 2,044 [2,040–2,053] | 1,682 [1,679–1,682] | 0.823× |
| EfficientNet-B0 (eFUN RGB reference) | 4,205 [4,143–4,243] | 2,550 [2,533–2,551] | 0.606× |

### Training

| Model | JPEG/DALI | L3/DALI | Ratio |
|---|---:|---:|---:|
| ViT-Ti | 2,085 [1,848–2,265] | 1,972 [1,944–1,991] | 0.946× |
| SwinV2-T | 1,379 [1,378–1,379] | 1,232 [1,231–1,246] | 0.894× |
| MobileNetV2 | 1,978 [1,882–2,013] | 2,065 [2,004–2,081] | 1.044× |
| ResNet-50 | 1,715 [1,714–1,735] | 1,594 [1,581–1,595] | 0.929× |
| EfficientNet-B0 (eFUN RGB reference) | 1,652 [1,586–1,771] | 1,754 [1,749–1,761] | 1.062× |

### Validation Top-1 (%)

| Model | JPEG/DALI | L3/DALI | Difference (percentage points) |
|---|---:|---:|---:|
| ViT-Ti | 74.076 | 74.090 | +0.014 |
| SwinV2-T | 78.980 | 78.992 | +0.012 |
| MobileNetV2 | 70.788 | 70.718 | -0.070 |
| ResNet-50 | 74.870 | 74.740 | -0.130 |
| EfficientNet-B0 (eFUN RGB reference) | 76.770 | 76.750 | -0.020 |

Accuracy is identical across the three trials of each pipeline. L3 is bit-exact
against the Pillow-decoded RGB source, while native JPEG/DALI can produce slightly
different pixels; equal predictions across those decoders are not assumed.

### Storage and interpretation

| Converted subset | Images | JPEG (GB) | L3 (GB) | Raw RGB (GB) | Preparation + exhaustive validation (s) |
|---|---:|---:|---:|---:|---:|
| val | 50,000 | 2.05 | 30.63 | 39.32 | 583.5 |
| train | 98,560 | 3.89 | 59.41 | 77.51 | 1041.8 |

GB uses decimal bytes. The converted files occupy about 15× the source JPEG size
and 77–78% of raw RGB size. Preparation times include checking every image and must
not be reported as pure encoding throughput.

L3 is slower for all five inference workloads in this implementation. Training
ratios span 0.894–1.062×; the ViT-Ti and EfficientNet-B0 JPEG/DALI training trials
show appreciable variation, and the MobileNetV2 and EfficientNet-B0 ranges overlap
between pipelines. These shared-host measurements do not establish a consistent
training advantage or any convergence result. They are limited to the repaired
512×512 codec and current DALI port; the upstream decoder kernel was not optimized in that initial implementation.

Raw records, commands, cache observations and concurrent-load records are retained
in [`results/l3_baseline_20261006`](../../../results/l3_baseline_20261006).
The machine-readable [summary](../../../results/l3_baseline_20261006/summary.csv)
links every underlying result. No paper tables have been changed automatically.
