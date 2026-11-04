# L3 configuration measurements (2026-10-06/07)

These observations use 32×32 patches and 32 decode streams. The source is
SNU-ARC/L3 commit `3b8e52c226ac8749025e14858d5c5f260b562085` with the local
codec repairs and DALI port described in [the run guide](README.md).
The notes were retained at documentation revision `8097fb3dc049`; the exact
local execution revision is not recorded here. GPUs, sample counts and timing
scope are specified below. The initial port records PyTorch 2.11.0+cu128,
DALI 2.2.0 and CUDA 12.8; separate software versions for the corrected runs
are not given in the retained notes.

The original `results/l3_patch32_20261006/` run tree, conversion JSON and profiler
captures are external and have no public download location recorded here.
The tables below preserve their reported numbers. They are separate from the
later [published deployment table](../../paper/results/section_6_2_3_deployment.csv)
and its [trial records](../../paper/results/README.md#trial-records-and-statistics).
Rebuild data and run fresh measurements with [the run guide](README.md); each
new run must record its own code, hardware and input identities.

## Full training storage

The full training measurement (`l3_patch32_20261006/storage_train_full.json`, external and not distributed)
completed on 2026-10-07: all 1,281,167 images passed pixel comparison.
The encoded total is 737,018,660,283 bytes: 735,034,132,600 payload bytes and
1,984,527,683 metadata bytes, with no separate persistent index. The corresponding
JPEG files total 50,600,614,992 bytes, giving 14.5654x storage amplification.
Encoding and exhaustive verification took 16,136.3 seconds; encoded files were
not retained. The training-window performance runs still use their original
98,560-image converted subset.

## Configuration validation and RTX 4090 measurements

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
`l3_patch32_20261006/profile/PRO6000_mobilenet24` trace and diagnostic summary (external, not distributed)
are retained separately from performance trials.

All 30 corrected 4090 inference trials completed (50,000 images
per trial, three trials per model and pipeline). Medians in images/s:

| Model | JPEG/DALI | L3/DALI | L3 / JPEG-DALI |
|---|---:|---:|---:|
| ViT-Ti | 4,802 | 4,496 | 0.936× |
| SwinV2-T | 1,059 | 1,041 | 0.983× |
| MobileNetV2 | 5,256 | 5,248 | 0.998× |
| ResNet-50 | 2,046 | 2,025 | 0.990× |
| EfficientNet-B0 (eFUN RGB reference) | 4,234 | 4,137 | 0.977× |

L3 Top-1 is identical across repeats for all five models. These measurements use
the paper's configuration and the author's stream arrangement. The
`l3_patch32_20261006/summary.csv` and raw results are external and not distributed.

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

The host was shared; accepted H100 trials recorded no target-GPU contention.

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
