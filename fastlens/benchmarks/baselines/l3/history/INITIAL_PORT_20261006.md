# Initial L3 port measurements (2026-10-06)

This superseded configuration used 64×64 patches and one decode stream.
Its encoded files are incompatible with the current 32×32 configuration;
[the current run guide](../README.md) prepares fresh data.

The record was preserved from documentation revision `8097fb3dc049`.
The local execution revision is not recorded. The upstream base is
`3b8e52c226ac8749025e14858d5c5f260b562085`. Throughput below was measured on
RTX 4090; the software, data populations and timing boundaries follow below.
Original `results/l3_baseline_20261006/` records and `summary.csv` are not
distributed, and no public download location is recorded. These tables preserve
the reported measurements; they do not supply the missing per-trial evidence.

All 60 formal trials completed: five models, two input pipelines, three trials each,
for both full-validation inference and training windows. No H100 throughput was measured for this configuration; all five H100 L3
training correctness checks passed. Software: PyTorch 2.11.0+cu128, DALI 2.2.0; plugin built
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
