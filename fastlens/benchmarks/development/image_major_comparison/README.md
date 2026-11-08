# Image-major encoding comparison

These measurements retain the one-image-per-rowgroup encoding (manifest v2).
The tiled encoding (manifest v3) reduces selected payload for some crops, but
did not improve throughput in these tests. This is a comparison of the current
implementations, not a ranking for every workload. Both remain supported.

## Configuration and evidence

- NVIDIA GeForce RTX 4090 (SM 89); Python 3.11.15, PyTorch 2.11.0+cu128,
  CUDA 12.8, cuDNN 9.19.0, eight CPU threads.
- ImageNet validation JPEGs normalized to 512×512 RGB 4:2:0; the same ordered
  selection of 16,384 images from 50,000, seed 11997733, batch 64, all 64 frequencies.
- Shared sample-manifest SHA-256:
  `a0be6a09e41a7ec5d55995a7ecd68d96f699df5be289d5c928deb4edfe6a7a77`.
  The existing datasets have seven v2 partitions and 37 v3 partitions; partition
  boundaries are not controlled independently of encoding in these results.
- Recorded base revision: `716622adff117b8cfe7ac45b3a555b9be9387608`.
  The checkout also contained documentation/validation changes subsequently
  committed in `08aace0`; it was not a pristine checkout. The pipeline contract
  reported clean, hashed benchmark sources. Recorded native extension SHA-256:
  `f6d99a2c3b16fc4363cea37ee53c3eacc6f0786963b05cbb490061eeeb8f3f70`.

[pipeline.csv](pipeline.csv) retains all ten process measurements; A is v2 and
B is v3. [crops.csv](crops.csv) retains all forty crop measurements and their
vector/payload counters. Values are copied from the original measurements,
without dropping slower trials. These are development results, not replacements
for the paper's Block-major versus tiled Image-major measurements.

## End-to-end pipeline

The [acceptance tool](../../../tools/jpeg_dct/compact_v3_acceptance.py) ran five
fresh processes per variant in alternating A/B order. Each process measured two
256-batch traversals, without warmup, using the ViT-Ti DCT model in FP32 and
`ResizedCenterCrop_DCT(32,28)`. The CSV's `cold` means the first traversal;
all measured block-device reads were zero, so it does not mean cold disk.

| Median images/s | Image rowgroups (v2) | Tiled (v3) |
| --- | ---: | ---: |
| First traversal | 2426.03 | 2153.28 |
| Second traversal | 2960.52 | 2774.18 |

Semantic validation and sample order passed: 64 sampled model inputs/logits
matched the JPEG reference, and all 16,384 Top-1 predictions agreed. Throughput
and variability acceptance gates failed (per-variant throughput CV 7–13%).
The noisy pipeline run alone cannot establish the size of a stable regression.
It is not a native crop-pushdown measurement: native payload ratios were 1.0.

After preparing matched contracts and validation records as described by the
[acceptance tool guide](../../../tools/README.md), run:

```bash
python fastlens/tools/jpeg_dct/compact_v3_acceptance.py run-ab \
  --contract-a "$CONTRACT_A" --contract-b "$CONTRACT_B" \
  --validation-a "$VALIDATION_A" --validation-b "$VALIDATION_B" \
  --output-dir "$HOME/tmp/image-major-pipeline" --legs 10
```

## Native crop requests

[crop_compare.py](crop_compare.py) preserves the original measurement loop with
input/output paths exposed as arguments. It measures synchronous native int16
DCT read/decode/gather to GPU, without a model, resize or dequantization. Each
fresh process performs one full warmup traversal and one measured traversal.
Five trials reverse A/B order and rotate the requests. Files are warmed into the
OS page cache; decoded and plan caches are disabled. Prefetch uses eight workers,
depth 16; decode batches contain 64 rowgroups; crop execution mode is `auto`.

Requests below are `[x, y, width, height]` in source pixels. `random128` chooses
x and y independently as `randrange(25) * 16`, with seed 11997733, once for the
shared image order. All requests use all 64 frequencies.

| Request | v2 images/s | v3 images/s | v3 decoded vectors / full | v3 payload / full |
| --- | ---: | ---: | ---: | ---: |
| `[0,0,512,512]` | 1670.60 | 1304.63 | 100% | 100% |
| `[64,64,128,128]` | 10074.20 | 7063.94 | 50% | 33.06% |
| `[192,192,128,128]` | 10357.94 | 3651.21 | 100% | 100% |
| `random128` | 9730.90 | 5595.30 | 60.77% | 47.89% |

Throughput is the median; CV is 0.8–1.4%. All block-device reads were zero.
v2 decoded/read its full-image payload for every request. Ratios use each
encoding's separate full-image run, not the cropped request's touched-rowgroup
denominator. Thus the tiled path demonstrably pushes down some crops, while
rowgroup/decode overhead still outweighs that benefit in these cached tests.
For both encodings and every request, 64 sampled outputs and block/sample order
matched an independent CPU coordinate selection from full-image v2 coefficients
exactly. Source-to-DCT correctness is covered by the pipeline validation above.

With the two native datasets and the shared pipeline sample manifest available:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
python fastlens/benchmarks/development/image_major_comparison/crop_compare.py \
  --manifest-a "$IMAGE_ROWGROUP_MANIFEST" \
  --manifest-b "$TILED_MANIFEST" \
  --sample-manifest "$SAMPLE_MANIFEST" \
  --output-dir "$HOME/tmp/image-major-crops"
```

Select an idle NVIDIA GPU with `CUDA_VISIBLE_DEVICES`. The output directory must
be new; outputs include validation, individual trials and `summary.json`.
