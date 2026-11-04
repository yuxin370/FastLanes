# L3 RGB baseline

This baseline adapts
[SNU-ARC/L3](https://github.com/SNU-ARC/L3/tree/3b8e52c226ac8749025e14858d5c5f260b562085),
commit `3b8e52c226ac8749025e14858d5c5f260b562085` (MIT license in `upstream/LICENSE`).
Cite Bae et al., *L3: Accelerator-Friendly Lossless Image Format for
High-Resolution, High-Throughput DNN Training*, ECCV 2022.

## Scope and implementation

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

Run from the repository root, in the `fastlens` environment:

```bash
export TMPDIR="$HOME/tmp"
python -m fastlens.benchmarks.baselines.l3.build \
  --nvcc "$CONDA_PREFIX/bin/nvcc" \
  --output "build/baselines/l3/libl3-patch32.so"
L3_LIBRARY="build/baselines/l3/libl3-patch32.so" \
  python -m unittest fastlens.benchmarks.baselines.l3.test_codec -v
```

The GPU tests cover 0–8-bit data, constant pixels, batches of 64 and partial
batches. The conversion command additionally checks every image against Pillow's
RGB decode of the source JPEG. L3 preserves those pixels exactly; it does not
recover pre-JPEG originals. Native JPEG decoders can differ slightly in pixels,
so the inference comparison reports measured accuracy rather than assuming equal
predictions between JPEG/DALI and L3/DALI.

```bash
python -m fastlens.benchmarks.baselines.l3.prepare --split val \
  --library "build/baselines/l3/libl3-patch32.so" --output "fastlens/data/baselines/l3"
python -m fastlens.benchmarks.baselines.l3.prepare --split train \
  --library "build/baselines/l3/libl3-patch32.so" --output "fastlens/data/baselines/l3"
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
python -m fastlens.benchmarks.baselines.l3.storage \
  --library "build/baselines/l3/libl3-patch32.so" \
  --output results/reproduced/l3/storage_train_full.json
```

This measures encoded file bytes, not filesystem allocation or conversion
throughput. The output is written only after the full manifest is processed.

## Paper-scale comparison

After preparation, pass the decoder and data to the chapter runner:

```bash
python -m fastlens.benchmarks.paper.section_6_2_3.deployment.run run \
  --phases training --l3-root fastlens/data/baselines/l3 \
  --l3-library build/baselines/l3/libl3-patch32.so
```

Inference uses a fresh DALI/L3 contract and the reference configuration described
in the [rebuilding guide](../../paper/REBUILDING_REFERENCES.md#connect-fresh-references-to-the-chapter-entry-points).
The chapter entry points keep the published sample populations, three trials,
model construction, preprocessing and timing boundaries.

## Results

Use the [published deployment table](../../paper/results/section_6_2_3_deployment.csv)
and [result notes](../../paper/results/README.md) for the paper evaluation.
The [configuration measurements](RESULTS.md) retain the 2026-10-06/07 validation,
storage and throughput observations, with their evidence availability.
