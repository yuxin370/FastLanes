# Author-native PCR TensorFlow reproduction

This experiment builds the authors' C++ `DatasetOp` at TF_PCR commit
[`c3cc6a9bad115925cd398d01cedd85af68aa1be2`](https://github.com/mkuchnik/TF_PCR/tree/c3cc6a9bad115925cd398d01cedd85af68aa1be2)
against CPU TensorFlow 2.15.1.
The upstream fork targets TensorFlow 2.5.1. The adapter retains its native
reader, protobuf parsing and buffer reuse; build-time edits adapt the
`Status` API, relocated metrics/name utility headers and the TSL file interface.
The protobuf compiler uses the installed TensorFlow schema include path, and
the op links the installed `libtensorflow_cc` for its dataset name utilities.
The dataset algorithm and `InputDatasets` interface remain unchanged. This is
an API adaptation of the author's C++ implementation, not the original TF 2.5.1
binary. The wrapper uses eight parallel file readers and disables read buffering
to keep scan-prefix reads bounded.

`prepare.py` rewraps existing raw PCR scan fragments in the authors'
`ScanGroup` protobuf layout without JPEG re-encoding. `validate.py` checks
every reconstructed JPEG, label and interleaved sample position.

`run.py` calls the authors' TensorFlow center-crop/bicubic preprocessing and
`data_prefetcher`, then runs the existing PyTorch RGB checkpoint. It compares
predictions by sample ID because native file interleave changes their order.
This preprocessing differs from the PIL path in the paper-comparable
[PCR experiment](../pcr/README.md). A single run here is a pipeline check,
without controlled cache state or a bandwidth cap; it is not a paper-table
throughput measurement or a training-convergence reproduction.

## Completed checks (2026-10-17)

The native op compiled and loaded successfully. All 4096 images at each of
scans 2, 5 and 10 matched the raw adapter byte for byte, with identical labels
and interleaved sample positions. Rewrapping produced 16 protobuf records
totalling 161,058,471 payload bytes without re-encoding source JPEGs.

End-to-end runs used an RTX 4090, eight workers on CPUs 8–23, batch size 64,
the existing MobileNetV2 RGB checkpoint, TF 2.15.1 and Torch 2.11.0+cu128.
Complete scan 10 and original JPEG produced identical Top-1 predictions and
identical ordered Top-5 lists for every sample ID.

| TF input path | Read payload MB | Top-1 | Top-5 | Single-run images/s |
|---|---:|---:|---:|---:|
| Original JPEG files | 168.10 | 70.093% | 89.185% | 917.5 |
| Native PCR, scan 10 | 161.06 | 70.093% | 89.185% | 752.8 |
| Native PCR, scan 5 | 56.12 | 67.334% | 87.573% | 890.3 |
| Native PCR, scan 2 | 36.00 | 59.717% | 82.690% | 903.4 |

Read counts exclude the small external index. All PCR configurations share
the same complete store. These timings are single pipeline checks with no
bandwidth cap or controlled cache state; they do not establish a throughput
ranking. The TF preprocessing accuracy also differs from the PIL experiment.
No training convergence experiment was run.

Raw results, including predictions keyed by sample ID, are in
`$HOME/tmp/pcr-tf-reproduction/results/{jpeg,pcr10,pcr5,pcr2}.json`.

## Reproduce

Run from the FastLanes repository. Use an isolated Python 3.11 environment with
`tensorflow-cpu==2.15.1`, `protobuf==4.25.3` and the existing Torch installation.
Install [protoc 3.21.9](https://github.com/protocolbuffers/protobuf/releases/tag/v21.9)
under `$HOME/tmp/pcr-tf-reproduction/protoc`.
The original release checkout is pinned to
`aa76741f5dc6dcb6d7676c62e2fd53a78f43f142`.

```bash
export TMPDIR="$HOME/tmp"
PCR_TF_ROOT="$HOME/tmp/pcr-tf-reproduction"
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python -m venv \
  --system-site-packages "$PCR_TF_ROOT/venv"
PCR_TF_PY="$PCR_TF_ROOT/venv/bin/python"
"$PCR_TF_PY" -m pip install tensorflow-cpu==2.15.1 protobuf==4.25.3
"$PCR_TF_PY" -m galp.benchmarks.pcr_tensorflow.build \
  --build-dir "$PCR_TF_ROOT" --protoc "$PCR_TF_ROOT/protoc/bin/protoc" --jobs 4
"$PCR_TF_PY" -m galp.benchmarks.pcr_tensorflow.prepare \
  --source "$HOME/tmp/pcr-adaptation/val4096" \
  --output "$PCR_TF_ROOT/val4096" --build-dir "$PCR_TF_ROOT" --workers 8
"$PCR_TF_PY" -m galp.benchmarks.pcr_tensorflow.validate \
  --data "$PCR_TF_ROOT/val4096" --op "$PCR_TF_ROOT/pcr_dataset.so"
"$PCR_TF_PY" -m galp.benchmarks.pcr_tensorflow.run \
  --data "$PCR_TF_ROOT/val4096" --op "$PCR_TF_ROOT/pcr_dataset.so" \
  --upstream-release "$HOME/tmp/pcr-adaptation/PCR_Release" \
  --route jpeg_files --output "$PCR_TF_ROOT/results/jpeg.json"
"$PCR_TF_PY" -m galp.benchmarks.pcr_tensorflow.run \
  --data "$PCR_TF_ROOT/val4096" --op "$PCR_TF_ROOT/pcr_dataset.so" \
  --upstream-release "$HOME/tmp/pcr-adaptation/PCR_Release" \
  --route pcr --scans 10 --output "$PCR_TF_ROOT/results/pcr10.json" \
  --reference "$PCR_TF_ROOT/results/jpeg.json"
```

Choose an available GPU and CPU affinity before running, and use fresh output
paths when repeating a check. This machine used
`CUDA_VISIBLE_DEVICES=GPU-40c637bd-acf5-ea1a-0df8-617138228467` and
`CUDA_MPS_PIPE_DIRECTORY=$HOME/tmp/pcr-tf-reproduction/no-mps` without changing
the existing MPS service.

## Deployment training adapter

`training.py` supplies inputs to `experiments/remaining_baseline_training.py`;
the existing model initialization, optimizer, 32-update warmup, complete state
restoration, and 65,536-image measurement loop are reused. The full source is
the existing 98,560-entry `results/preparation/p0_ffcv/samples.json` selection,
in its original order. Records contain 256 images. Parallel interleave uses
`block_length=256` to preserve that order.

This training adapter preserves the native C++ reader and TF JPEG decode/crop/
resize execution, with the paper's existing input request: seed 11997733,
epoch 2, per-sample keyed boxes/flips, 224-pixel bilinear resize without
antialiasing, and normalization to `[-1,1]`. These differ from the author's
random TF crop, bicubic resize and ImageNet normalization used in the pilot
above. TF `INTEGER_ACCURATE` JPEG decoding is explicit: its default fast IDCT
failed the original PIL/OpenCV input-error gate; accurate IDCT passed the
unchanged four-image gate. The adapter also checks original JPEG inputs under
identical TF operations and exact first-batch restoration after rebuilding
the source. It adds no TensorFlow RNG state to the optimizer protocol because
all augmentation parameters come from the existing deterministic decisions.

```bash
"$PCR_TF_PY" -m galp.benchmarks.pcr_tensorflow.prepare \
  --training-manifest results/preparation/p0_ffcv/samples.json \
  --output "$PCR_TF_ROOT/train98560" --build-dir "$PCR_TF_ROOT" \
  --workers 16 --jpegtran /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/jpegtran
"$PCR_TF_PY" experiments/remaining_baseline_training.py \
  --gpu H100 --model mobilenet24 --path pcr --check \
  --pcr-data "$PCR_TF_ROOT/train98560" --pcr-op "$PCR_TF_ROOT/pcr_dataset.so" \
  --pcr-upstream-release "$HOME/tmp/pcr-adaptation/PCR_Release" \
  --output "$PCR_TF_ROOT/training/H100/mobilenet24/check"
```

For a measured trial, replace `--check` with `--trial 0` and
`--correctness-report .../check/result.json`, and choose a fresh output path.
The four RGB training families are ViT-Ti, SwinV2-T, MobileNetV2 and
EfficientNet-B0. A completed window is throughput evidence, not convergence.

The native training source was prepared in 237.07 seconds: 98,560 pixel-verified
images, 385 records, and 3,729,592,528 payload bytes. H100 and RTX 4090 diagnostics
passed for all four families, with 1,024 warmup and 1,024 measured images per
check. Each verified the original input-error limits, original-JPEG TF
reference, exact first-batch reconstruction, complete training-state
restoration, finite loss, and a real parameter update. All four initial model
hashes match the selected original PyTorch RGB baselines. These diagnostics used
the older `CUBLAS_WORKSPACE_CONFIG=:4096:8` launch setting. They are retained under
`results/pcr_native_20261017/training/<gpu>/<model>/check_cublas4096/` and are not
formal throughput results. The current paper protocol requires that variable
to be unset; formal trials require a fresh matching `check/` result.

`read_boundary.cc` counts and optionally paces native `ReadNBytes` calls before
their buffers become available. All reader threads share one rate with no
idle credit; the source remains in the warm page cache. `pcr_boundary_begin`
takes bytes per second, with zero selecting counting only. Load the library
with `tf.load_op_library` before opening its counter exports with `ctypes`.
The external index is excluded from payload counts.

## Paper deployment and bandwidth configuration

The paper experiment uses `training.py` and `inference.py`, independently of
the 4,096-image pilot above. Its result root is
`results/pcr_native_20261017/`. The measured matrix contains:

| Experiment | GPUs | RGB families | Input settings | Trials | Points |
|---|---|---|---|---|---|
| Training | RTX 4090, H100 | ViT-Ti, SwinV2-T, MobileNetV2, EfficientNet-B0 | All 10 scans | 3 | 24 |
| Inference | RTX 4090, H100 | Same four families | Scans 10 and 9 | 3 | 48 |
| Bandwidth | RTX 4090, H100 | MobileNetV2 | Crops 448/160; 125/500/2,000 MB/s; JPEG records/scans 10/scans 9 | 3 | 108 |

MobileNet-DCT24 and DCT32 share the MobileNetV2 RGB measurements. ResNet is
excluded from the paper's RGB deployment comparison. The main deployment
figure uses all scans. Scan 9 is a separate fidelity/throughput comparison;
both scan settings use the same complete stored representation.

Every formal launch unsets `CUBLAS_WORKSPACE_CONFIG`, binds CPUs
`0-23,48-71` and memory to NUMA node 0, and sets OMP/MKL/OpenBLAS and Torch
host threads to eight. GPU measurements run serially. Training retains
16 input workers, 32 warmup updates, state/input restoration, and 64 measured
updates over 65,536 distinct images (microbatch 64, accumulation 16).
ViT uses FP32; other training models use BF16 autocast. Inductor uses its
default mode, with TF32 and CUDA Graphs disabled. Each GPU/model first passes
a fresh check under this same environment.

Inference uses eight input workers, FP32, no compilation, no TF32, and the
full 50,000-image validation set. Batch size is 50 for Swin and 64 otherwise.
Swin retains its original shuffled sample order and 256-pixel whole-source
resize; other models retain physical sample order and a 448-pixel center crop
resized to 224. Labels follow each checkpoint's mapping. The native adapter
uses TensorFlow bicubic resize and each model's normalization. Thus its
accuracy is reported separately from pipelines using PIL preprocessing.
Timing includes reader construction/startup, model execution, CE/Top-1/Top-5,
prediction copies, and final synchronization; model loading, tracing, 20 model
warmup batches, and reader shutdown are excluded. Each model/GPU has a
separate complete original-JPEG reference. Full-scan PCR must reproduce all
of that reference's ordered Top-5 predictions by sample ID.

Bandwidth runs use the original 16,384-image physical prefix, batch 64, and
the same worker/precision settings. Each configuration has a separate full
warmup pass before its three measured passes. JPEG records contain original
JPEG bytes in a single scan group and use the same native reader and aggregate
read limiter as PCR. Timers include CE/Top-1/Top-5 computation, Top-1 prediction
copies, and reader shutdown, matching the original bandwidth experiment.
Rates are decimal MB/s and apply to warm-cache application reads; this is
a bandwidth emulation, not a remote-storage deployment. Native read counters
are checked against the selected record sizes. The FastLens comparison reuses
the paper's existing compact DCT24 measurements, including the H100 2,000 MB/s
recheck. RGB and DCT configurations have different inputs and checkpoints, so
this comparison does not isolate a storage-format speedup.

Formal result locations are:

```text
training/<gpu>/<model>/trial_<0,1,2>/result.json
inference/<gpu>/<model>/pcr_s<10,9>/trial_<0,1,2>/result.json
bandwidth/<gpu>/crop<448,160>_MBs<125,500,2000>/<jpeg_record,pcr_s10,pcr_s9>/trial_<0,1,2>/result.json
```

The finite launch scripts used on this host are
`$HOME/tmp/pcr-tf-reproduction/train_matrix.py` (select `--gpu` and optional
`--models`) and `$HOME/tmp/pcr-native-formal/run_all.py` (the 156 inference and
bandwidth points). Individual inference/bandwidth commands use the module's
`--data`, `--op`, `--upstream-release`, `--model`, `--route`, `--scans`,
`--protocol`, `--source-crop`, `--read-mbps`, `--reference`, and `--output`
arguments. Complete launches and raw measurements are retained alongside each
result. Neither fixed training windows nor validation passes establish
training convergence.

The formal matrix is complete: 24 training, 48 inference, and 108 bandwidth
points. All eight training checks passed with `CUBLAS_WORKSPACE_CONFIG` unset.
Full-scan inference matched every ordered Top-5 prediction of all eight
complete TensorFlow JPEG references. The 12 bandwidth configurations use the
same ordered 16,384 sample IDs as the corresponding FastLens measurements;
read counts, payload sizes, and aggregate rate limits were verified. Each
bandwidth configuration has its own accuracy results: for the 160-pixel crop,
nine scans reduce Top-1 from 57.44% to 51.50%, so the full-validation accuracy
table cannot substitute for this crop-specific comparison.
