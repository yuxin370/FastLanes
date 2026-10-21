# PCR storage adaptation

This adapts [Progressive Compressed Records](https://github.com/mkuchnik/PCR_Release)
at commit `aa76741f5dc6dcb6d7676c62e2fd53a78f43f142` to the repository's existing
MobileNetV2/ResNet-50 RGB checkpoints and validation preprocessing.
It is a **PCR storage adaptation with a Python/PyTorch reader**, not a reproduction
of the original optimized TensorFlow runtime.

The payload follows the upstream `tensorflow/PCR_Converter.py` raw-byte format:
proto2 `MetadataRecord`, then scan groups containing the same scan from each
image. The external per-image scan-length index is JSON rather than SQLite or
RocksDB. `jpegtran -optimize -progressive` performs lossless coefficient-domain
conversion through stdin/stdout; source JPEGs are never overwritten. Every
converted image is checked for exact RGB pixel equality with the source.

Three input routes use the same Pillow decoder and torchvision preprocessing:

- `jpeg_files`: original individual JPEGs through the existing `rgb.Inputs`
  deployment reader and a DataLoader with model-sized batches. The rate-limited
  variant adds pacing around the file read without changing its RGB tensors.
- `jpeg_record`: original JPEGs packed at the same record granularity, isolating
  packing effects from progressive encoding and truncation.
- `pcr`: one shared stored representation, reading only the requested scan-group
  prefix and appending JPEG EOI after reconstructing each image.

Preparation defaults to the existing deterministic shuffled validation selection;
`--physical-order` selects the order used by the full-validation CNN deployment
experiments. Packed records are processed in parallel by DataLoader workers,
then split into model batches in the parent. Full scans and packed JPEGs must
match the original JPEG predictions. Partial scans are evaluated against the **unchanged
RGB checkpoint**; their accuracy is not evidence about retraining convergence.
This is a deployment comparison, not a matched DCT-input comparison.

Bandwidth simulation uses one shared completion schedule across all workers,
without accumulating idle credit. A completion deadline is reserved before the
read and awaited afterward, matching `fastlens/benchmarks/paper/shared/input_boundary_controls.cpp`.
Rates are decimal MB/s; source pages are warm. This does not model cold storage,
remote latency or a real network. Issued reads, persistent payload sizes and
external index size are separate. H2D input bytes are calculated from tensors,
not measured with CUPTI. The timer includes reader/worker startup, input work,
model execution, Top-5 prediction copies, metrics and final synchronization.
Fixed-count iteration keeps worker shutdown outside the timer, matching
`fastlens/benchmarks/paper/shared/remaining_baseline_inference.py`. Model loading,
20 model warmup batches, external index parsing and warming the source pages
are excluded.

From the repository root, with the existing CUDA environment on PATH:

```bash
export PATH=$CONDA_PREFIX/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$HOME/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY="$HOME/tmp/pcr-paper-config/no-mps"
python -B -m unittest fastlens.benchmarks.baselines.pcr.test_pcr -v
numactl --physcpubind=8-23 --membind=0 python -B -m fastlens.benchmarks.baselines.pcr.prepare \
  --output "$HOME/tmp/pcr-paper-config/val50000" --count 50000 --physical-order \
  --record-images 256 --workers 16
python -B -m fastlens.benchmarks.baselines.pcr.sweep --data "$HOME/tmp/pcr-paper-config/val50000" \
  --output "$HOME/tmp/pcr-paper-config/mobilenet4090" --gpu 4090 --trials 3 \
  --cpus 8-23 --scans 10 9 --read-mbps 0 125 25 \
  --reference results/remaining_baselines_20261012/inference/4090/mobilenet24/pytorch/trial_0/result.json
```

Use fresh output directories. This runs original JPEG, packed JPEG and PCR scans
10/9 at unlimited, 125 and 25 MB/s, interleaving three trials (36 measurements).
It uses the existing inference resource monitor and CPU/memory NUMA node 0;
the selected GPU must be free. GPU results, predictions and commands remain in
each trial directory, with medians/ranges in `summary.csv`.
Use `--read-mbps 25` on the sweep to isolate a lower-bandwidth regime. The pilot
JPEGs average about 41 kB/image, so 125 MB/s permits roughly 3,046 images/s from
storage alone, above this CPU input path's observed delivery rate.
The separate MPS pipe path is necessary on the current host: its existing MPS
server exposes H100 only, making a directly selected RTX 4090 unavailable to
clients using the default pipe. This setting leaves that server unchanged.

The test suite checks RGB/grayscale conversion, source preservation, full
reconstruction, decoding truncated scan prefixes, genuine partial reads,
packed JPEG identity, malformed JPEG boundaries, aggregate bandwidth pacing
and exact paced/unpaced RGB reader tensor equality.
Requests exceeding a record's available scans return full fidelity, including
records containing only grayscale JPEGs.

## Full-validation deployment results (2026-10-17)

The full comparison uses all 50,000 ImageNet-512 validation images in the
physical `fastlens_image_id` order, including the final 16-image model batch.
Preparation took 119.69 seconds with 16 workers and verified exact RGB pixel
equality for every converted image. There are 196 records of at most 256 images.
Scans 8 and 9 were first checked on the existing 4,096-image selection; scan 9,
the highest incomplete prefix, is the sole truncated configuration in this
comparison. Its fixed-checkpoint accuracy is reported separately from full
fidelity; this is not a training-convergence experiment.

All configurations use RTX 4090, CPU affinity 8–23, NUMA memory node 0, model
batch size 64, eight model threads, FP32 with TF32 disabled, and 20 untimed model
warmup batches. The RGB paths use eight loader workers, one PyTorch thread per
worker, and the unchanged MobileNetV2 RGB checkpoint and preprocessing. BLAS
thread environment variables are one. The original JPEG, packed JPEG, PCR scan
10 and PCR scan 9 run at unlimited, 125 and 25 decimal MB/s. Each round also
includes FastLens at unlimited and 25 MB/s; configurations rotate between rounds.
The bounded design is 14 configurations and three repetitions, all serial.

FastLens uses the existing `remaining_baseline_inference.py` deployment runner,
full source coefficients, the existing 448×448 source crop and the MobileNetV2
DCT24 checkpoint. A small scratch wrapper loads the existing
`libinput_boundary_controls.so` to impose the same shared read-completion model
and count `.fls` reads. It changes no product code. Its DCT checkpoint has a
different accuracy from the RGB checkpoint, so this is a comparison of complete
deployments, not equal model inputs or equal accuracy. The author's native C++
kernel has also been adapted and validated in the separate
[TensorFlow reproduction](../pcr_tensorflow/README.md). Its TensorFlow
preprocessing differs, so those results are not pooled into this 50,000-image
Pillow comparison; this adapter remains a Python/PyTorch implementation.

Full-fidelity RGB runs are checked against every Top-5 prediction in
`results/remaining_baselines_20261012/inference/4090/mobilenet24/pytorch/trial_0/result.json`.
FastLens is checked against the corresponding `fastlens/trial_0/result.json`.
Sample identities and order match those references. Historical **throughputs**
are not reused: their NUMA-node CPU placement differs from this cohort's fixed
16-CPU affinity. An existing conversion job was left running on CPUs 24–47.
The separate TensorFlow compilation and GPU checks occurred only between complete trials;
the existing ResourceGuard monitors the target GPU and retains host observations.

The PCR payload occupies 1,963,862,562 bytes, shared by scans 10 and 9. Scan 9
issues 1,236,081,168 bytes per full pass. Original JPEGs occupy 2,051,552,155 bytes;
their packed control occupies 2,051,596,539 bytes. The shared external JSON index
occupies 20,117,406 bytes and is read before timing; it includes sample metadata
and both comparison routes. The existing full FastLens store occupies
4,252,434,994 bytes, of which its deployment path issues 3,459,862,361 bytes per
pass. These application-read counters do not measure physical disk traffic.

All 42 measurements completed. Throughput is the median of three interleaved
repetitions, with min–max in brackets. Accuracy and issued bytes are identical
across repetitions and bandwidth settings. GB and MB below are decimal.

| Deployment | Issued GB/pass | Top-1 (%) | Top-5 (%) | Unlimited (images/s) | 125 MB/s (images/s) | 25 MB/s (images/s) |
|---|---:|---:|---:|---:|---:|---:|
| Original JPEG, RGB | 2.052 | 70.746 | 89.676 | 1,551.7 [1,502.2–1,571.6] | 1,398.3 [1,394.4–1,420.1] | 603.3 [603.1–603.5] |
| Packed original JPEG, RGB | 2.052 | 70.746 | 89.676 | 1,443.1 [1,438.7–1,454.5] | 1,363.5 [1,352.5–1,371.9] | 597.6 [596.8–597.7] |
| PCR scan 10, RGB | 1.964 | 70.746 | 89.676 | 1,065.3 [1,055.7–1,069.0] | 1,016.2 [1,009.4–1,025.4] | 619.3 [618.1–619.6] |
| PCR scan 9, RGB | 1.236 | 69.426 | 89.014 | 1,203.3 [1,201.6–1,221.1] | 1,148.3 [1,147.2–1,175.5] | 965.1 [963.4–971.6] |
| FastLens, DCT24 | 3.460 | 67.704 | 88.010 | 5,165.4 [5,164.2–5,184.5] | — | 352.1 [351.9–352.1] |

Every full-fidelity RGB run matches all 50,000 reference Top-5 predictions,
including their order; FastLens similarly matches its own DCT24 reference.
Scan-9 predictions are identical across its nine runs. FastLens at 125 MB/s was
not measured in this bounded matrix.

This adapter is slower than original JPEG at unlimited and 125 MB/s. At
25 MB/s, full-fidelity PCR provides a small 1.027× throughput improvement over
original JPEG. Scan 9 reduces issued bytes by 39.75% relative to original JPEG
and reaches 1.600× throughput, with a 1.320-percentage-point Top-1 decrease.
It is an accuracy–bandwidth tradeoff, not an equal-accuracy speedup. FastLens
has the highest unlimited throughput but the lowest throughput at 25 MB/s,
consistent with its larger issued-read volume. Its separate checkpoint's
accuracy must accompany that deployment comparison.

For every RGB route, calculated input-image and label H2D traffic is
30,106,000,000 bytes per pass: CPU decode and preprocessing produce the same
tensor shapes even for truncated scans. PCR's storage-read savings therefore
do not reduce this adapter's H2D input traffic. These results characterize
warm-page read-completion limits and this CPU Python/PyTorch adapter, rather
than physical network performance, the optimized TensorFlow runtime or
training convergence.

Prepared PCR inputs, raw predictions, commands, resource observations and
results are at `$HOME/tmp/pcr-paper-config/`. The exact bounded drivers are `comparison.py` and
`comparison_rest.py`; the latter continues the final two rounds after the first
round's resource handoff. `fastlens_control.py` is the native-control wrapper.
The combined measurements are in `comparison/trials.csv`, with per-configuration
medians and ranges in `comparison/summary.csv`. The first controller was stopped
only at its idle prompt after a complete trial; no completed measurement was rerun.

## Pilot results (2026-10-17)

These historical subset results used the original pilot implementation:
all routes, including `jpeg_files`, processed 256-image records; timing included
worker shutdown; pacing reserved completion after the actual read. They are
retained as feasibility observations and are not pooled with the full-validation
measurements using the updated deployment timing above.

RTX 4090, MobileNetV2 RGB, 4,096 validation images, FP32, batch 64, eight input
workers, 256 images/record, warm source pages and CPU/memory NUMA node 0.
The main sweep completed 51 measurements; the 25 MB/s follow-up completed 15.
Every configuration has three interleaved repetitions. Throughput below is the
median with min–max in brackets; accuracies were identical across repetitions.

| Input | Issued MB | Top-1 (%) | 25 MB/s (images/s) | 125 MB/s (images/s) |
|---|---:|---:|---:|---:|
| Packed original JPEG | 168.10 | 70.87 | 487 [484–489] | 1,008 [877–1,012] |
| PCR scan 10 | 160.94 | 70.87 | 476 [473–476] | 831 [818–832] |
| PCR scan 5 | 56.06 | 65.75 | 890 [881–910] | 990 [905–1,047] |
| PCR scan 2 | 35.98 | 60.64 | 1,022 [986–1,028] | 983 [963–1,030] |

Scan 5 and scan 2 reduce issued bytes by 66.65% and 78.60%, yielding 1.83x and
2.10x throughput at 25 MB/s. Their fixed-checkpoint accuracy drops by 5.13 and
10.23 percentage points. These are accuracy–performance tradeoffs, not matched
accuracy speedups. Full-fidelity PCR and packed JPEG predictions match all
4,096 original-JPEG predictions in every measurement. Conversion checked every
image's RGB pixels; scan boundaries also matched the upstream C scanner.

The 125/500/2000 MB/s measurements do not show a consistent truncation speedup;
CPU decoding/preprocessing and the short window limit interpretation, and some
throughput ranges are broad. These results do not characterize the original
TensorFlow implementation or training convergence. They also are not a direct
FastLens comparison: this pilot uses the common RGB model and input path.

All scan choices use the same 160.94 MB PCR store. The shared external JSON
index occupies 1.64 MB and contains both comparison routes and sample metadata.
Calculated model-input H2D traffic is 2,466,283,520 bytes for every route because
decoding and preprocessing happen on CPU. Storage-read savings do not imply
H2D savings for this adapter.

The results support a full-scan deployment control and a separate truncated-scan
accuracy–bandwidth comparison. No paper tables or headline results were changed.
Raw predictions, commands, host-load observations and summaries are retained at:

- `$HOME/tmp/pcr-adaptation/mobilenet4090_nomps/summary.csv`
- `$HOME/tmp/pcr-adaptation/mobilenet4090_25mbps/summary.csv`
