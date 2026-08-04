# DCT-major benchmark run guide

## 1. Dataset contract

The default DCT-major input is:

```text
galp/data/imagedataset_dct/ImageNet-val/manifest.bin
```

Build the immutable access sidecars once. This reads existing metadata and
validates existing payload identity; it does not decode/re-encode JPEGs and
does not modify `.fls`:

```bash
./build/galp/tools/jpeg_dct/galp_block_major_access_tool \
  galp/data/imagedataset_dct/ImageNet-val/manifest.bin \
  --output-dir /tmp/galp-block-major-access-v1-real \
  --exhaustive-rank-validation \
  --output-json /tmp/galp-block-major-access-v1-real.json
```

Pass the directory into every formal contract with:

```text
--block-major-access-dir /tmp/galp-block-major-access-v1-real
```

The contract records the directory and companion-index SHA-256. Each
DCT-major pipeline sets `GALP_BLOCK_MAJOR_ACCESS_DIR` before constructing its
reader. Omitting the option is allowed only for an intentional legacy fallback;
it is not valid evidence for the planless result.

The runner requires manifest version 1 and records it as
`dct-major/spatial-major-image-minor`. The image-major control must be manifest
version 2 or 3. Both manifests must contain the same image count and their label
sidecars must contain identical label arrays.

The canonical sample view is built by sorting JPEG paths below
`DATA_ROOT/val`. It is not permuted. Every artifact records:

```text
ordinal == galp_image_id == physical image rank
shuffle == false
drop_last == false
```

`--sample-count` may be used for partial-tail checks. The runner derives the
number of measured batches so every requested sample is consumed exactly once;
an explicitly conflicting `--measurement-batches` value is rejected.

Every run requires a new or empty `--output-dir`. A non-empty directory is
rejected so an earlier immutable contract and its results cannot be silently
overwritten.

The contract resolves exactly one `_galp_direct_dct*.so` from the configured
Torch binding directory and records its path, file identity, size, and SHA-256.
The binary is also part of the runtime fingerprint gate, so rebuilding the
extension after contract creation invalidates the run instead of silently
mixing new source with an old binding. Formal planless validation additionally
requires the logical image-descriptor count to equal the measured image count;
counting the same descriptor table once per workset is rejected.

Use `--hash-samples` and `--hash-payloads` for an archival contract. These
options hash data outside the timed region and can take several minutes.

## 2. Workloads

### Feature extraction

`--workload feature-extraction` loads the original domain-specific checkpoint
and returns the output after:

```text
LayerNorm -> mean pool -> Linear(192,192) -> Tanh
```

The final `Linear(192,1000)` is omitted. No weight is changed and no retraining
is performed. `--feature-stage pooled` provides a diagnostic tap immediately
after mean pooling.

`--materialize-features` writes a standard `.npy` array. Its copy/write time is
reported separately and is excluded from the canonical input+model throughput.

### Evaluation

`--workload evaluation` retains the original classifier and records Top-1,
Top-5, complete prediction hashes and semantic samples.

## 3. Strict DCT A/B

`dct_major_full` uses the existing full-grid reader followed by RGB-no-more's
reference `ResizedCenterCrop_DCT(32,28)`.

For a Y-only grayscale JPEG, the legacy `ycbcr_dct_grid` planner cannot create
the grid. The benchmark-only full control therefore retries that image as a
`compact`, uncropped, forced `full-rowgroup-decode`: it reconstructs every Y
block at its original coordinate, creates the RGB-no-more reference zero Cb/Cr
grid, and only then runs the reference resize/crop. The successful compact
batch remains the source of all physical counters. This is a compatibility
fallback for full decode, not crop pushdown; native totals expose
`grayscale_full_fallback_count`, `grayscale_zero_chroma_image_count`, and
`grayscale_full_y_block_count`.

`dct_major_pushdown` passes the same output contract to GALP's native
transformed-grid reader. The default mode reads complete selected rowgroups and
decodes selected vectors. The following alternatives are exposed without
changing source:

```text
--dct-major-crop-execution-mode full-rowgroup-decode
--dct-major-crop-execution-mode rowgroup-read-selected-decode
--dct-major-crop-execution-mode vector-range-read-selected-decode
--dct-major-crop-execution-mode auto
```

`dct_major_legacy_pushdown` uses the identical manifest, crop mode, segment,
model, and sample order, but sets `enable_planless_execution=false`. It is the
authoritative eager-expanded baseline for planning time and Host/Pinned/GPU
peak comparisons.

Published results must include full, legacy-pushdown, and planless-pushdown in
one contract. Both physical layouts must first verify coefficient-exact against
the source JPEGs. The expanded spatial-major fixed transform can differ by one
integer DCT level because of arithmetic order, so the transformed-input gate is
`max_abs <= 0.001` and `mean_abs <= 0.0001`; the model-output cosine gate is
`>= 0.999`. These are the same limits used by the existing RGB-no-more system
benchmark. The validator additionally requires strict reductions in physical
bytes, actual decoded vectors and requested source blocks.

Evaluation also requires semantic-sample Top-1 agreement of 100% and full-run
Top-1 agreement of at least 99.9%. All observed accuracy and agreement values
remain in the report; the tolerance must not be used to hide a failed raw-DCT
coefficient check.

## 4. Segment locality

DCT-major places rows from adjacent images together inside one spatial block
group. The adapter therefore requests a physical segment, retains its native
output, and emits ordinary model batches in canonical order. A second segment
is prefetched while the current segment is consumed.

For batch 50 the single-run default segment is 1,000 images. This avoids a
cross-segment model batch while nearly filling a 1,024-row FastLanes vector.
It is a practical default, not a claim that 1,000 is universally optimal. The
complete suite must include:

```text
50, 250, 500, 1000, 1024 images/segment
```

On the current version-1 DCT-major manifest, the CPU planner selected-vector
coverage rises from 59.0436% at segment 50 to 92.2953% at 1,000 and 92.3077%
at 1,024. Thus 1,024 is the exact physical-alignment candidate and 1,000 is the
batch-aligned candidate. The suite chooses between them using the first
independent process's cold end-to-end throughput, then cold
time-to-first-batch. Hot-repeat and steady throughput remain diagnostic
columns and cannot select the published segment.

Cold scope begins at `run_pipeline` entry. It includes CUDA availability/device
setup, input verification, reader/adapter construction, checkpoint/model
construction, model priming, loader preparation, the data path, model, and
metrics. Binding, profile and optional post-decode diagnostic imports are timed
separately. Planless pushdown deliberately does not import the heavy full-grid
diagnostic module.

When the first real model output is available, `pipeline.py` emits a flushed
`GALP_FIRST_OUTPUT_READY` marker. `commands.json` records
`spawn_to_first_output_seconds` at receipt of that marker as the process-cold
TTFT, plus the complete subprocess wall time. The marker therefore excludes
later semantic NPZ/JSON serialization. In-process `time_to_first_batch_ms`
remains the application-scope cold metric. There are no warmup batches or
discarded repeats in the cold result. First segment preparation may overlap
model construction, but both remain inside their applicable cold clocks.

Reader startup diagnostics report manifest/path work, companion-index load,
metadata parse/index, static-profile construction, reader Python/native
constructor time, eager versus lazy shard counts, and descriptor cache
open/validation/current/bound bytes. The active-output CPU schedule and CUDA
kernel are separately reported; whole pipeline wall time must not be used as a
substitute for either.

Block-major bounded double buffering is controlled with:

```text
--block-major-double-buffer auto|on|off
```

`auto` currently retains bounded overlap. Before fixing the automatic choice
for a machine, compare it with full-capacity worksets using independent-process
ON/OFF/OFF/ON rather than sequential repeats in one process:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/diagnostics/run_cold_double_buffer_abba.py \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-cold-double-buffer-abba
```

The result is `cold_double_buffer_abba.json`. It compares process-scope cold
throughput, cold TTFT, subprocess wall time, workset/active-output counters,
and semantic outputs. It never uses a second repeat as the primary result.

After the current binding passes `run_suite.py --gates-only`, run the same
planless configuration in four independent process-cold A/B/B/A legs:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/diagnostics/run_cold_planless_abba.py \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-planless-cold-abba
```

A and B intentionally name the same immutable configuration; the balanced
labels expose order/page-cache effects rather than comparing implementations.
The report includes process TTFT, application TTFT/throughput, full subprocess
wall, CPU active-output scheduling, CUDA planless kernel, producer-active time,
and semantic equality. Hot batch-loop numbers are diagnostic only.

## 5. Formal run order

The preferred entry point is `run_suite.py`; it runs the following phases:

1. CPU compact-planless segment planning sweep;
2. five fail-fast planless GPU/kernel/cache/workset gates covering sequential,
   random, duplicate explicit-crop, cross-shard and grayscale requests;
3. feature and evaluation seven-pipeline smokes;
4. a GPU segment-locality sweep and automatic segment selection;
5. a strict 1,000-image-per-leg legacy/planless crop ABBA;
6. feature extraction 50K;
7. evaluation 50K;
8. model-only ceilings using `diagnostics/model_ceiling.py`.

Do not drop a negative or zero speedup. The performance result is accepted when
the sample/semantic/physical evidence is valid and repeat CV is at most 5%.

The decision gate is available independently of all long work. This command
runs the CPU planning sweep and exactly 23 requests across the five GPU cases,
then stops. It does not construct a model and cannot enter the locality, ABBA,
5K, or 50K phases:

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run_suite.py \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-planless-v2-gates-20260801-r1 \
  --gates-only
```

Success is `ok: true` and `gates_only: true` in `suite_results.json`. Each
device audit must additionally report one active-output schedule build, valid
monotonic offsets, schedule worksets equal executed worksets, exact active/
skipped/index/offset accounting, a positive separately timed planless CUDA
kernel, and zero expansion/cache counters. A failed gate returns nonzero and
leaves its JSON and log for inspection. Use a new output directory for a new
attempt, or the identical command with `--resume` after completed phases.

Before any long run, execute at least the two-image kernel gate below. It compares planless,
an identical planless repeat, and the expanded legacy control. It also
exercises the bounded multi-workset/double-buffer path:

```bash
CUDA_VISIBLE_DEVICES=0 \
./build/galp/tools/jpeg_dct/galp_block_major_plan_audit \
  galp/data/imagedataset_dct/ImageNet-val/manifest.bin \
  --descriptor-dir /tmp/galp-block-major-access-v1-real \
  --count 2 --compare-legacy --execute \
  --decode-batch-rowgroups 64 --prefetch-workers 2 \
  --workset-capacity-mib 512 \
  --output-json /tmp/galp-block-major-gpu-smoke-2-current.json
```

The gate requires planless selection, zero expanded/source-list/sort counters,
identical repeat hashes, maximum legacy difference at most one DCT integer
level, a positive planless-kernel launch count, exact strategy accounting, and
a bounded workset/double-buffer estimate no larger than its configured
capacity. It also requires zero oversized rowgroups, zero full-segment staging,
and zero exact-plan, decoded-rowgroup, sparse-vector, resize-weight and
conversion-matrix cache activity. These are emitted as
`execution_strategy_accounting_valid`, `execution_resource_bounds`, and
`execution_cache_contract_valid`; `--execute` returns nonzero if any gate
fails.

`run_suite.py` executes this gate plus four additional real-data variants:
random order, duplicate IDs with different explicit crops, a request crossing
the shard-0/shard-1 boundary, and grayscale image 239. Each variant repeats the
same semantic, determinism, strategy, resource and zero-cache checks before the
suite can start a locality or formal run.

The end-to-end runner exposes the same bound as
`--decode-workset-capacity-mib` (default 512). Its MiB value is stored in the
immutable contract; validation fails if native execution reports a different
byte capacity.

For a true ABBA sequence, first generate a one-repeat contract, then run the
dedicated executor:

```bash
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run.py \
  --preset e2e --repeats 1 \
  --pipelines dct_major_legacy_pushdown dct_major_pushdown \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-abba-contract \
  --dry-run

PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/diagnostics/run_crop_abba.py \
  --contract /tmp/galp-dct-major-abba-contract/contract.json \
  --output-dir /tmp/galp-dct-major-abba
```

The executor uses `legacy -> planless -> planless -> legacy`, keeps each leg in
a separate process/directory, and rejects sample-order, repeatability,
cross-path semantic, or Host/Pinned/GPU peak regressions.

## 6. Artifacts

Every run produces:

```text
contract.json
sample_manifest.json
canonical_index.csv
commands.json
pipeline_<name>.json
semantic_<name>.npz
results.json
results.csv
validation.json
report.md
```

Optional feature materialization produces `features_<pipeline>.npy`.

The primary ratios are:

```text
dct_major_pushdown / dct_major_full
dct_major_pushdown / dct_major_legacy_pushdown
dct_major_pushdown / image_major_pushdown
dct_major_pushdown / dali
dct_major_pushdown / pytorch
dali / pytorch
```

The first three are strict same-domain layout/crop comparisons. GALP-vs-RGB ratios are
deployment-level context because they use domain-specific checkpoints.
