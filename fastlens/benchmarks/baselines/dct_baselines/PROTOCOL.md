# DCT baseline protocol

[Run guide](README.md) · [Results and evidence](RESULTS.md)

## Inputs and equivalence

`prepare` freezes `matched.json`, `source.json`, `profile.json` and
occurrence-level `trace.jsonl`. The source records JPEG hashes, original
quantized coefficients (q), quantization tables (Q), component geometry,
sampling, labels and image identities. A matched store preserves these values;
it does not resize or re-encode JPEGs. Record the actual source production
recipe separately from model preprocessing.

`verify-inference` loads the pretrained checkpoint and checks source q/Q,
model inputs, logits, top-5 predictions and task metrics against the independent
validation transform. Inference uses FP32, the validation crop and no RandAugment
or Mixup. The matched protocol declares `atol=rtol=1e-5` before execution and
requires identical top-1 predictions and top-5 class sets. Tolerances stay fixed
when a check fails. Training has separate calibration and precision rules.

Transformer inference preserves the upstream operation order
`-1+((q+1024)/2040)*2`, with separately rounded FP32 operations. CNN normalization
and projection follow the selected model profile. The directed streamed
projection regression covers the canonical clamp containing zero; custom
clamps with padding are outside its validated scope.

Passing evidence binds source, contract, checkpoint state, implementation,
device and store content. FastLens verification also binds compressed shards,
metadata, access sidecars and native binaries. Timing validates these identities
before starting; changed files require new verification. Files must remain
unchanged during a run. A passing subset check covers only that subset.

The reusable verifier (`python -m
fastlens.benchmarks.baselines.dct_baselines.verification`) caches source checks
and reference fingerprints by source, trace, geometry, normalization, labels,
batch size and numerical environment. Bit-identical inputs may reuse checkpoint
outputs; differing batches require independent input and model comparisons.
Evidence records whether model outputs were computed directly or reused.
FastLens uses its own full verifier.

## Storage and resource accounting

IC/FS readers inflate complete image/batch or group/diagonal members, then gather
requested frequencies and block rectangles in logical order, including repeated
IDs. Payload identities, NPZ CRC, shape and dtype checks protect stored data.
`dtype_policy=int16` gives a common-width control. IC `safe` selects width over
the whole image/batch; FS `paper` and `safe` select width over each complete
diagonal. Overflow fails the build. Incomplete builds have no usable `store.json`.

Offline `build` uses eight processes by default, one complete storage group per
task; `--workers` controls concurrency independently of inference workers.
The Python `build_store` API defaults to one worker. Conversion includes JPEG
coefficient extraction and store encoding, excluding creation of the common
JPEG source. Storage totals include archive headers, labels, Q, geometry and
manifests. Separate FixedK stores count as separate copies.

Default reader budgets are 512 MiB for active decoded arrays, 1 GiB per reader
for cached/active/gathered arrays, zero decoded cache, and 4 GiB for retained
host input. Set `--decode-budget-mib`, `--reader-budget-mib`,
`--decoded-cache-mib` and `--host-input-budget-mib` during preparation.
These bounds exclude manifest objects, inflate/transform workspaces and model
state. They do not bound total process RSS. The replay reserves current/next
pools, stacking and pinned staging; CUDA completion waits occur at pool
boundaries.

FastLens `decode_workset_capacity_mib` bounds decode worksets. The 512 MiB
transient diagnostic includes split-source values and metadata, recording used
and allocated capacity separately. It reports observed peaks rather than
limiting all native allocations. Model/output allocations and process-wide
memory must be measured separately; Torch peaks omit non-Torch GPU allocations.

## Timing and cache conditions

The E2E timer starts before requesting the first preparation pool and ends after
final model/metric CUDA completion. It includes planning, reads, decode,
transforms, batch organization, H2D, model execution and metrics. Model loading,
source/content verification, storage accounting and cache preparation precede
timing. The initialization duration includes content verification; that substage
is not an additive duration. Input-only mode follows the same path through
tensor readiness and omits model execution.

Verification warms input files. Default replay leaves page-cache state
unmeasured. `--observe-cache` records page residency;
`--cold-cache` evicts input files with `POSIX_FADV_DONTNEED` before each pass and
requires zero resident pages under `mincore`. Reader/model metadata remains
initialized. Cache observations and decoded-cache counters are distinct.
Application read bytes are distinct from physical disk bytes; an unmeasured
physical byte count is null.

The existing `run_inference_pilot.sh PREPARED OUTPUT` launcher provides these
adapted protocols:

| Option | Excluded warmup | Measured window | Input cache |
| --- | --- | --- | --- |
| Default pilot | None | Dataset replay | Unmeasured |
| `--ic-protocol` | Complete passes totaling at least 60 seconds | Complete passes totaling at least 180 seconds | Observed after preflight/warmup, no per-pass eviction |
| `--fs-protocol` | None | Complete passes totaling at least 300 seconds | Verified cold files before every pass |

The IC timing is inspired by *The Image Calculator*, §5.1, pp. 13–14. The
[Frequency Shaping paper, §6.1](https://vldb.org/cidrdb/papers/2025/p27-sirin.pdf)
uses 256-image groups and repeated traversals over five minutes. These JPEG-DCT
protocols use source512 images and their own checkpoints/batch settings. The
per-pass cold-read control is an explicit experimental choice.

Passes may extend the target duration by one full traversal. Accuracy is taken
from the first measured pass; later passes must reproduce its top-1/top-5
counts. First-pass, per-pass and total times are reported separately. Three
fresh-process repeats rotate system order. IC and FS timing summaries remain
separate.

The launcher expects `PREPARED/contract/source.json` and the three stores named
in the run guide. `PYTHON` selects the environment and `CUDA_VISIBLE_DEVICES`
must contain the selected GPU UUID (obtain it with `nvidia-smi -L`). By default
it uses an 8194-image population; `--expected-images 50000` also checks 1000
classes. The native manifest and access sidecars must be prepared at the paths
used by the launcher. Dated continuation scripts depend on external run trees;
use the CLI for a new dataset or different paths.

The IC/FS protocol defaults use eight DataLoader processes, prefetch factor two,
256-image preparation pools and batch64. Prefetch counts pools, not model
batches. All methods keep the final tail, use the same source order and share
eight logical CPUs and the selected GPU. Each reference worker has a 1 GiB
reader/decode limit; the common retained-input bound is 32 GiB.
`--settings FILE.json` can set method-specific pools, workers, prefetch,
decode/reader/cache budgets and store roots while preserving logical requests
and model batch boundaries. Record those settings with each result.

The process monitor invalidates observations above a sampled 32 GiB process-tree
RSS sum or with prohibited GPU contention. RSS sums double-count shared pages
and exclude OS page cache; peaks between samples may be missed. MPS sharing
policy and process-lifetime NVML peaks are recorded separately from the timed
window. Use exclusive measured resources and record background I/O and RAM
availability. A baseline-only continuation is not a new matched FastLens run.

`--profile-stages` creates separate profiling runs with CPU stage durations and
CUDA events. Concurrent stages cannot be summed into E2E latency. Native
adapter H2D events measure labels; native data movement is in native counters.
Profiling runs and ordinary timings cannot share a summary.

Use `--resume-validation` only before timing has begun; `--resume-measurements`
retains valid completed repeats. `--reuse-verification PREVIOUS_RUN` starts a
fresh measurement directory from matching passed evidence. Changed contracts,
settings or identities require a fresh run; invalid or incomplete records stop
reuse. Original measurements and failure logs are retained.

## Training

The matched training protocol requires identical sample order, crop policy,
augmentation decisions, initial state and global resource limits. Existing
IC/FS adapters use CPU geometry plus shared GPU augmentation. Their timings
measure this split, rather than a fully CPU-augmented IC implementation or
storage alone.

`prepare --training` writes disjoint calibration and verification traces. Each
occurrence freezes crop/flip, RandAugment operations, component-specific
parameters, Mixup partners/weights and soft targets, including tails. Replay
samples no new augmentation decisions. CNN paths retain the complete
192-channel intermediate until augmentation and projection. Swin training uses
the 224-pixel/window-7 architecture; its 256-pixel inference checkpoint is
incompatible.

`snapshot` freezes model, AdamW, independent decay, scheduler and all RNG state.
Each backend/device needs its own `calibrate` artifact with explicitly chosen
input/model/state limits, followed by `verify-training` using that artifact.
The checks compare exact q/Q and targets, then inputs, logits, loss, gradients,
parameters, buffers and AdamW moments after updates. Both sides restore the
same snapshot. CPU calibration does not qualify CUDA verification.

Training reuses the published AdamW, clipping, warmup/cosine schedule and
sample-weighted accumulation. Accumulation closes at preparation-pool and epoch
boundaries without dropping tails; changing pool size can change effective
update batches. ViT uses FP32; CNN/eFUN/Swin use BF16 autocast. Correctness replay
uses eager execution and is separate from throughput and convergence runs.

CNN/eFUN integration uses `dct_models.training_pls --input-backend ic|fs
--baseline-store DIR`. For a common B6 crop/order condition across all routes,
use `python -m fastlens.benchmarks.dct_models.matched_training launch --help`.
Transformer integration uses `training_pls.run_matrix --execution-backend
ic-matched-dct|fs-matched-dct --baseline-store DIR` with the same mapping, model,
seed, condition and recipe. Stores must cover the full premixed population.

DataLoader adapters merge requests once per actual microbatch and preserve
order/duplicates. Counters include initialization and consumed batches;
unconsumed lookahead is outside their accounting. Matching declarations alone
cannot establish numerical equality or equal global resource use. Small
calibration runs do not establish full-epoch throughput or convergence.
