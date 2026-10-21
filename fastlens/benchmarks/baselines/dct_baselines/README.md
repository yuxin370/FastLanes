# IC / Frequency-Store DCT baselines

Implementation of the format/reference portion of
[the supplied plan](../../../../benchmark_results/archive/pre_fastlens_cleanup_20261021/fastlens/ImageCalulatorRe-implement/round-1/fastlens_agent_pack/IMPLEMENTATION_PLAN.md).
Matched methods are **paper-guided reimplementations with a JPEG-DCT adapter**.
Native mechanisms use separate APIs and storage identities.

## Current experiment scope (round 4)

The [final fairness revision](../../../../benchmark_results/archive/pre_fastlens_cleanup_20261021/fastlens/ImageCalulatorRe-implement/round-4/FastLens_Baseline_Fairness_and_Agent_Scope.md)
supersedes the earlier priority given to common GPU postprocessing. The primary
comparison is fixed-checkpoint inference with identical source content, samples,
input semantics and global resource limits. IC-style processing can remain on
CPU; FastLens retains its native CUDA/fused path. Different device assignments
are part of the system comparison, not an input-equivalence failure.

`verify-inference` uses the existing pretrained model loaders without resetting
weights, records the loaded state hash and model source files, and checks inputs,
logits, top-5 predictions and task metrics against the independent validation
transform. No RandAugment or Mixup runs in inference. Native CNN preparation
pools remain intact and use streamed projected ranges; Transformer outputs are
sliced at the model boundary. The IC/FS adapters finish geometry and normalization
on CPU before H2D. This describes the present adapters, not every execution
detail of the original FS system.

Existing training integrations use CPU geometry plus shared GPU augmentation.
Their results are **shared partial processing**, not the IC main CPU-augmentation
baseline and not a fully isolated format ablation. A training system comparison
would additionally require the CPU IC augmentation path. Common postprocessing
ablations and IC-native representation search remain optional. Historical A0/B6
results and previous-version training evidence retain their original scope.

Formal store startup now checks current JPEG hashes, including replacement at
the same path. Inference timing checks the loaded checkpoint state against its
gate, records source verification separately and labels cache state unmeasured.
The existing result schema/report retain method names and add experiment type,
execution partition and loaded model state; unclassified historical records are
not silently promoted to matched system results.

The native completion planner maps physical descriptors through `request_index`.
Split-source device values and metadata now have separate used/capacity peaks,
both included in the existing transient totals and 512 MiB diagnostic gate.
`decode_workset_capacity_mib` limits decode worksets, not the entire pipeline;
the diagnostic gate reports whether the measured total meets 512 MiB, and is
not an allocator admission limit. Model/output allocations and process-wide
resource limits still need their own deployment measurements.
The directed regression uses the canonical clamp containing zero. The review's
P3 custom-clamp/padding edge case is deferred and is not covered by this update.

Run the directed CUDA regression after rebuilding `_fastlens_direct_dct`:

```bash
export PYTHON=python3
export CUDA_VISIBLE_DEVICES=GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76
"$PYTHON" -B -m fastlens.torch.streamed_projection_test \
  --manifest fastlens/data/compressed/imagenet512_val_block_major/manifest.bin \
  --module-path build/fastlens/torch --output "$HOME/tmp/round4-streamed-projection.json"
```

Prepare **inference** contracts from the existing frozen source without
rebuilding the stores. Run each model in a fresh process to avoid upstream
top-level module-name collisions. The following checks both required branches
with batch64, preparation pool1024 and the retained tail. The inference
contract declares `atol=rtol=1e-5` before running, using the existing CNN logits
check in `dct_models/evaluate_shards.py`. Native Transformer inference preserves
the upstream operation order `-1+((q+1024)/2040)*2`, with separately rounded FP32
division, multiplication and subtraction in the existing finalization kernel.
The previous `(q+4)*float32(1/1020)` simplification passed the input tolerance but
failed the ViT-Ti logit gate on H100. The frozen tolerances are unchanged; top-1
predictions and top-5 class sets must agree exactly. Training normalization is
unchanged. A failure stops the run for diagnosis.

For a short check after rebuilding `_fastlens_direct_dct`, use
`python -B -m fastlens.benchmarks.baselines.dct_baselines.diagnose_fastlens --prepared RUN/vitti`,
where `RUN/vitti/contract/matched.json` is the existing inference contract. It
checks the first microbatch's source q/Q, inputs and pretrained logits exactly;
the full gate is still required afterward.

```bash
(
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
prepared="$HOME/tmp/dct-round3-prod-h100-20261010-012949"
run="$HOME/tmp/dct-round4-inference-$(date +%Y%m%d-%H%M%S)"
export FASTLENS_BLOCK_MAJOR_ACCESS_DIR="$PWD/fastlens/data/compressed/imagenet512_val_block_major/access"
for model in resnet24 vitti; do
  contract="$run/$model/contract"
  "$PYTHON" -B -m fastlens.benchmarks.baselines.dct_baselines prepare \
    --source-manifest "$prepared/contract/source.json" \
    --model "$model" --upstream "$HOME/RGB-no-more" \
    --batch-size 64 --pool-size 1024 --workers 4 --torch-threads 2 \
    --reader-budget-mib 2048 --host-input-budget-mib 4096 \
    --seed 11997733 --atol 1e-5 --rtol 1e-5 --output "$contract"
  for name in cpu_dct fastlens ic_shared ic_batched fs; do
    case "$name" in
      cpu_dct) backend=cpu_dct; extra=() ;;
      fastlens) backend=fastlens; extra=(--root fastlens/data/compressed/imagenet512_val_block_major/manifest.bin
                     --module-path build/fastlens/torch --source-workers 48) ;;
      ic_shared) backend=ic; extra=(--root "$prepared/ic_matched_shared") ;;
      ic_batched) backend=ic; extra=(--root "$prepared/ic_matched_batched") ;;
      fs) backend=fs; extra=(--root "$prepared/fs_matched_b8") ;;
    esac
    "$PYTHON" -B -m fastlens.benchmarks.baselines.dct_baselines verify-inference \
      --config "$contract/matched.json" --backend "$backend" "${extra[@]}" \
      --device cuda:0 --output "$run/$model/$name.verified.json" \
      2>&1 | tee "$run/$model/$name.log"
  done
done
echo "Passed: checkpoint inference gates. Results: $run"
)
```

These are correctness runs, not throughput measurements. `bench --run-kind
inference` now accepts `--backend cpu_dct|fastlens|ic|fs` and `--root` (the existing
`--store` spelling remains supported). IC/FS and JPEG use the same parallel CPU
materialization in the gate and timed replay; FastLens uses its native producer
and CUDA processing. Model loading, current-source verification and content
checks precede timing. The window includes the first pool, read/decode/transform,
H2D, model execution, metric collection and final GPU completion. Native metrics
are read only after that completion, without per-microbatch synchronization.

FastLens `verify-inference` still performs the complete source q/Q, input and
model-output checks. It also binds the selected compressed shards, metadata and
access sidecars to the same evidence using content hashes. Subsequent `bench`
repeats compare those hashes, the native binaries and the existing JPEG source
identities before timing; they do not repeat full source coefficient decoding.
Changed files fail before timing. Evidence produced before this content binding
must be regenerated once with `verify-inference`. This changes neither the
measured pipeline nor its warmup, cold-cache or repeat settings.

The historical **8194-image pilot** used the existing source and stores, both ResNet24 and
ViT-Ti, five systems and three fresh-process repeats with rotating system order.
All child processes inherit the same eight logical CPUs and selected GPU.
CPU adapters use four workers and two Torch threads; native producer defaults
remain intact under that CPU affinity. Model batch64/pool1024 keeps retained
CPU input arrays within the declared 4 GiB budget with a 2 GiB store reader and
512 MiB decode budget. These reader bounds exclude transform workspaces/model
state and do not impose a global RSS or native allocation limit. The existing
process monitor records RSS and NVML peaks, including non-Torch GPU allocations.
Its measurements cover process lifetime; only the benchmark JSONL contains E2E
throughput. Use an otherwise idle GPU for the run.

```bash
bash fastlens/benchmarks/baselines/dct_baselines/run_inference_pilot.sh \
  "$HOME/tmp/dct-round3-prod-h100-20261010-012949" \
  "$HOME/tmp/dct-round4-pilot-h100-$(date +%Y%m%d-%H%M%S)"
```

The script refreshes correctness evidence under the new pool/thread settings
before measuring. Old results remain intact; they cannot authorize timing a
different code version or contract. Output includes per-run JSONL/counters,
process observations and `summary.json` using the existing method names and
report. JPEG/native byte counts not yet measured are null, not zero. Native
pipeline metrics are kept in the counter detail; Torch GPU peaks alone do not
measure native memory. Source checks affect cache warmth, which is unverified.
These untuned subset runs retain `headline_eligible=false`; they do not by
themselves establish full-dataset speedups or a storage-format-only comparison.

If validation stops before any performance measurement, rerun the script with
the same prepared/output paths and append `--resume-validation`. It checks and
reuses unchanged successful gates, reruns missing gates, and retains earlier
failure logs. This does not retry failures automatically or resume partial
timing runs. A passing short diagnostic is not a substitute for a full gate.

### IC main protocol and FS timing control

The author PDF already in this repository is
[The Image Calculator, §5.1, pp. 13–14](../../../docs/related_work/SIGMOD-2024-The%20Image%20Calculator.pdf).
It specifies eight PyTorch DataLoader workers, prefetch factor two, approximately
one minute of warmup and three minutes of profiling. It tunes batch size per
model without specifying one universal value. The
[Frequency Shaping paper, §6.1](https://vldb.org/cidrdb/papers/2025/p27-sirin.pdf)
uses groups of 256 images and repeated dataset traversals over five minutes.

Use `--ic-protocol` for the **IC-inspired matched benchmark**, not an exact
reproduction of the original IC experiment. `--fs-protocol` is a separate
300-second cold-read control. By default both use two checkpoints, five systems,
8194 images and three fresh-process repeats; `--expected-images 50000` selects
the complete validation population and also requires 1000 classes:

```bash
bash fastlens/benchmarks/baselines/dct_baselines/run_inference_pilot.sh \
  "$HOME/tmp/dct-round3-prod-h100-20261010-012949" \
  "$HOME/tmp/dct-ic-protocol-h100-$(date +%Y%m%d-%H%M%S)" --ic-protocol

# Separate output directory and summary for the FS timing control.
bash fastlens/benchmarks/baselines/dct_baselines/run_inference_pilot.sh \
  "$HOME/tmp/dct-round3-prod-h100-20261010-012949" \
  "$HOME/tmp/dct-fs-protocol-h100-$(date +%Y%m%d-%H%M%S)" --fs-protocol
```

The reference CPU paths use eight DataLoader **processes**, each with one
sequential reader, prefetch factor two and no decoded cache. A reference
DataLoader task is one 256-image preparation pool, delivered as four batches of 64.
Thus prefetch counts pools, not model microbatches. All five systems traverse
the same source order, retaining the final partial batch (2 images for the pilot,
16 for 50K). This inference ordering
avoids repeatedly decoding the same stored group across shuffled pools; it is
an explicit matched-experiment choice, not an ordering claimed from the papers.
FastLens retains its native producer under the same eight-CPU affinity.

Each reference CPU worker has a 1 GiB reader/decode limit; the common host input bound
is 32 GiB across workers and queued/prepared/pinned tensors. These engineering
limits preserve bounded lifetimes without serializing different groups through
the old shared 512 MiB decode budget. They are not paper parameters or total RSS
limits. Per-method settings may reallocate these internal budgets. A separate
process monitor invalidates runs above a sampled 32 GiB process-tree RSS sum or
with foreign compute processes observed on the selected GPU. RSS sums count
shared pages more than once, and exclude the OS page cache. This is an observed
process-memory ceiling, not a cgroup limit or a bound on all machine memory.
All methods use the same machine, eight-CPU affinity and GPU; record available
RAM and background I/O conditions when interpreting page-cache residency.

IC warmup runs complete passes until at least 60 seconds, then discards its
time, samples and counters. Measurement runs complete passes until at least
180 seconds; the FS control measures at least 300 seconds with no separate
warmup. Durations can exceed these targets by one dataset pass. The clock starts
before requesting the first input and ends after the final model/metric CUDA
completion. Read, decode, transforms, transfers and model work are included.
Model initialization, source/content checks and cache preparation are excluded
for every system. The IC-inspired protocol observes input page residency after
preflight and warmup, then runs continuously without per-pass eviction. It does
not assume the complete dataset is cached. First measured pass time, every
pass time, total window time and replay counts are separate fields. Accuracy is
computed from the first measured pass only; subsequent passes must reproduce
its Top-1/Top-5 counts. The FS cold-read control evicts input files before each pass using
`POSIX_FADV_DONTNEED` and checks zero resident pages with `mincore`; incomplete
eviction fails the run. This implements a common cold-input condition without
globally clearing other jobs' caches. This explicitly chosen per-pass eviction
is an adapted control, not a claim to reproduce either paper's cache command.
Model/reader metadata remains initialized. Cache
observations are saved in the existing per-run counters file.

The benchmark also samples process-tree CPU, physical reads and RSS during the
measured window, including live DataLoader children. Short-lived child shutdown
work and memory peaks between samples may be missed. The external NVML/process
report still covers process lifetime. `--profile-stages` enables a separate
profiling run with NPZ read/inflate/CPU-transform durations and CUDA events for
model work and adapter H2D. Native decode/transform metrics remain in the native
pass records; the adapter H2D event measures only labels for native execution.
Concurrent stages cannot be summed to obtain E2E latency. Profiling runs are
identified in resolved parameters and cannot share a summary with normal runs.

This remains a **JPEG-DCT matched adaptation**: H100, 512-pixel JPEGs, these
checkpoints, the 8194-image subset and batch64 are retained, whereas IC's paper
scales images to 256×256 and tunes each model's batch size. These results cannot
be described as a full native IC/FS reproduction or proof of optimal batch
size. They retain `headline_eligible=false`. Keep the IC and FS summaries
separate. To resume an interrupted validation stage, supply the original
protocol option **and** `--resume-validation`; completed timing runs are not
resumed or overwritten by that option. Use `--resume-measurements` to retain
valid completed repeats and run missing ones. Reuse checks the numerical
evidence, contract, timing protocol, GPU/CPU allocation and process result;
incomplete or invalid records stop the run and are never silently overwritten.
`continue_pilot_4090.sh` resumes the interrupted October 12 pilot in a new
directory, linking its passed ResNet gates, CPU-DCT repeat and successful
FastLens retry. The original failure logs remain in their original directory.

Successful checks already persist `MODEL/SYSTEM.verified.json` with
`status: passed`. For **new performance measurements using existing checks**,
use `--reuse-verification PREVIOUS_RUN` with a new output directory and the
same source, model settings and GPU. This links the original contracts and
passed evidence, checks their identities, and runs fresh measurements. It
never falls back to numerical verification if an artifact is missing or stale.
No existing measurement or evidence is overwritten. To resume that new run
later, use `--resume-measurements` without `--reuse-verification`.

For the current 50K baseline run, after the requested gates have passed:

```bash
ROOT=$HOME/tmp/dct-50k-20261013-152452
bash fastlens/benchmarks/baselines/dct_baselines/run_inference_pilot.sh \
  "$ROOT" "$ROOT/h100-remeasure-$(date +%Y%m%d-%H%M%S)" \
  --ic-protocol --expected-images 50000 --settings "$ROOT/reader_settings.json" \
  --systems "ic_shared ic_batched fs" --reuse-verification "$ROOT/h100"
```

This measures only IC/IC-b/FS; existing CPU-DCT results are reused without new
verification or timing. FastLens results from its earlier protocol remain
separate. Numerical progress in subsequent launcher runs is
printed at intervals of at least 1024 images and at completion, for both
source q/Q and final-input checks. Errors and other diagnostic output remain
visible. This is a launcher-only change: existing implementation hashes and
numerical evidence are unchanged. Performance processes still check file and
configuration identities before timing; they do not repeat q/Q or reference
input/logit comparisons.

For the October 13 full-50K memory failure, use
`continue_50k_h100.sh PREPARED NEW_OUTPUT [PREVIOUS_RUN]`. It links existing
contracts, passed gates, reference caches and valid completed repeats from
`PREVIOUS_RUN` (default `PREPARED/h100`), leaving invalid/incomplete measurements
and their logs in the original directory. The launcher validates linked IC/FS
results before skipping them and runs only missing repeats. This also handles a later
GPU-contention interruption without repeating successful measurements. It
continues only IC/IC-b/FS for ResNet24 and ViT-Ti. Existing valid CPU-DCT results
are linked for reporting; missing CPU-DCT checks or repeats are not scheduled.
The failed IC-shared run peaked at 33.32 GiB summed process-tree RSS against
the unchanged 32 GiB cap; its decoded cache had zero hits on 150,000 measured
requests. Both IC readers now use 1024 MiB per worker for these measurements
instead of 2048 MiB (eight workers); FS remains at 512 MiB. This saves up to
8 GiB of retained IC arrays, but the process monitor still decides validity.

The launcher's repeatable `--measurement-cache-mib ic_shared=1024` (also
`ic_batched` or `fs`) forwards the existing benchmark `--cache-bytes` override.
It changes only measured cache capacity, preserving the verified contract and
numerical semantics. Actual capacity is recorded in each row's variant and
`decoded_cache_bytes_per_reader`; resume rejects a different capacity. Pass
the same overrides when resuming this continuation with `--resume-measurements`.
This capacity reduction addresses the observed resource limit and is not a
new throughput-tuned configuration. Remaining ViT-Ti checks use the shared
source/input/model verification cache described below.

`complete_workloads_h100.sh PREPARED NEW_OUTPUT PREVIOUS_RUN` waits until the
current baseline driver has exited and the selected GPU has remained idle for
60 seconds, then continues the two current models and serially runs
`mobilenet24`, `mobilenet32`, `resnet64`, `efun`, and `swinv2`. All seven models
share the output directory and verification cache. Existing valid repeats
are retained; only IC/IC-b/FS are scheduled. CPU-DCT is reused for reporting and
FastLens is excluded. The queue passes `--wait-for-gpu`: GPU
contention at a task boundary waits for 60 continuous idle seconds instead of
terminating the queue. Already-idle boundaries incur no extra delay.

With NVIDIA MPS, both the queue and measurement monitor query `get_server_list`
and `get_client_list SERVER_PID` through `nvidia-cuda-mps-control`. The server
itself does not block execution. Clients owned by another UID block; same-UID
MPS clients are allowed. Other non-MPS compute processes outside the benchmark
tree still block. A failed or unresolvable client query is a check failure,
never an idle result. No server is terminated and Compute Mode is unchanged.
The process JSON records this sharing policy and observed MPS server/client
PIDs; server-level GPU memory cannot be attributed to an individual experiment.
Interference during a measured repeat still invalidates that repeat.
A failed command stops the queue and keeps
its logs. MobileNet uses preparation pool128 across the three IC/FS paths because
its 112x112 inputs exceed the 32 GiB input bound at pool256 for IC-batched;
model microbatch64, eight workers and the common resource cap stay fixed.

New CPU-DCT/IC/FS checks use `python -m
fastlens.benchmarks.baselines.dct_baselines.verification` and share
`RUN/verification_cache/` across the selected workloads:

- A source q/Q pass is saved immediately, independently of the later model
  check. Passed gates from any existing workload in the run can supply this
  evidence, including when resuming only a later model.
- The independent input reference is generated once per distinct source,
  request trace, preprocessing, batch size and numerical environment. The cache
  holds shapes, dtypes, ordered image/occurrence IDs, labels and SHA-256 tensor
  fingerprints, not the full tensor dataset. Equivalent preprocessing can be
  shared across checkpoints; different normalization, geometry or labels cannot.
- Model results are computed once per checkpoint/workload. Existing full gates
  can supply them via `--model-evidence`, without re-running their model. A
  subsequent store with bit-identical input batches uses these results and does
  no reference JPEG decode, reference resize or model forward. Its own actual
  reader/preprocessing still traverses every image once.
- A differing batch is checked against a freshly computed independent input
  and both model outputs using the original declared tolerance. It never
  widens tolerances or approves an input merely because its shape matches.

Evidence records `verification_method`, direct model-forward counts and reused
input/model batch counts, so inferred equivalence is distinguishable from a
direct logit comparison. Only a complete successful gate binds a reusable input
or model cache. Changed/incomplete cache files fail, and failed numerical checks
do not produce a passed gate. Existing measured implementation hashes remain
unchanged; this verifier lives in a separate subpackage. FastLens retains its
existing verifier and is excluded from the current baseline-only continuation.

The direct verifier accepts `--config`, `--backend`, `--root`, `--device`,
`--cache-dir`, `--output`, and optional `--source-evidence`/`--model-evidence`.
The launcher selects the existing evidence automatically. Already-running
launchers keep their loaded verification commands; switch at a verification
boundary and use `--resume-measurements` to retain completed measurements.

### Revised P0 execution

1. Run the targeted regressions and the updated 8194 pilot first. The verified
   4090 native regression is independent of the Python timing/cache changes.
   `--models "resnet24 vitti"`, `--systems "cpu_dct fastlens ic_shared ic_batched fs"`
   and `--verify-only` can restrict the same launcher; no alternate model loader
   or numerical tolerance is introduced.
2. Run `run_reader_tuning.sh PREPARED NEW_OUTPUT` on CPU with otherwise idle
   input storage. It uses a separate seed (11997734) and both source/shuffle
   requests. Its 22 cases cover the three reference stores, IC-b group32/64,
   and individual worker4, pool512 and cache512 MiB changes to IC-b/FS. This is
   a bounded diagnostic, not a full Cartesian search. FS group256 remains the
   reference. Three measured passes follow one excluded full warmup pass.
   Raw JSON records include inflate counts, q volume, bytes, cache hits, CPU,
   RSS and observed average inflate concurrency. Reader-only throughput does
   not include geometry, transfers or a model. Confirm chosen candidates with
   `--input-only` and E2E measurements before freezing final settings.
   The October 13 reader results show that FS cache512 hits every request in
   source order, while IC-b cache512 hits none: the decoded 8194-image working
   sets are about 3.21 GiB and 12 GiB respectively. Run
   `confirm_reader_configs_4090.sh` for the bounded E2E follow-up: IC-shared
   and IC-b each compare cache0/cache2048 MiB per worker; FS compares
   cache0/cache512. All use eight workers, pool256, the same source-order
   tuning trace, two models and three rotating repeats (36 measurements).
   Reader budgets are 2560 MiB per worker for IC-shared, 2816 MiB for IC-b,
   and 1536 MiB for FS, identical within each cache/no-cache pair. IC-b needs
   room for its 2048 MiB cache reservation, about 294 MiB of gathered outputs
   and a complete 384 MiB NPZ member; 2560 MiB was insufficient. The common
   input and sampled process-tree RSS caps remain 32 GiB. The retry retains
   the two passed, unchanged ResNet IC-shared contracts from the interrupted
   confirmation run; both IC-b contracts use the corrected budget.
   Existing source q/Q evidence is reused, while final inputs/logits
   are checked for each new contract. This explicitly measures decoded-cache
   reuse; a fully cached 8194-image result must not be extrapolated to 50K.
3. The confirmation in
   `$HOME/tmp/dct-reader-confirm-4090-20261013-115959/summary.json`
   passed all 36 measurements and 12 numerical gates. Inputs and logits were
   exact; the largest sampled process-tree RSS sum was 22.71 GiB. Freeze
   `p0_reader_settings.json`: eight workers, pool256, prefetch2; IC-shared and
   IC-b cache2048 MiB per worker, FS cache512; reader budgets as above. Keep
   IC-shared group1 and IC-b/FS group256. CPU-DCT and FastLens retain the
   launcher's common protocol settings and their existing execution paths.
   The cached working set fits the 8194-image confirmation, but not full 50K;
   report actual full-population cache hits, including zero hits if observed.
   Do not combine these newer baseline measurements with older FastLens runs
   to claim a new matched speedup.
4. Build the complete 50K source and stores with the frozen configuration:

```bash
cd .
bash fastlens/benchmarks/baselines/dct_baselines/prepare_50k.sh
```

The CPU-only script freezes the existing JPEGs using eight processes, checks
50,000 unique images and 1,000 classes, and builds each store with eight
processes. Stores are built sequentially to avoid competing build measurements.
An optional first argument selects a new output directory; the default is
`$HOME/tmp/dct-50k-TIMESTAMP`. It saves each store's build time, CPU/RSS and
persistent bytes in `*.build.json`, keeps failure logs and copies the selected
reader settings to that directory. It does not regenerate JPEGs or change
quality/subsampling. The producer recipe retains its existing provenance scope:
byte reconstruction was checked on 8194 images, not on all 50K originals.
Full-source q/Q checks against each backend still precede inference timing.

On success it prints the exact H100 command using the new directory and
`--settings .../reader_settings.json --ic-protocol --expected-images 50000`.
Run that command with idle GPU/CPU/storage resources. It verifies and measures
both models across five systems with three repeats, including the 16-image
final model batch. This is a fresh run, with full-population numerical evidence.

`--settings FILE.json` supplies per-method physical settings: `pool_size`,
`workers`, `prefetch_factor`, `decode_budget_mib`, `reader_budget_mib`,
`decoded_cache_mib`, and `root` to select a rebuilt store. Logical
request order, batch64, FP32 and checkpoint stay common. Each method receives
the same complete inference request trace; physical pools must preserve model
batch boundaries and final tails. The reader cache budget is **per worker** and
must fit inside the per-reader budget; the combined input bound is checked.
Use `--request-order shuffle --seed 11997734` for separate tuning requests;
the main source-order run retains seed11997733. Resuming with changed settings
fails; use a fresh output directory instead.

The CPU tuner and formal GPU throughput runs must not compete for the same
CPUs/storage. Separate numerical tests on 4090 can run while preparing code,
but timing observations require exclusive measured resources. OOM and other
command failures retain logs and stop the launcher. The process report records
GPU contention or memory-budget failures as `valid=false`; the summary refuses
those observations. No process is terminated to free another job's resources.

First complete 50K timing and numerical results are still required before
publication. The existing conservative `headline_eligible=false` remains; a
passing script is not an automatic publication endorsement. Partial-DCT/zero
fill, IC-StaticK and common-processing ablations remain the later P1 stage.
The all64 source dependency check has not been weakened.

Run from the repository root with NumPy, Pillow, PyTorch and the existing
`dct_manip` coefficient extension. Model adapters additionally require the
existing RGB-no-more / DCTNet / FUN source trees and checkpoints. `report`
requires `jsonschema`. No downloads or native rebuild are needed for the
format implementation. The FastLens integer-source gate additionally uses
the existing `fastlens_dct_storage` target; rebuild it after this patch:
`cmake --build build --target fastlens_dct_storage -j2`.

```bash
export PYTHON=python3
"$PYTHON" -m unittest discover -s fastlens/benchmarks/baselines/dct_baselines/tests -t .
"$PYTHON" -m fastlens.benchmarks.baselines.dct_baselines --help
```

## Implemented scope

| Method | Physical organization |
| --- | --- |
| `ic_matched_shared` | One image per NPZ, int32 q and uint16 Q |
| `ic_matched_fixedk` | Same, retaining explicit profile source frequencies at every spatial position |
| `ic_matched_batched` | Default 256 images, one concatenated image-major q member plus Q |
| `fs_matched_b8` | Default group256, 15 independent diagonal NPZs, range-checked int8/int16, original JPEG q/Q |
| `fs_native` (Python API) | B32/Q50, 63 diagonals; independent native contract |

`dtype_policy=int16` is the common-width control. IC `safe` narrows over the
whole image/batch; FS `paper` and `safe` choose width over each complete
diagonal. Overflow raises an error. Store metadata records the policy and
each member's actual dtype. A failed build can leave a partial directory;
it has no usable `store.json` and is not silently resumed.

Offline `build` uses **eight processes by default**, independently of the
inference worker count in its input contract. Override with `--workers N`;
`--workers 1` runs serially. Each task extracts original JPEG q/Q, compresses
and writes one complete storage group. At most N groups are in flight, and
only source metadata crosses processes on the CLI path. Completed group
metadata is merged in source order. Compression, group numbering, dtype rules,
tail images and the content fingerprint remain identical to serial builds.
Conversion CPU time includes the parent and joined worker processes; model/GPU
execution is not involved. Worker errors propagate and no usable manifest is
published after a failed build. Existing output directories are never replaced.

```bash
"$PYTHON" -B -m fastlens.benchmarks.baselines.dct_baselines build \
  --config "$HOME/tmp/dct-round3-prod-h100-20261010-012949/contract/matched.json" \
  --method fs_matched_b8 --group-size 256 --workers 8 \
  --output "$HOME/tmp/fs-parallel-$(date +%Y%m%d-%H%M%S)"
```

The Python `build_store` API accepts either a frozen source manifest (so JPEG
extraction also runs inside workers) or the existing `SourceImage` iterator.
Its default remains `workers=1`; pass `workers=N` explicitly for parallel use.

`StoreReader.plan/read` accepts component-specific natural frequency indices
and rectangular block regions. It reads and inflates whole image/batch or
group/diagonal units, gathers afterwards, and restores request order,
including duplicates. Every NPZ has a saved SHA-256 and size; the store
fingerprint covers those identities and its layout. Verification checks all
payloads before measurement. Ordinary reader use also hashes each decoded
buffer; timed replay over frozen files omits that duplicate hash and redundant
ZIP metadata scan. NPZ CRC and array shape/dtype checks remain. Files must not
change during a run. Preflight is reported separately and warms the OS cache;
only the explicit cold-input protocols above evict and verify input pages
afterwards. Old stores without
payload identities and old contracts without frozen labels must be rebuilt.

Decoded caching has an explicit byte budget (default zero). Decode/scatter
futures release complete groups in bounded chunks instead of retaining all
misses. Defaults are 512 MiB active decoded arrays and 1 GiB per reader for
LRU, active arrays and gathered q/Q. CLI options `--decode-budget-mib`,
`--reader-budget-mib` and `--host-input-budget-mib` freeze these budgets.
The replay additionally reserves current/next pools, stacking and CUDA pinned
staging against its host input budget (default 4 GiB). These are retained-array
bounds, not total RSS limits: manifest/index objects, NumPy inflate/transform
workspace and model/optimizer memory are excluded. The original pilot uses
threaded NPZ decode/CPU transforms and one prefetched pool; the paper protocol
uses the DataLoader settings and per-worker limits described above. CUDA
submission is bounded to two preparation
pools, with completion waits at pool boundaries. There are no per-coefficient
Python decode loops.

Storage supports grayscale and ragged component grids. The seven model
adapters (`mobilenet24`, `mobilenet32`, `resnet24`, `resnet64`, `efun`,
`vitti`, `swinv2`) consume the existing source512 4:2:0 geometry. CNN/eFUN
outputs remain CHW; Transformer outputs remain separate Y and Cb/Cr grids.
Every resize profile explicitly requires all 64 source frequencies, so its
FixedK store currently retains all64. Output channel count never determines
source K.

## Freeze, build, verify and replay

The supplied draft YAML is a specification, not a runnable config. `prepare`
resolves a compact versioned `matched.json`, `source.json`, `profile.json`
and occurrence-level `trace.jsonl` from actual repository inputs. It hashes
source JPEGs and q, records component IDs/sampling/valid geometry/table IDs,
and freezes checkpoint, numerical operation order, output tensors and label
mapping. Existing Q tables are extracted directly through libjpeg. There
is no RGB round trip, Q50 requantization or baked online crop in a matched
store.

Provide `source-recipe.json` recording the actual common source preprocessing
(resize, encoder/version/settings, quality, subsampling and provenance).
Do not infer that recipe from JPEG q/Q or substitute the model's output
geometry. The local validation source's complete historical production
recipe still needs provenance confirmation before formal experiments.

```bash
run="$HOME/tmp/ic-fs-resnet24"
"$PYTHON" -m fastlens.benchmarks.baselines.dct_baselines prepare \
  --selection fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json \
  --source-recipe /path/to/source-recipe.json \
  --model resnet24 --upstream "$HOME/RGB-no-more" \
  --count 16 --batch-size 4 --pool-size 16 --workers 8 --torch-threads 1 \
  --seed 11997733 --output "$run/contract"

for method in ic_matched_shared ic_matched_fixedk ic_matched_batched fs_matched_b8; do
  "$PYTHON" -m fastlens.benchmarks.baselines.dct_baselines build \
    --config "$run/contract/matched.json" --method "$method" --output "$run/$method"
  "$PYTHON" -m fastlens.benchmarks.baselines.dct_baselines verify \
    --config "$run/contract/matched.json" --store "$run/$method" \
    --device cpu --check-model --training-steps 10 --output "$run/$method.verify.json"
  "$PYTHON" -m fastlens.benchmarks.baselines.dct_baselines bench \
    --config "$run/contract/matched.json" --store "$run/$method" \
    --evidence "$run/$method.verify.json" --device cpu --run-kind inference \
    --repeats 3 --output "$run/$method.results.jsonl"
done
"$PYTHON" -m fastlens.benchmarks.baselines.dct_baselines report \
  "$run"/*.results.jsonl --output "$run/summary.json"
```

Use a fresh directory for each `prepare/build`; results files also refuse
overwrite. Omit `--count` to import the complete selection. Batch/pool/worker
settings are caller choices, not claims about the papers' model batches.
Run different DCTNet model profiles in separate processes, as required by
the existing upstream imports. To check CUDA after driver recovery, replace
`--device cpu` with `--device cuda` in both `verify` and `bench` and use new
evidence/result paths. No GPU timing was validated in this implementation
session.

`verify` checks all stored q and Q against fresh independent JPEG extraction,
then all trace tensors against the existing CPU validation reference; it
optionally compares checkpoint logits and complete backward/SGD updates.
Tolerance is frozen during `prepare` (exact by default). Nonzero tolerances
require independent calibration; they are never relaxed on failure.
The CLI's SGD update check is a semantic smoke test, not the published
optimizer/scheduler recipe. Failed verification emits no passing evidence.
The benchmark rejects stale contracts, stores, implementation hashes,
devices or missing inference-logit verification.

The inference checkpoint command retains the existing validation crop.
The training commands below consume explicit crop/flip, RandAugment,
Mixup partners/weights and soft targets. Both retain all source frequencies
needed by the spectral resize; neither substitutes output channel count for K.

## Common training trace and numerical gates

`prepare --training` writes separate calibration and verification traces
with disjoint image identities. It freezes actual crop/flip and both
RandAugment operations, including component-specific Cutout centers/pads,
translation distances and ChromaDrop channel. Mixup stores both FP32 beta
weights and the partner occurrence within the actual microbatch, including
one-image tails. Repeated images remain distinct occurrences. No augmentation
RNG is sampled during replay.

`UnifiedReplay` connects `cpu_dct`, `fastlens`, `ic` and `fs` to those same
rows. CPU-DCT extracts q with JPEG2DCT and applies literal upstream geometry;
IC/FS use the original-q store adapter; FastLens uses the public native
pipeline with explicit crop/flip descriptors. CNN paths retain the complete
192-channel intermediate for augmentation before model projection. CNN
spatial augmentation scales the canonical 28-block parameters to its actual
grid; Transformer chroma uses half the luma grid. Training Swin uses the
existing **224px training architecture**, rather than the 256px inference
checkpoint architecture.

All paths share frozen augmentation and target decisions. The independent
oracle uses upstream scalar augmentation and separate channel normalization,
Mixup and soft-target construction; the replay uses existing grouped operations.
The scalar-per-channel oracle uses device FP32 mean/std tensors, matching the
declared tensor division; Python scalar divisors use a different CUDA reciprocal
optimization. A host CUDA regression covers this observed failure without
relaxing numerical tolerances.
Training profiles record geometry, integer round/clamp, RandAugment with
per-operation clamp, model normalization and Mixup in their execution order.
Hashed labels and one batch/epoch/pool validator cover both traces.
The existing AdamW, independent decay, warmup/cosine update
scheduler, gradient clipping and sample-weighted accumulation are reused.
Accumulation closes at preparation-pool and epoch boundaries without dropping
tails. The default 300-epoch schedule uses that same boundary rule over the
declared full population, rather than the short correctness trace; explicit
`--total-updates` overrides are recorded. Changing pool size can change effective
update batches and is not automatically a matched input-only comparison.
ViT uses FP32; CNN/eFUN/Swin use the existing BF16 autocast recipe.
These correctness runs use eager model execution, not performance timings.

`snapshot` records the from-scratch initial model, AdamW state, decay state,
scheduler and Python/NumPy/Torch/CUDA RNG. Its hash is shared by all methods.
The profile's pretrained checkpoint remains an inference-reference identity;
it does not replace this complete training snapshot.

Example from the repository root (supply the actual source production recipe):

```bash
run="$HOME/tmp/ic-fs-training-resnet24"
"$PYTHON" -m fastlens.benchmarks.baselines.dct_baselines prepare \
  --selection fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json \
  --source-recipe /path/to/source-recipe.json \
  --model resnet24 --upstream "$HOME/RGB-no-more" \
  --count 17 --batch-size 64 --pool-size 4096 --workers 8 --torch-threads 1 \
  --seed 11997733 --training --epochs 10 --calibration-count 4 \
  --output "$run/contract"
"$PYTHON" -m fastlens.benchmarks.baselines.dct_baselines snapshot \
  --config "$run/contract/matched.json" --output "$run/initial.pt"
"$PYTHON" -m fastlens.benchmarks.baselines.dct_baselines build \
  --config "$run/contract/matched.json" --method fs_matched_b8 --output "$run/fs"

"$PYTHON" -m fastlens.benchmarks.baselines.dct_baselines calibrate \
  --config "$run/contract/matched.json" --snapshot "$run/initial.pt" \
  --backend fs --root "$run/fs" --device cpu --updates 10 \
  --input-atol 0 --model-atol 0 --state-atol 0 --rtol 0 \
  --output "$run/fs.cpu.calibration.json"
"$PYTHON" -m fastlens.benchmarks.baselines.dct_baselines verify-training \
  --config "$run/contract/matched.json" --snapshot "$run/initial.pt" \
  --backend fs --root "$run/fs" --device cpu --updates 10 \
  --calibration "$run/fs.cpu.calibration.json" --output "$run/fs.cpu.verify.json"
```

Repeat with `--backend ic --root /path/to/ic-store`, or `--backend cpu_dct`
without a root. Each backend has its own calibration artifact. For FastLens
the source selection must contain the actual `fastlens_image_id` mapping. The
integer gate uses `read-source-manifest`, which reads the canonical block-major
source through the existing CPU dataset reader and restores natural frequency
and raster order without invoking the replay planner.

On the user's H100, set
`CUDA_VISIBLE_DEVICES=GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76` and use
`--device cuda:0`. FastLens also needs
`--backend fastlens --root fastlens/data/compressed/imagenet512_val_block_major/manifest.bin
--module-path build/fastlens/torch`.
Calibrate on that device before verifying; CPU calibration cannot authorize
GPU verification. Nonzero input/model/state limits must be explicitly chosen
and supported by the independent golden measurements. Spectral resize can
straddle integer rounding ties, so CNN tolerances must account for the actual
channel normalization. Limits are never increased after a verification failure.

G1 checks selected q/Q exactly against fresh JPEG extraction. G2/G3 compare
final inputs, exact soft targets, logits, loss, accumulated gradients, model
parameters/buffers and AdamW moment tensors after each update. Both executions
restore the same complete snapshot and RNG; decay, scheduler and resulting RNG
states must also match. Calibration and verification reject changed source,
profile, trace, checkpoint, implementation, snapshot, native binding or device
identities. Passing evidence records observed stage errors and the frozen
limits; it remains `headline_eligible=false`.

## Existing experiment integration

This package owns storage and numerical gates. Formal training uses the existing
experiment runners, models, schedules, checkpoints and result schemas; it does
not introduce a separate training suite. Both adapters replace coefficient
acquisition and preserve their established JPEG preprocessing and labels.

CNN/eFUN: `dct_models.training_pls --input-backend ic|fs --baseline-store DIR`
uses the existing JPEG training adapter and shared native GPU augmentation.
The store must cover the complete premixed mapping with identical logical IDs,
labels and source paths, retaining all64 frequencies. Native B6 versus default
IC/FS A0 retains the existing crop/order comparison; select the same condition
when making a strictly matched storage comparison. Store variant, source hash,
GPU identity and optional `--baseline-evidence` are saved in `training.json`.

```bash
export CUDA_VISIBLE_DEVICES=GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76
export CNN_IC_STORE=/path/to/full-train-ic-store
export CNN_FS_STORE=/path/to/full-train-fs-store
bash fastlens/benchmarks/dct_models/cnn_training_baselines.sh plan ic-fs-training \
  'mobilenet24:ic mobilenet24:fs resnet24:ic resnet24:fs efun:ic efun:fs'
# Change plan to probe/full/data/profile to use the established run modes.
"$PYTHON" -m fastlens.benchmarks.dct_models.report_cnn_suite \
  --output-dir "$HOME/tmp/cnn-with-ic-fs" \
  --additional-training /path/to/ic-run /path/to/fs-run
```

ViT-Ti/Swin: keep the existing `training_pls.run_matrix` arguments and change
`--execution-backend` to `ic-matched-dct` or `fs-matched-dct`, adding
`--baseline-store DIR`. Keep the existing train/val manifests, layout plan,
premixed mapping/hash, registered model, seed, condition and precision. The
original immutable run manifest now records the store identity. Epoch metrics,
validation, optimizer/accumulation and checkpoint/resume remain in the existing
runner. `report_equal_image_performance --dct-reference-run DIR` accepts a
JPEG/IC/FS reference and emits the same CSV/JSON, using the concrete store
variant as the row label. `report_convergence_reference --reference-run DIR`
also accepts these references. Existing paired source/model/recipe/GPU checks
still apply; archived runs with different identities cannot establish a matched
speedup merely by appending rows.

Both formal DataLoader adapters merge spatial/frequency requests with
`__getitems__` once per actual model microbatch and use the same bounded
decode/scatter reader, with zero decoded cache. They preserve duplicate/order
semantics and charge group reads once. Raw CNN records and Transformer epoch
records save actual read/decode counters and memory peaks from consumed
microbatches, including worker initialization. Unconsumed DataLoader lookahead
is not measured; this remains a G4 limitation. These integrations stay
`headline_eligible=false`. GPU calibration, measurement validation and complete
epoch measurements remain required before using them in a speed headline.

## Native sanity API

```python
import numpy as np
from fastlens.benchmarks.baselines.dct_baselines.native import (
    NativeSpec, encode_ic, fs_source, mosaic, write_ic_archive, read_ic_archive,
)
from fastlens.benchmarks.baselines.dct_baselines.store import build_store, StoreReader
from fastlens.benchmarks.baselines.dct_baselines.contracts import ReadRequest

spec = NativeSpec(block_size=32, k=3, scalar_quantization=50)
expected = encode_ic(rgb_uint8_256x256, spec)  # int32 [3,24,24]
write_ic_archive("image.npz", [rgb_uint8_256x256], spec)
write_ic_archive("batch.npz", rgb_images, spec)  # explicit IC-b concatenation
build_store((fs_source(rgb, i) for i, rgb in enumerate(rgb_images)),
            "fs_native_store", "fs_native")
with StoreReader("fs_native_store") as reader:
    requests = [ReadRequest("0", (tuple(range(1024)),) * 3)]
    views, counters = reader.read(reader.plan(requests))
    actual = mosaic(np.stack(views[0].dense()), 3)
```

All native choices are explicit in `NativeSpec.contract()`. `candidates()`
enumerates 6048 candidates; it does not train or implement the self-design
optimizer. `fit_statistics(..., split="train")` computes per-plane population
statistics including structural zeros, and rejects validation/test data
and zero variance. Save the statistics as a shared hashed artifact before
using `normalize_mosaic`. Native inference never multiplies scalar Q back.

## Measurement and current limits

`bench` times physical planning, complete reads/inflate/gather, CPU transform,
batch organization, pinned H2D, optional inference, metric collection and
last CUDA completion. Initialization is outside the window and its manifest
read bytes are separately reported. No first measured pool is prepared
before the timer. No excluded warmup mutates model state. H2D includes the
final float32 DCT tensors and labels; this is explicitly a CPU transform
adapter. Input-only replay uses the same full path through tensor readiness.

Inference replay supports FP32 only and rejects a BF16 contract. Model requests
now include the frozen crop's component block regions; requested q counts and
full-grid q counts are distinct, while complete touched members are still decoded.
CPU core-seconds and wall time use the same window. Content verification,
reader initialization and storage accounting are outside that window.
The detailed initialization duration includes content verification;
`content_verification` reports that substage and is not an additive extra.

Each raw row follows the local `result.schema.json` (`0.2-diagnostic`); `report`
validates it and groups matching contracts/runtime settings, implementation,
store content and binding fingerprints. Summaries preserve model/profile,
device, config/checkpoint identities, batch/pool/precision and raw run references.
Adjacent
counter JSONs record logical requests, per-pool deduplicated q demand,
actual decoded q/bytes, file/read counts, decoded-cache hits/evictions and
decode amplification. Complete archive bytes include ZIP/NPY headers and
metadata. Physical disk bytes are **null**, not inferred from application
reads. Logical and allocated storage bytes include labels, tables, geometry
and manifests. Conversion includes JPEG coefficient extraction and store
encoding, excluding creation of the common resized/reencoded JPEG source.
Separate FixedK stores are separate copies and must be summed in a family
storage comparison.

All current timings are diagnostic, `headline_eligible=false`; OS cache
warmth is unverified. The common training replay and published-recipe numerical
gates are implemented. In this session MobileNet24, ViT-Ti and ResNet24 each
passed ten paired CPU updates on existing synthetic JPEGs, with zero observed
logit/loss/gradient/parameter/AdamW differences; all IC variants, FS and CPU-DCT
matched those source tensors and targets. Three real ImageNet validation images
also passed FastLens integer q/Q/geometry checks. These are bounded correctness
checks, not full ImageNet training. On the user's H100, all nine combinations
of those three architectures with IC / IC-b / FS also passed ten calibration
and ten verification updates with zero input/model/state tolerances and zero
observed errors. CNN uses BF16 autocast and ViT uses FP32. Raw host JSON/logs
are archived under `round-2-response/evidence-h100`; implementation, store,
snapshot and calibration identities have been checked. FastLens CUDA on the
same real source/trace remains outstanding. The sandbox itself has no CUDA.
Full temporary-file tests also require the host's writable `$HOME/tmp`.
Equal-budget multi-consumer runs, cache/resource calibration, complete
50K/epoch experiments and the second GPU remain unperformed. This implementation
does not mark the full cross-backend CUDA G3 gate, G4/G5 or the entire research
plan complete.
See [integration map and assumptions](ASSUMPTIONS.md).
The round-2 fixes, full integration patch and raw CPU/H100 evidence are
in [round-2-response](../../../../benchmark_results/archive/pre_fastlens_cleanup_20261021/fastlens/ImageCalulatorRe-implement/round-2-response/README.md).

### Round 3: matched entry and native search preparation

The review's A0/B6 mismatch, unverified cache labels and cache-hit admission
failure are valid. The production accumulation, FastLens CUDA, worker lifetime
I/O, global memory and native optimization requirements are additional work;
the existing small H100 gates do not satisfy them. Native representation search
and matched storage comparison answer different questions and retain separate
contracts.

For new CNN storage comparisons use the existing training driver through this
launcher. It sends **B6 to all five routes**, preserving the historical A0/B6
launcher and its archived results. Stores must cover the entire same population
and original JPEG coefficients. Paths below are placeholders for those stores;
the tiny round-2 stores cannot be used with the full training mapping.

```bash
export PYTHON=python3
export CUDA_VISIBLE_DEVICES=GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76
"$PYTHON" -B -m fastlens.benchmarks.dct_models.matched_training launch \
  --profile resnet24 --manifest /path/to/dct/manifest.bin \
  --mapping /path/to/ordered_mapping.csv --mapping-sha256 ACTUAL_MAPPING_SHA256 \
  --ic-store /path/to/ic_shared --ic-batched-store /path/to/ic_batched \
  --fs-store /path/to/fs --output "$HOME/tmp/dct-matched-b6" --plan
```

Remove `--plan` to run; `--max-pools 2 --validation-count 1000` makes a bounded
probe. Each directory uses the existing `training.json` and `epoch_*.json`
format, and can be appended with `report_cnn_suite --additional-training`.
To compute a diagnostic paired speedup, run:

```bash
"$PYTHON" -B -m fastlens.benchmarks.dct_models.matched_training check \
  "$HOME/tmp/dct-matched-b6/native/training.json" \
  "$HOME/tmp/dct-matched-b6/fs/training.json"
```

This check requires matching source/mapping, seed, crop/order, model profile,
initial weights, recipe and augmentation code, precision, runtime/GPU, compile
settings, pool/accumulation, and actual consumed epoch order/pool/update counts.
It supports fresh runs; resumed optimizer states are outside this contract.
Legacy results remain reportable but cannot provide this newly required identity.
Matching declarations and schedules do not replace real q/Q and GPU tensor
gates or an equal global CPU/memory budget. `headline_eligible` stays false.
After preflight the cache state is explicitly unmeasured; epoch 2 is a repeat,
not evidence that all files are resident. Existing CSV metric keys are retained.

`native_search prepare` implements **N1 only**: it freezes the search task,
candidate list and isolated train-fit/search-validation/final-evaluation IDs.
N2 training/profiling, N3 measured selection and N4 final deployment are not yet
implemented. It reuses `NativeSpec` and its explicitly chosen codec semantics;
it does not reinterpret a DCT24 checkpoint as a native model.

Create a task JSON using this example structure. Values describe a proposed
experiment and must be set to the actual source, resource and training choices
before freezing; preparation does not execute GPU work. Manifest paths are
resolved relative to the task file. Manifests use the existing `samples` entries
(`logical_sample_id`, `label`, `path`). Describe the actual JPEG producer in
`source_recipe`; unknown historical settings must not be guessed.

```json
{
  "train_manifest": "/path/to/train.json",
  "final_manifest": "/path/to/val.json",
  "search_val_count": 50000,
  "split_seed": 20261009,
  "model": {
    "family": "resnet50", "classes": 1000,
    "stem": "3x3 stride 1, padding 1, no maxpool; adaptive average pooling",
    "small_input_policy": "keep native size; training batch >=2 for BatchNorm"
  },
  "rgb_preprocessing": {
    "source_recipe": {"producer_script": "ACTUAL_SCRIPT", "parameters": "ACTUAL_PARAMETERS"},
    "train": {"crop": "RandomResizedCrop 256, scale [0.08,1], ratio [0.75,1.3333333333333333]", "flip_probability": 0.5},
    "evaluation": {"resize_short_side": 256, "center_crop": 256}
  },
  "hardware": {
    "gpu_uuid": "GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76", "gpu_name": "NVIDIA H100 80GB HBM3",
    "cpu_threads": 16, "host_memory_bytes": 17179869184, "cuda_memory_bytes": 68719476736
  },
  "objective": {
    "metric": "min_e2e_latency_ms", "accuracy_unit": "percentage_points",
    "accuracy_delta_pp": 0.5, "reference_candidate_id": "B008_k008_s1x1_q050"
  },
  "budgets": {"search_gpu_hours": 240, "latency_limit_ms": null, "storage_limit_bytes": null},
  "training": {
    "scratch_epochs": 210, "transfer_epochs": 20, "seed": 11997733,
    "optimizer": {"type": "SGD", "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0001, "scheduler": "cosine per fit"},
    "transfer_state": "copy model weights/BN; reset optimizer and scheduler; reset RNG to declared fit seed"
  },
  "final_training": {"epochs": 40, "confirmation_seeds": [11997733,11997734]},
  "runtime_candidates": {"batch_sizes": [64], "data_workers": [4,8,16], "decode_workers": [1], "cache_bytes": [0,536870912]}
}
```

```bash
"$PYTHON" -B -m fastlens.benchmarks.baselines.dct_baselines.native_search prepare \
  --task "$HOME/tmp/native-search-task.json" --output "$HOME/tmp/ic-native-search"
```

Outputs are `search_contract.json`, `candidate_manifest.json`, `splits.json` and
their hash references. The reference candidate's accuracy must later be measured
on search validation; no reference accuracy or feasible representation is invented.
The project domain `ic_native_subset96_v1` fixes no sampling and scalar q50,
includes both k endpoints of all six B values, and draws another 84 from the
lexicographic remaining list with NumPy PCG64 seed 20261009. The saved 96 specs
are authoritative; this is not the paper's unpublished exact sample. A 16-bucket
plan uses measured bytes/image, fixed tie-breaks and key indices 0,3,6,9,12,15;
all six candidates per key bucket require training. Predicted bucket accuracy
cannot satisfy an individual candidate's final accuracy constraint.

Splits use a fixed global permutation of sorted training IDs. Normalization and
weights use train-fit, selection/tuning uses search-validation, and the final
manifest is reserved for evaluation after selection. Preparation checks IDs and
paths for overlap and reads metadata only. Source manifest hashes do not prove
current image bytes; actual source content must be bound during encode/fit.
This step produces no accuracy, latency or optimized deployment claim.
Archived round-2 CPU/H100 evidence remains tied to its original implementation
hashes. The changed reader requires fresh numerical evidence before a new
production run can cite it; historical evidence is not rewritten.

### Real source512 producer and production numerical fixture

The train producer is
`system_rgbnomore/training/prepare_imagenet512_v3_train.py:275–283`:
decode to RGB, stretch to 512×512 with Pillow BILINEAR, then
`save(..., format="JPEG")` with default encoder options. The actual log
`fastlens/data/system_rgbnomore/e2e_v3/prepare_imagenet512_v3_train.full.log`
records phase `all`, 32 resize workers and 1,281,167 converted / zero reused
images. The archived `imagenet512_train_compact_v3/artifacts/plan.json` and
`COMPLETE.json` record the same semantic recipe and Pillow 12.2.0 runtime.
No separate historical validation-generation command was recovered.
The upstream `RGB-no-more/utils/mp_scripts.py:37–46` implements the same
RGB/BILINEAR/default-JPEG flow, called with size512 for both splits by
`utils/pipeline_utils.py:246–251`.

Fifty real validation images distributed across the full manifest were
reconstructed from the original `e2e_v2/imagenet/val` JPEGs in memory:
all reconstructed bytes equal the current source512 files. Explicit quality75,
subsampling2 also gives identical bytes; actual SOF headers are 4:2:0.
Same-recipe reconstruction of three train images (beginning, middle and end
of the manifest) from their original class TAR members is also byte-exact.
These checks establish byte equivalence for the checked images, not recovery of a
historical command or a full-population verification. BICUBIC and LANCZOS
failed the initial three-image reconstruction check.

The next numerical fixture uses 8,194 real validation images and checks this
producer **for every selected image** before source extraction. It binds both
original/source JPEG hashes and then verifies source hashes again against the
frozen q/Q manifest. It neither regenerates existing JPEGs nor replaces the
compressed FastLens source.

Each disjoint calibration/verification trace has 4,097 images: four whole
physical 1,024-image segments in a pool, plus one tail image. Source positions
0–4095 and 4096–8191 supply the two full pools; positions8192/8193 supply their
tails. Crop keys use actual source position //1024, per-PLS; flip/RandAugment/
Mixup use the existing keyed helpers. Batch64, accumulation16 and group256
yield five updates per trace: four 1,024-sample updates with 16 microbatches
each, and one single-sample tail update. The published 300-epoch training
schedule is explicitly 375,600 updates (1,252×300). The trace is one finite
numerical fixture, not a full training epoch or an accuracy experiment.

Run on the host H100, from the repository root:

```bash
export PYTHON=python3
export CUDA_VISIBLE_DEVICES=GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -o pipefail
RUN_ROOT="$HOME/tmp/dct-round3-prod-h100-$(date +%Y%m%d-%H%M%S)"
"$PYTHON" -B -m fastlens.benchmarks.baselines.dct_baselines.production_gate \
  --model vitti --output "$RUN_ROOT" 2>&1 | tee "$RUN_ROOT.driver.log"
```

The command performs producer checks, prepare/build of three stores, a complete
initial-state snapshot, and calibration/verification for CPU-DCT, FastLens,
IC, IC-b and FS. ViT-Ti uses FP32; `--model resnet24` uses BF16 autocast.
All tolerances start at zero and a failure stops the run. Per-stage raw logs,
calibration and verification JSON are preserved in the new output directory.
This includes several thousand JPEG reconstructions and store conversions;
it is substantially larger than the round-2 fixture. Success ends with
`Passed: ... 5 backends, 5 updates each ...`. On failure inspect the last named
stage's `.log` and retain the original failure; do not relax tolerances.

An input mismatch includes `Input failure diagnosis` in the stage log: the
microbatch ID, exact raw/augmented coefficient mismatch counts, and up to eight
affected images with coordinates and CPU FP32 values before rounding (after
flip). These use the same captured raw tensors that produced the failed inputs.
The native pre-rounding accumulator is not exposed. Diagnostic output does not
authorize a tolerance change or mark the gate passed.

For the observed first-batch FastLens failure, the seven raw chroma differences
are reproduced by resizing the top/bottom source blocks separately and adding
their FP32 results. A complete-source calculation with the same conversion
matrix matches the CPU values. The CUDA down2 path now retains source values
for outputs spanning decode worksets and transforms them once at their last
owner. Rebuild `_fastlens_direct_dct` before checking this change. The focused check
below verifies q/Q for the original 64-image batch, then checks raw coefficients,
augmentation/Mixup inputs and targets with zero tolerance:

```bash
"$PYTHON" -B -m fastlens.benchmarks.baselines.dct_baselines.diagnose_fastlens \
  --prepared "$HOME/tmp/dct-round3-prod-h100-20261010-012949"
```

This input-only check does not replace the five-backend production gate below.

FastLens calibration and verification both receive the native extension directory
`build/fastlens/torch` through `--module-path`. The production launcher uses
`--source-workers 48` for FastLens G1 checks. Each worker checks a contiguous
range of native image IDs with its own CPU bridge process; the log reports
completed source counts. Direct `calibrate`/`verify-training` commands default
to one source worker and accept the same flag. Source parallelism preserves
exact q/Q and geometry checks; GPU batch order and training arithmetic retain
their existing execution. The native per-image reader still materializes
rowgroups independently, so this change does not remove repeated group decoding.
After a failure following store build
and snapshot creation, reuse those inputs with `--prepared` and write results
to a new directory:

```bash
PREPARED="$HOME/tmp/dct-round3-prod-h100-20261010-012949"
RUN_ROOT="$HOME/tmp/dct-round3-prod-h100-retry-$(date +%Y%m%d-%H%M%S)"
"$PYTHON" -B -m fastlens.benchmarks.baselines.dct_baselines.production_gate \
  --model vitti --prepared "$PREPARED" --output "$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT.driver.log"
```

This reuses the frozen contract, three stores and untouched initial snapshot.
All five backends are calibrated and verified again with the current code;
the failed run's logs and results remain in their original directory.

The FastLens branch uses the public explicit-transform CUDA pipeline with the
existing source manifest and its `access` sidecars. Formal multiworker training,
worker lifetime/unused-lookahead I/O, equal global resource budgets and tuned
performance are still separate M2 requirements. No G4 or headline qualification
is granted by this fixture. Validation images used here must not initialize a
later quality/search model; only the untouched initial snapshot is exported.
