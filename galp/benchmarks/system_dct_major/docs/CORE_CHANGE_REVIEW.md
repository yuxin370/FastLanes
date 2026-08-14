# Core-change review record

> **Historical implementation review.** This records the boundary approved
> during block-major development. Current runtime selection is frozen in
> `block-major-p4-scheduled-bounded-110-v1`; use [README](../README.md) and
> [RUN_GUIDE](RUN_GUIDE.md) for the supported interface.

## Approved implementation boundary

The earlier benchmark-only audit established that manifest-v1 DCT-major
segments expanded millions of host transform items and repeatedly scanned all
logical output blocks for every resident workset. The requested implementation
now includes core changes under `galp/include`, `galp/src/jpeg`, the Torch
binding, tools and tests. Repository-root `src/` remains untouched by this
feature.

## Implemented invariants

- Existing manifest-v1 `.fls`, metadata and coefficient payload bytes are not
  rewritten. Block-major access remains an additive, CRC-bound sidecar format.
- Reader construction validates the lightweight companion index but maps and
  source-validates a shard descriptor only on first touch. Metadata/index state
  is likewise per-shard lazy for planless manifest-v1 operation.
- Compact planning retains request/output order and duplicate fan-out while
  creating zero fixed-transform items, output source lists or global transform
  sort items.
- The actual byte-bounded/double-buffer workset partition is frozen first.
  Active output ownership is then built once into a deterministic workset-major
  offset table and flattened index. Runtime worksets consume disjoint slices;
  they do not rebuild or rescan the segment schedule.
- The planless kernel stays output-owned and uses no floating-point atomics or
  per-source output partial buffer. Workset accumulation order is stable.
- Full-segment pinned staging and retained decoded-rowgroup/exact-plan caches
  remain disabled. Current/peak plan, schedule, descriptor, pinned and native
  device byte accounting is exposed.
- CPU active-output schedule time and CUDA kernel time use separate metrics.
  Process-cold first output has an explicit flushed marker and does not include
  later artifact serialization.

## Review evidence required

Pure CPU tests cover duplicate IDs, different crops, cross-shard ownership,
grayscale, down2 and rational inverse mapping, outputs spanning worksets, empty
worksets, partial tails, strict slice ordering, concurrent lazy first touch and
legacy fallback. The real 1K metadata planning sweep is retained as a no-CUDA
sanity check.

Current GPU acceptance is intentionally a separate user-run stage. The final
binding SHA must be frozen into a new contract, then the five-case gates must
prove semantics, deterministic repeat hashes, one schedule build per segment,
valid resource bounds and zero forbidden cache/expansion counters. Only a
passing current-binding gate authorizes the independent-process cold A/B/B/A
and later DALI/PyTorch comparisons.
