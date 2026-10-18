# Round 2 storage review (independently executed)

## Scope

Reviewed the unchanged reconstructed `contracts.py` and `store.py`. The independent
probe imports those exact modules and supplies only the unavailable
`galp.benchmarks.common.sha256_file` helper using Python SHA-256. No Torch, JPEG,
GPU, formal DataLoader or model claim is made by these probes.

- `store.py` SHA-256: `4c33ead0dc179688a180555ad023f0af374f510535b09fe39548d0bac3695db2`
- `contracts.py` SHA-256: `69f2873798746939ed2e1e7eb758e011f8317eedeb8aebeb3c1e93852aabcf18`

Run artifact: `results.json`; source: `probe_storage.py`.

## R01: fixed in examined path

`write_npz` (61-69) records each encoded NPZ's SHA-256, encoded size, decoded array
bytes and coefficient payload bytes. `build_store` (160-164) incorporates this
file inventory and representation metadata into `content_fingerprint`.
`StoreReader.__init__` (236-243) checks the fingerprint, `verify_content` (265-271)
checks every payload, and `_decode_file` (318-325) checks the exact buffer it will
decode before passing it to NumPy.

Independent legal rewrites retaining shape and dtype were performed for q and Q
in both IC and FS stores (four changed-file cases). For each case both preflight
and actual read rejected the changed bytes (eight expected rejections).

A reader returning an old cached, already-validated array after a file replacement
still returns the expected content; it is not the old stale-evidence bug. A new
read from storage is checked. `__init__` alone intentionally does not preflight all
payloads; measurement and verification entry points explicitly do so.

## R02: fixed in examined retained-array lifetime path

`read` allocates final gathered q/Q outputs, reserves the configured LRU bound,
and admits whole NPZ units by decoded-byte budget and worker limit (343-373,
421-437). Workers scatter before returning, and uncached complete arrays do not
escape the worker; completed futures are cleared (392-419).

Independent 32-image/8-group fixture, 4 images per IC-b group:

|Workers|Decode admission|Observed live complete decoded bytes|Alive after return|Group opens|
|---:|---:|---:|---:|---:|
|1|19,968 B|19,968 B|0 B|8|
|4|39,936 B|39,936 B|0 B|8|

A barrier forced two simultaneous worker outputs in the parallel case; the
observations come from weak references to the real arrays, not only the reader's
own counters. Gathered outputs were checked against independent source slicing.

The README (48-57) explicitly limits these budgets to retained arrays and excludes
Python manifest/index objects and NumPy inflate/transform workspaces. It would be
incorrect to restate this as a verified total-RSS hard cap. Formal DataLoader worker
aggregate memory and lookahead remain to be measured.

## Mechanism regression results

Cold reads and five repeated fully cached reads were checked for IC single-image,
IC-b and FS with ragged component grids, nontrivial Q values, duplicate/unsorted
requests, spatial subregions and partial/unsorted diagonal frequencies. Every q/Q
was exact. Cold requests were coalesced; all warm repetitions had zero file opens
and zero decompressed q. FS retained its whole-diagonal overfetch semantics.

## Low-priority issue: cache-only requests reserve a nonexistent decode

`store.py` 366-369 tests every requested unit against the available decode budget,
including units that are already in the decoded cache. This can reject an entirely
cached request that fits in the real retained-array limit.

Reproduced example:

- Already decoded cache: 1,664 B.
- Requested full output including Q: 1,664 B.
- Required resident memory: 3,328 B.
- Memory limit: 3,584 B.
- Result: `MemoryError`, because the remaining 256 B is less than a 1,664 B unit
  even though no unit requires decoding.

Priority P3: an overly conservative admission rule, not incorrect data, unbounded
memory, or a headline blocker by itself. Determine misses first and reserve decode
capacity only for them, while preserving the output plus cache bounds.

The per-read `decoded_cache_peak_bytes` also reports zero for an all-hit read
because it is updated only on insertion (418); initialize it from current cache
occupancy if it is intended as actual resident peak rather than added occupancy.

## Performance/fairness follow-ups (not demonstrated bugs)

1. Per-use SHA-256 in `_decode_file` is inside the timed read path. Compare whether
   JPEG/FastLens incur equivalent integrity checks, or quantify/isolate this added
   engineering cost under a common integrity policy. Full preflight warms the OS
   cache; the README correctly excludes cold-cache claims for those runs.
2. Formal CNN/Transformer adapters now call the same reader via `__getitems__` once
   per actual microbatch and attach read counters once, which repairs the prior
   per-sample repeated decode within a microbatch. They instantiate per-worker
   readers with zero decoded cache and one internal decode worker, so this is not
   an implementation of cross-microbatch/cross-consumer FS sharing. The latter
   needs its own equal-budget experiment.
3. A full group256 plus randomly scattered microbatches can legitimately create
   large read/decode amplification. Include sequential/complete-group/shared
   consumer cases as well as random ones. Measure sensible bounded caches and
   batch/pool coalescing before selecting the strongest matched baseline.
