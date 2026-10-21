# JPEG-DCT pipeline parallelism

The JPEG-DCT writer exposes independent controls for work that occurs in
different pipeline phases:

- `--layout-threads`: dataset-wide JPEG layout scan.
- `--shard-decode-threads`: coefficient decoding inside one active shard.
- `--shard-workers`: shards processed concurrently.
- `--encoding-workers-per-shard`: generic FastLanes rowgroup encoders used by
  one shard.
- `--verify-workers`: coefficient-exact verifier workers, partitioned at
  manifest shard boundaries.

`--threads N` remains a compatibility option. Used alone, it maps to both
layout and shard-decode threads. Combining it with a different explicit
`--layout-threads` or `--shard-decode-threads` value is an error; values are
never silently normalized.

Layout scan completes before shard processing starts. Within each shard,
coefficient decode and FastLanes encoding are sequential phases, so their
thread counts do not multiply. The writer enforces both
`layout_threads <= 256` and
`shard_workers * max(shard_decode_threads, encoding_workers_per_shard) <= 256`
before publishing output. The exact verifier has a separate hard maximum of
16 workers. All phases join their workers and propagate failures before the
transactional generation is published.

Shard assignment and output commit remain deterministic. Changing any of the
parallel controls must not change a shard FLS or metadata byte. The manifest
itself contains its transactional generation directory name, so manifests
from separate successful runs can differ even when every referenced artifact
is byte-identical.

The full ImageNet preparation wrapper defaults layout/decode to 32/4 and exact
verification to 16. The current 100-JPEG canary favors 16 shard workers with
2 encoding workers per shard over the tested 24x1, 24x2, and 32x1 variants,
but a fresh canary under the intended machine load remains required before a
new full-data run.
