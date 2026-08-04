# Deterministic parallel encoding

FastLanes can encode independent rowgroups concurrently while preserving the existing FLS byte layout. The original
`Connection::to_fls(path)` API remains the default and uses one worker. Parallelism is enabled explicitly:

```cpp
fastlanes::EncodingOptions options;
options.worker_count = 4;
options.max_inflight_rowgroups = 8;
options.max_inflight_bytes = 256ULL * 1024 * 1024;

connection.to_fls(output_path, options);
const auto& stats = connection.get_last_encoding_stats();
```

`max_inflight_rowgroups == 0` selects an adaptive window of twice the effective worker count. `rowgroups_per_task == 0`
selects an adaptive contiguous chunk size. `max_inflight_bytes == 0` disables the completed-payload byte limit.
Deterministic ordered commit is mandatory; setting `deterministic_ordered_commit` to false is rejected.

## Thread-safety and ordering

Table preparation, schema selection, and descriptor preparation remain serial. The resulting rowgroup and column order
is fixed before workers start.

The encoder then creates a fixed-size worker pool. Each claimed task covers a contiguous rowgroup range. Each encoded
rowgroup owns its output `Buf`, `InterpreterState`, expression objects, and a deep copy of its rowgroup descriptor. Workers
read different input rowgroups and never write a shared output buffer or the connection's published descriptor.

Completed payloads are admitted in rowgroup order. The caller thread is the only writer: it appends each payload in the
original order, assigns its final prefix-sum offset, and moves the completed descriptor into a private table descriptor.
That descriptor replaces the connection descriptor only after all workers join successfully. Scheduling order therefore
cannot alter payload order, column layout, offsets, footer bytes, or the final file.

The inflight rowgroup window counts claimed but not yet committed rowgroups, including work that is still encoding. It
bounds the task queue and applies backpressure before more work is claimed. The optional byte limit applies to completed
payloads admitted to ordered commit. One payload larger than the byte limit is allowed only when the completed queue is
empty, which guarantees progress. Worker-local expression memory is additionally bounded by the effective worker count;
there are no detached threads or per-rowgroup `std::async` calls.

The first exception cancels unclaimed work, wakes blocked workers, joins the complete pool, and is rethrown on the caller
thread. Output is written in a unique staging directory beside the destination. The FLS file is published last, after its
inline or external descriptor and optional JSON sidecar are complete. Failure removes the staging directory and does not
publish a partial FLS file.

## Statistics

`Connection::get_last_encoding_stats()` exposes the requested and effective worker counts, encoded rowgroup count,
resolved rowgroup window and chunk size, peak inflight rowgroups, peak completed-payload bytes, and wall times for serial
preparation/schema selection, rowgroup encoding, finalization/publication, and the complete `to_fls` call. The statistics
describe the most recent successful `to_fls` call.

## Generic microbenchmark

Configure with `FLS_BUILD_BENCHMARKING=ON`, build `bench_parallel_encoding`, and run for example:

```sh
./benchmark/bench_parallel_encoding \
  --rowgroups 32 \
  --columns 8 \
  --rows-per-rowgroup 65536 \
  --type i64 \
  --workers 1,2,4,8 \
  --repetitions 3
```

The benchmark constructs a synthetic `MemoryTable` and emits CSV containing complete and rowgroup-encoding wall times,
phase-specific rows/s and encoded bytes/s, peak inflight rowgroups and bytes, resolved bounds, task chunk size, and process
peak RSS. `--schema auto|compressed|uncompressed` selects normal schema spelling, a type-appropriate generic compressed
operator, or an uncompressed diagnostic. Supported data types are `i8`, `i16`, `i32`, `i64`, `float`, `double`, and
`string`. `--cardinality` controls synthetic value diversity so codec compute can be separated from ordered-write
bandwidth; zero preserves the type-specific default.

The `ParallelEncoder.*` unit tests cover mixed numeric/string input, single and multiple rowgroups, uneven tails and
rowgroup sizes, workers 1/2/4/8, worker oversubscription, inline and external footers, bounded inflight configuration,
empty input, decode equality, staged cleanup, SHA-256 equality, and five repeated parallel runs.
