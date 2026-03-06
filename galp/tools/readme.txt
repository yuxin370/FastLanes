GALP CLI Quick Usage

Replace the placeholders below:
- <GALP_CLI> = /path/to/FastLanes/build/galp/tools/galp_cli
- <FLS_FILE> = /path/to/FastLanes/data/fls/galp-test/data.fls

1) Decompress full table to CSV
<GALP_CLI> read_table <FLS_FILE> /tmp/out.csv

2) Decompress one rowgroup only
<GALP_CLI> read_table <FLS_FILE> /tmp/out.csv --rowgroup 0

3) Benchmark (GPU decompress only, default mega-kernel)
<GALP_CLI> benchmark <FLS_FILE> --samples 100

4) Benchmark (full table, non-mega path)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --no-mega-kernel

5) Benchmark (true one-kernel-per-sample mixed dispatch)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --gpu-dispatch-kernel

6) Benchmark with global write-back enabled
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --gpu-dispatch-kernel --write-back

7) Benchmark with launch-overhead estimate (mega-kernel)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --estimate-launch --launch-iters 10000

8) Benchmark one rowgroup with launch-overhead estimate (multi-kernel rowgroup path)
<GALP_CLI> benchmark <FLS_FILE> --samples 100 --estimate-launch --launch-iters 10000 --rowgroup 0

Notes
- Default mega path aggregates a full-table workset, then dispatches by type (can launch multiple kernels per sample).
- `--gpu-dispatch-kernel` enables GPU-side mixed dispatch and runs one mixed kernel launch per sample.
- `--no-mega-kernel` runs per-rowgroup worksets (typically more launches and higher host overhead).
- `--write-back` forces benchmark kernels to write decompressed outputs to global memory.
