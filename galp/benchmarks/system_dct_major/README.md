# DCT-major locality benchmark

This directory is the isolated experiment harness for the GALP physical order

```text
component -> spatial DCT block -> image
```

which the storage API names `kSpatialMajorImageMinor` and this benchmark calls
**DCT-major**. The harness remains separate from `system_rgbnomore`; the
block-major access descriptor, compact planner, production reader, CUDA kernel,
Torch statistics and tests are implemented under their corresponding `galp/`
directories. No repository-root `src/` file is modified.

## Questions answered

The runner measures two workloads with one immutable, no-shuffle sample order:

- feature extraction: RGB-no-more ViT-Ti through `classhead.ch_tanh`, producing
  `[N,192]` penultimate features;
- evaluation: the unchanged `[N,1000]` classifier with Top-1/Top-5 accounting.

The pipeline matrix is:

| Pipeline | Domain | Purpose |
| --- | --- | --- |
| `dct_major_full` | DCT | full-image DCT decode, then reference validation crop |
| `dct_major_legacy_pushdown` | DCT | eager expanded-plan crop-pushdown control |
| `dct_major_pushdown` | DCT | native fixed-grid crop pushdown over contiguous DCT-major segments |
| `image_major_pushdown` | DCT | current image-major layout control |
| `image_major_v2_pushdown` | DCT | explicit image-major v2 layout control |
| `image_major_v3_pushdown` | DCT | Compact-v3 tiled-z32 image-major layout control |
| `rgbnomore` | DCT | strict CPU-DCT semantic reference |
| `dali` | RGB | nvJPEG/GPU transform deployment baseline |
| `pytorch` | RGB | standard PIL/torchvision baseline |

GALP and RGB-no-more share the DCT checkpoint. DALI and PyTorch share the RGB
checkpoint. DCT-vs-RGB ratios are deployment context, not elementwise model
equivalence.

For the unified six-path layout/deployment comparison, select
`dct_major_pushdown image_major_v2_pushdown image_major_v3_pushdown rgbnomore dali pytorch`.
All six consume the same immutable ascending validation sample manifest.

## Invariants

- `shuffle=false` in every sampler/reader;
- `drop_last=false` and partial-tail support;
- sample ordinal equals physical `galp_image_id`;
- all DCT paths request all 64 coefficients;
- FP32 model and preprocessing contract;
- raw DCT manifests must be coefficient-exact against their source JPEGs; transformed
  DCT comparisons allow at most one normalized integer level (`1/1020`), mean
  input error `<=1e-4`, and model-output cosine `>=0.999`, matching the existing
  RGB-no-more validation contract;
- physical bytes, vectors, source blocks, rowgroups and preads are reported;
- the validator rejects a crop-pushdown claim unless bytes, decoded vectors and
  requested source blocks all fall below the full-image control.
- the validator compares planless against the same-crop legacy eager path and
  rejects any increase in Host RSS, pinned-memory, native-GPU, or Torch-GPU peak;
- compact-plan current/peak bytes are reported independently.
- `--decode-workset-capacity-mib` is frozen in the contract (default 512 MiB)
  and runtime validation requires the exact corresponding byte value.

## Quick start

Generate and inspect a contract without running CUDA:

```bash
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run.py \
  --preset smoke \
  --workload feature-extraction \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-dry-run \
  --dry-run
```

Run the seven-pipeline feature smoke:

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run.py \
  --preset smoke \
  --workload feature-extraction \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-feature-smoke
```

Run the formal 50K evaluation:

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run.py \
  --preset e2e \
  --workload evaluation \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-eval-50k
```

The formal preset uses batch 50, all 50,000 validation images, five repeats,
and excludes repeat 0 from the hot aggregate. DCT-major defaults to a
1,000-image physical segment, the largest multiple of batch 50 below the
FastLanes 1,024-row vector width. Use `--dct-major-segment-size` for sensitivity
sweeps.

## CPU-only diagnostics and tests

```bash
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/diagnostics/segment_sweep.py \
  galp/data/imagedataset_dct/ImageNet-val/manifest.bin \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --segment-sizes 1 8 32 50 128 256 512 1024 \
  --output-json /tmp/galp-dct-major-segment-sweep.json

/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m unittest discover \
  -s galp/benchmarks/system_dct_major/tests -v
```

See [`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md) for descriptor construction,
artifacts and experiment sequencing. The earlier change-review document is
historical context; the implemented path is tracked in
[`docs/BLOCK_MAJOR_PLANLESS_DESIGN.md`](docs/BLOCK_MAJOR_PLANLESS_DESIGN.md).

## Complete publication suite

`run_suite.py` is the reproducible entry point for the complete no-shuffle
test. It performs the CPU planless planning sweep, five fail-fast GPU
kernel/cache/workset gates (sequential, random, duplicate explicit crops,
cross-shard and grayscale), both seven-pipeline semantic smokes, a
5,000-image GPU locality sweep at segments `50/250/500/1000/1024`, a
1,000-image-per-leg legacy/planless ABBA, both 50K workloads for six deployment paths,
and four model-only ceilings. The segment with the highest locality-sweep
end-to-end p50 is selected automatically for ABBA and the formal runs.

Before committing to the complete run, execute only the bounded preflight. It
plans the five segment candidates, runs 23 requests across the five GPU gates,
then writes `suite_results.json` and exits without starting a model, locality
run, ABBA, 5K, or 50K phase:

```bash
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run_suite.py \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-planless-gates-20260731 \
  --gates-only
```

```bash
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run_suite.py \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-complete-20260730 \
  --dry-run
```

Remove `--dry-run` and use a fresh directory to execute it. If an actual run is
interrupted after one or more successful phases, repeat the same command with
`--resume`. The runner never overwrites an incomplete phase.

The recommended publication suite covers `dct_major_full` in both semantic
smokes, uses the same-crop legacy eager path in ABBA and formal memory/performance
comparisons, and omits only full decode from the two 50K formal runs. Add
`--include-full-in-formal` for an exhaustive seven-pipeline 50K run. Exact test
volumes and interpretation are in
[`docs/COMPLETE_TEST.md`](docs/COMPLETE_TEST.md).
