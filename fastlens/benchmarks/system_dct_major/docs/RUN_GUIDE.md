# Block-major benchmark run guide

Run from the FastLens repository root. A complete input consists of the
block-major manifest and shards, matching labels and access sidecars, the current
Torch extension, paired checkpoints and a canonical validation sample order.
The [reference rebuilding guide](../../paper/REBUILDING_REFERENCES.md) describes
public-source data and checkpoint preparation.

## Build and prepare access metadata

```bash
cmake --build build --target _fastlens_direct_dct fastlens_block_major_access_tool --parallel
build/fastlens/tools/jpeg_dct/fastlens_jpeg_dct_tool \
  --verify-manifest fastlens/data/compressed/imagenet512_val_block_major/manifest.bin \
  --verify-workers 16 fastlens/data/system_rgbnomore/e2e_v3/imagenet_512/val
build/fastlens/tools/jpeg_dct/fastlens_block_major_access_tool \
  fastlens/data/compressed/imagenet512_val_block_major/manifest.bin \
  --output-dir fastlens/data/compressed/imagenet512_val_block_major/access \
  --output-json fastlens/data/compressed/imagenet512_val_block_major/access/build.json
```

Coefficient verification must report `coefficient_mismatches: 0` and `exact: true`.
Build access metadata before taking the contract snapshot. The scheduled profile
also uses `access/shard_000000.active_output_schedule.bin`; materialize it on a
CUDA device before timing:

```bash
PYTHONPATH=.:build/fastlens/torch \
FASTLENS_BLOCK_MAJOR_ACCESS_DIR="$PWD/fastlens/data/compressed/imagenet512_val_block_major/access" \
FASTLENS_PHASE6_NATIVE_PHYSICAL=1 \
python3 - <<'PY'
from fastlens.profiles.rgbnomore import VALIDATION_CENTER_CROP_512
from fastlens.torch import DirectDctReader

reader = DirectDctReader(
    "fastlens/data/compressed/imagenet512_val_block_major/manifest.bin",
    module_path="build/fastlens/torch",
)
pipeline = reader.pipeline(VALIDATION_CENTER_CROP_512, coefficients=range(32))
pipeline.start([list(range(1024))])
next(pipeline)
pipeline.close()
PY
test -f fastlens/data/compressed/imagenet512_val_block_major/access/shard_000000.active_output_schedule.bin
```

Full-frequency and selected-frequency requests share the ownership schedule;
their physical byte ranges are computed separately.

## Generate a fresh contract and run

```bash
export PYTHONPATH="$PWD:$PWD/build/fastlens/torch${PYTHONPATH:+:$PYTHONPATH}"
python -m fastlens.benchmarks.system_dct_major.run \
  --preset e2e --workload evaluation \
  --pipelines dct_major_pushdown rgbnomore dali pytorch \
  --dct-major-manifest fastlens/data/compressed/imagenet512_val_block_major/manifest.bin \
  --dct-major-label-map fastlens/data/compressed/imagenet512_val_compact_v3/labels.json \
  --block-major-access-dir fastlens/data/compressed/imagenet512_val_block_major/access \
  --dct-coeffs all --sample-count 50000 --batch-size 50 --repeats 1 \
  --output-dir "$HOME/tmp/fastlens-block-major-contract" --dry-run
```

Inspect the contract, then run without `--dry-run` in a new output directory.
Use the first full shard (`--sample-count 1024`) for a functional check.
`--sample-count` determines measured batches and retains the partial tail.
Block-major runs use zero warmup batches. Contracts freeze the sidecars, binding,
source identities, sample ordinals and labels before measurements.

For the coefficient-selection experiment, add
`dct_major_coefficient_pushdown`, supply an explicit selection such as
`--dct-coeffs first:32`, and provide a completed matching
`--raw-mask-oracle-dir`. Generate that oracle with the
[coefficient-mask evaluator](../../coefficient_mask_evaluator/README.md).
The K64 baseline continues to request all coefficients.

## Measurement and validation

`application-overlapped` permits input prefetch during application startup;
`controlled-io` begins cold reads after model priming.
`--evict-pipeline-file-cache` evicts only the selected pipeline's immutable
input pages. These are experiment boundaries, not reader implementation options.

Inspect `validation.json`, `results.json` and `report.md` for matching sample
traces, same-domain numerical checks, coefficient-mask oracle agreement,
physical read/decode counters and source/binary identities. K64 comparisons keep
all coefficients; approximate K selections retain separate accuracy checks.
Performance records should be collected without profiling on an idle GPU.
The [H2D guide](../../paper/section_6_3/h2d_traffic/README.md) describes separate
resource captures.

`run_suite --help` lists the full sequence of contract preparation, semantic
checks, functional runs, formal runs and model-only controls. `--dry-run` writes
its plan; `--resume` continues the same configuration without overwriting
incomplete phase outputs.
