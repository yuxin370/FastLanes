# IC / Frequency-Store DCT baselines

This package compares CPU-DCT, FastLens and JPEG-DCT adaptations of Image
Calculator (IC) and Frequency Shaping (FS). The matched inference path uses the
same source images, sample order, model checkpoint and input semantics. IC/FS
perform geometry and normalization on CPU; FastLens uses its native CUDA path.
These are paper-guided adaptations to JPEG coefficients. Native IC/FS
representation experiments have separate contracts.

- [Protocol](PROTOCOL.md): source identity, numerical checks, resource budgets and timing.
- [Results](RESULTS.md): published tables, measurement scope and available evidence.
- [Integration map](ASSUMPTIONS.md): implementation choices and model adapters.

## Supported methods

| Method | Physical organization | CLI backend |
| --- | --- | --- |
| CPU-DCT | Extract original JPEG coefficients on demand | `cpu_dct` |
| FastLens | Compressed DCT store and native CUDA processing | `fastlens` |
| `ic_matched_shared` | One image per NPZ, int32 q and uint16 Q | `ic` |
| `ic_matched_batched` | Default 256 images, concatenated image-major q and Q | `ic` |
| `ic_matched_fixedk` | Explicit source frequencies at every spatial position | `ic` |
| `fs_matched_b8` | Default group256, 15 diagonal NPZs, range-checked int8/int16 q and original Q | `fs` |

Storage supports grayscale and ragged component grids. The seven model adapters
(`mobilenet24`, `mobilenet32`, `resnet24`, `resnet64`, `efun`, `vitti`, `swinv2`)
require source512 4:2:0 geometry. Spectral resize requires all 64 source
frequencies, including for models with fewer output channels. FixedK therefore
retains all64 for these resize profiles. Shape-preserving and compact-input
experiments use their [paper entries](../../paper/README.md).

## Dependencies

Run from the repository root in the `fastlens` environment. The format and
reference tools use NumPy, Pillow, PyTorch and the RGB-no-more `dct_manip`
extension; `report` also uses `jsonschema`. Model adapters need the corresponding
RGB-no-more, DCTNet or FUN source tree and checkpoint. Follow the
[reference preparation guide](../../paper/REBUILDING_REFERENCES.md) for inputs.
FastLens additionally needs its built Torch extension and storage bridge:

```bash
cmake --build build --target _fastlens_direct_dct fastlens_dct_storage --parallel
python -m fastlens.benchmarks.baselines.dct_baselines --help
python -B -m unittest discover -s fastlens/benchmarks/baselines/dct_baselines/tests -t .
```

## Prepare and run a small inference comparison

Supply a sample selection and its actual JPEG production recipe, including
resize, encoder/version/settings, quality, subsampling and provenance. The
selection must map `fastlens_image_id` to the native store for a FastLens run.
Use a fresh output directory. This 16-image example checks the workflow; its
throughput describes only that subset.

```bash
(
set -e
export CUDA_VISIBLE_DEVICES=0  # Select an available GPU.
run="$HOME/tmp/dct-inference"
python -m fastlens.benchmarks.baselines.dct_baselines prepare \
  --selection /path/to/val.json --source-recipe /path/to/source-recipe.json \
  --model resnet24 --upstream "$HOME/RGB-no-more" \
  --count 16 --batch-size 4 --pool-size 16 --workers 4 --torch-threads 1 \
  --request-order source --seed 11997733 --atol 1e-5 --rtol 1e-5 \
  --output "$run/contract"

for method in ic_matched_shared ic_matched_batched fs_matched_b8; do
  python -m fastlens.benchmarks.baselines.dct_baselines build \
    --config "$run/contract/matched.json" --method "$method" \
    --workers 8 --output "$run/$method"
  backend=ic
  if [ "$method" = fs_matched_b8 ]; then backend=fs; fi
  python -m fastlens.benchmarks.baselines.dct_baselines verify-inference \
    --config "$run/contract/matched.json" --backend "$backend" \
    --root "$run/$method" --device cuda:0 --output "$run/$method.verified.json"
  python -m fastlens.benchmarks.baselines.dct_baselines bench \
    --config "$run/contract/matched.json" --backend "$backend" \
    --root "$run/$method" --evidence "$run/$method.verified.json" \
    --device cuda:0 --run-kind inference --repeats 3 \
    --output "$run/$method.results.jsonl"
done
python -m fastlens.benchmarks.baselines.dct_baselines report \
  "$run"/*.results.jsonl --output "$run/summary.json"
)
```

The `1e-5` input/logit tolerances are declared before verification; top-1 and
top-5 class sets must agree exactly. A failed check stops that method from
producing usable timing evidence. Inspect the error before continuing.

Use the same `verify-inference` and `bench` commands for the other systems:

| System | Arguments replacing `--backend` and `--root` |
| --- | --- |
| CPU-DCT | `--backend cpu_dct` (omit `--root`) |
| FastLens | `--backend fastlens --root /path/to/manifest.bin --module-path build/fastlens/torch` |

Give each system its own evidence and result filename. Keep the same device and
contract for verification and timing. Omit `--count` for the complete selection;
choose resource budgets and timing conditions from [Protocol](PROTOCOL.md).
Run different upstream model families in fresh processes to avoid their global
module-name collisions.

## Training and native representations

`prepare --training`, `snapshot`, `calibrate` and `verify-training` replay frozen
crop/flip, RandAugment and Mixup decisions against an independent reference.
Calibration uses disjoint images and the same complete initial training state.
See [training protocol](PROTOCOL.md#training) for constraints and integration
commands. Existing IC/FS training adapters use CPU geometry and shared GPU
augmentation; their measurements describe that execution split.

The Python `native` module exposes `NativeSpec`, `encode_ic`, `fs_source` and
mosaic helpers; `fs_native` uses B32/Q50 and 63 diagonals. `candidates()` enumerates
6048 representations. `native_search prepare` freezes a task, candidate list
and disjoint data splits; search training, measured selection and final
deployment are not implemented. Native normalization uses training-only
statistics, including structural zeros.
