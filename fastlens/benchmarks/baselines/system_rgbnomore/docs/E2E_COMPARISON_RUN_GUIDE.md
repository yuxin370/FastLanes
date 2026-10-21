# RGB-no-more end-to-end inference guide

Run from the FastLens repository root after the [reference preparation
steps](../../../paper/REBUILDING_REFERENCES.md). The paired model registry in
`inference/model_factory.py` specifies each architecture, checkpoint pair,
preprocessing pipeline, input shape and native semantic profile together.

| Model ID | Checkpoint files |
| --- | --- |
| `rgbnomore-vitti-224-v1` | `imgnetRGBViTTi_ep300_74.1.pth`, `imgnetDCTViTTi_ep300_75.1.pth` |
| `rgbnomore-swinv2-t-256-window8-v1` | `imgnetSwinRGB_ep300_79.0.pth`, `imgnetSwinDCT_ep300_79.4.pth` |

The Swin inference model uses input size 256 and window size 8. Select the
registered model rather than loading those weights into a 224/window-7 training
variant. The source JPEG recipe is 512x512; the default dimension check is 512.

## Build and run

```bash
cmake --build build --target _fastlens_direct_dct --parallel
export PYTHONPATH="$PWD:$PWD/build/fastlens/torch${PYTHONPATH:+:$PYTHONPATH}"
python -m fastlens.benchmarks.baselines.system_rgbnomore.inference.run --help
python -m fastlens.benchmarks.baselines.system_rgbnomore.inference.run \
  --preset smoke --pipelines fastlens rgbnomore dali pytorch \
  --output-dir "$HOME/tmp/fastlens-inference-smoke"
```

After the functional check, use `--preset e2e` and a new output directory.
This preset uses batch 50, zero warmup batches, 1,000 measured batches and five
repeats. Override `--repeats 1` when generating a single-pass prediction reference
or a single-window profiler input. Supply the checkpoint, data, manifest and
label-map arguments shown in the [package README](../README.md) when their
locations differ from the defaults.

The `fastlens` path uses the native production profile. Workload parameters
such as batch size and workers are configurable; implementation policies such
as rowgroup prefetch, caches, streams and kernel launch limits are native-owned.

## Contracts and validation

`--dry-run` still reads and checks real input metadata, then writes
`contract.json`, the selected samples and `commands.json`. Removing `--dry-run`
executes each pipeline and the validator. It is not necessary to obtain an
old contract from a previous experiment.

After creating or relocating a store, use
`--refresh-fastlens-payload-fingerprints` once outside a timed run. Subsequent
runs reuse and verify the resulting payload fingerprint cache. Generate fresh
contracts after changes to inputs, binaries or source files.

DCT pipelines are checked against their same-domain reference. RGB comparisons
retain their decoder/preprocessing diagnostics. Examine `validation.json`
separately from performance statistics. Pipeline JSON files link to the contract
and semantic arrays, including full-population predictions needed by later runs.

The [rebuilding guide](../../../paper/REBUILDING_REFERENCES.md) shows how to pass
these files to a direct paper worker and how to capture H2D traffic using the
newly generated command, without the dated historical selector.
