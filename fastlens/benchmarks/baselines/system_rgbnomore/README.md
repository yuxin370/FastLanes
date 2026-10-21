# RGB-no-more system benchmarks

This package provides paired DCT and RGB workloads for the independent FastLens
project. Run commands from the FastLens repository root. The final-paper
experiments are organized under [paper/](../../paper/README.md).

| Pipeline | Input | Implementation |
| --- | --- | --- |
| `fastlens` | JPEG DCT | Native Direct-DCT reader and model profile |
| `rgbnomore` | JPEG DCT | RGB-no-more reference reader |
| `dali` | RGB | JPEG decoding and GPU preprocessing |
| `pytorch` | RGB | PyTorch/torchvision reference |

DCT paths share a DCT checkpoint; RGB paths share an RGB checkpoint. All use the
same canonical sample order without implicit shuffle. RGB/DCT comparisons are
system comparisons with distinct model inputs.

## Setup and inference

Use [Rebuilding reference inputs](../../paper/REBUILDING_REFERENCES.md) to obtain
public model sources/checkpoints and prepare ImageNet-512, manifests and fresh
prediction records. Published DCT checkpoints expect 512x512 source JPEGs; the
runner checks their dimensions. The native library and Torch extension are
built through the repository's root CMake project.

```bash
export PYTHONPATH="$PWD:$PWD/build/fastlens/torch${PYTHONPATH:+:$PYTHONPATH}"
python -m fastlens.benchmarks.baselines.system_rgbnomore.inference.run \
  --preset e2e --pipelines fastlens rgbnomore dali pytorch \
  --output-dir "$HOME/tmp/fastlens-e2e" \
  --data-root fastlens/data/system_rgbnomore/e2e_v3/imagenet_512 \
  --index-csv fastlens/data/system_rgbnomore/e2e_v2/indexbase_val.csv \
  --rgb-checkpoint fastlens/data/system_rgbnomore/e2e_v2/checkpoints/imgnetRGBViTTi_ep300_74.1.pth \
  --dct-checkpoint fastlens/data/system_rgbnomore/e2e_v2/checkpoints/imgnetDCTViTTi_ep300_75.1.pth \
  --fastlens-manifest fastlens/data/compressed/imagenet512_val_compact_v3/manifest.bin \
  --fastlens-label-map-json fastlens/data/compressed/imagenet512_val_compact_v3/labels.json \
  --torch-binding-dir build/fastlens/torch
```

Use a fresh output directory. `--preset smoke` selects a small functional run;
`--dry-run` generates the contract and commands without executing the pipelines.
See [the inference guide](docs/E2E_COMPARISON_RUN_GUIDE.md) for model selection.

| Preset | Batch | Warmup batches | Measured batches | Repeats |
| --- | ---: | ---: | ---: | ---: |
| `smoke` | 2 | 1 | 2 | 1 |
| `e2e` | 50 | 0 | 1000 | 5 |

The measured interval includes reads, decoding, transforms, H2D, forward passes,
metrics and final CUDA synchronization. Model construction, checkpoint loading,
contract generation and warmup are outside it. Workload overrides are recorded
in the contract; native planner, cache and launch policies belong to C++ profiles.

## Training and outputs

The [training guide](docs/TRAINING_BENCHMARK_RUN_GUIDE.md) describes the four-path
correctness and throughput runner. Its `dali` path uses planned D2 augmentation.
The equal-image runner exposes `d2`, `d3` and `pytorch`; D3 uses DALI-native
shuffle/augmentation and is a separate performance reference.

Outputs include `contract.json`, a sample manifest, `pipeline_<name>.json`,
semantic arrays, `results.json/csv`, `report.md`, `validation.json` and commands.
Diagnostics are separate from the production API. CPU checks:

```bash
python -m unittest -v \
  fastlens.benchmarks.baselines.system_rgbnomore.tests.test_system_benchmark \
  fastlens.benchmarks.baselines.system_rgbnomore.tests.test_training_benchmark
```
