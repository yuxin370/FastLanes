# Block-major DCT benchmarks

FastLens groups data by `component -> spatial DCT block -> image`. The storage
API names this layout `kSpatialMajorImageMinor`; these benchmark modules retain
the `dct_major` package name. Run commands from the FastLens repository root.
Final-paper configurations and results are in [paper/](../paper/README.md).

## Pipelines and workloads

| Pipeline | Domain | Role |
| --- | --- | --- |
| `dct_major_pushdown` | DCT | Native block-major crop pushdown with all 64 coefficients |
| `dct_major_coefficient_pushdown` | DCT | The same reader with explicit source-frequency selection |
| `rgbnomore` | DCT | Same-domain reference |
| `dali` | RGB | JPEG decoder and GPU preprocessing |
| `coordl` | RGB | Single-node JPEG cache comparison |
| `ffcv` | RGB | Prepared beton input |
| `pytorch` | RGB | PIL/torchvision comparison |

DCT paths share a DCT checkpoint; RGB paths share an RGB checkpoint. Supported
workloads are `feature-extraction`, `evaluation` and `training-throughput`.
Feature extraction returns penultimate features; evaluation records logits and
Top-1/Top-5. Training-throughput uses fixed crops and actual FP32 AdamW updates;
it is distinct from the augmented 300-epoch training experiments.

FFCV preparation uses canonical sample order and decoded RGB pixels by default.
Conversion is outside timing. `--ffcv-beton` reuses an existing matching store.
CoorDL and FFCV preserve ordinals and the partial tail.

## Input semantics and runtime

All readers use fixed order and retain the partial tail. Full stores contain
64 coefficients per component. Source-frequency masks apply before dequantization
and frequency mixing; output remains the profile's dense grid. The default
center-crop profile takes a 56x56 luma-block region from the 64x64 source and
resizes to 28x28, with corresponding chroma geometry and FP32 normalization.

The native runtime owns bounded I/O, prefetch, caches, streams and launch policy.
Python specifies workload, model and input requests. Source selection accepts
`all`, `first:N`, `list:i,j,...`; the benchmark also resolves `random:K:SEED` to
an explicit list. Contracts record storage-column and natural-frequency indices.
Use the matching full-population raw-mask reference for reduced-frequency
accuracy comparisons. K64 references can be generated directly from public
inputs using the [rebuilding guide](../paper/REBUILDING_REFERENCES.md).

## Run

Prepare the source store and access schedules before creating contracts, as
shown in [RUN_GUIDE.md](docs/RUN_GUIDE.md). A full-frequency inference example:

```bash
export PYTHONPATH="$PWD:$PWD/build/fastlens/torch${PYTHONPATH:+:$PYTHONPATH}"
python -m fastlens.benchmarks.system_dct_major.run \
  --preset e2e --workload evaluation \
  --pipelines dct_major_pushdown rgbnomore dali pytorch \
  --dct-major-manifest fastlens/data/compressed/imagenet512_val_block_major/manifest.bin \
  --dct-major-label-map fastlens/data/compressed/imagenet512_val_compact_v3/labels.json \
  --block-major-access-dir fastlens/data/compressed/imagenet512_val_block_major/access \
  --dct-coeffs all --sample-count 50000 --batch-size 50 --repeats 1 \
  --output-dir "$HOME/tmp/fastlens-block-major-evaluation"
```

Add `--dry-run` to generate the contract and commands only. A 1,024-image run
covers the first complete shard for a functional check. Production block-major
runs use zero warmup batches; model preparation is separate from input timing.
The `run_suite` entry point provides contract preparation, semantic gates,
functional runs, formal runs and model-only controls. Consult `--help` and use a
fresh output directory; `--resume` continues a matching incomplete suite.

Fixed-input training-throughput uses FP32 AdamW with learning rate 1e-4,
betas 0.9/0.999, epsilon 1e-8 and weight decay 0.05, excluding bias and
one-dimensional parameters from decay. It retains model/optimizer state across
repeats. This workload has no random augmentation, accumulation or AMP.
For paper training use the section-specific entry points instead.

## CoorDL dependency

[CoorDL](https://github.com/msr-fiddle/CoorDL) is a separate dependency; installing
modern NVIDIA DALI does not supply its caching reader. The integration uses
revision `bcde72da21781aab0661eacdedbbd1dcca5f4cfe`. Since both packages import as
`nvidia.dali`, use a separate CoorDL interpreter via `--coordl-python`.

The included [compatibility patch](coordl_compat.patch) updates build compatibility
without changing cache or preprocessing algorithms. This recipe uses CUDA 12.8
and pybind11 2.13.6; it also needs the benchmark Python packages and OpenCV,
libjpeg-turbo and Protobuf development files. Set `coordl_cuda` to the installed
CUDA 12.8 toolkit before running the configuration command.

```bash
coordl_repo="$PWD"
coordl_env=$CONDA_PREFIX
coordl_source="$HOME/tmp/coordl-baseline-source"
coordl_build="$HOME/tmp/coordl-build-cu128"
coordl_cuda="$HOME/tmp/coordl-cuda-12.8"
mkdir -p "$HOME/tmp"
git clone https://github.com/msr-fiddle/CoorDL.git "$coordl_source"
git -C "$coordl_source" checkout bcde72da21781aab0661eacdedbbd1dcca5f4cfe
git -C "$coordl_source" submodule update --init --depth 1 \
  third_party/boost/preprocessor third_party/dlpack third_party/rapidjson third_party/pybind11
git -C "$coordl_source/third_party/pybind11" fetch --depth 1 origin tag v2.13.6
git -C "$coordl_source/third_party/pybind11" checkout v2.13.6
git -C "$coordl_source" apply "$coordl_repo/fastlens/benchmarks/system_dct_major/coordl_compat.patch"
TMPDIR="$HOME/tmp" cmake -S "$coordl_source" -B "$coordl_build" -G Ninja \
  -DARCH=x86_64 -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_TOOLKIT_ROOT_DIR="$coordl_cuda" -DCUDA_TARGET_ARCHS=89 \
  -DPYTHON_EXECUTABLE="$coordl_env/bin/python" -DCMAKE_PREFIX_PATH="$coordl_env" \
  -DProtobuf_USE_STATIC_LIBS=OFF -DPYBIND11_TEST=OFF \
  -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF -DBUILD_NVTX=OFF -DBUILD_LIBTIFF=OFF \
  -DBUILD_NVOF=OFF -DBUILD_NVDEC=OFF -DBUILD_LIBSND=OFF -DBUILD_NVML=OFF -DBUILD_FFTS=OFF
TMPDIR="$HOME/tmp" cmake --build "$coordl_build" \
  --target install_backend_impl python_function_plugin -j8
TMPDIR="$HOME/tmp" "$coordl_env/bin/python" -m pip install \
  --target "$coordl_build/dali/python" future==1.0.0
cat > "$coordl_build/coordl-python" <<EOF
#!/usr/bin/env bash
export PYTHONPATH="$coordl_build/dali/python\${PYTHONPATH:+:\$PYTHONPATH}"
exec "$coordl_env/bin/python" "\$@"
EOF
chmod +x "$coordl_build/coordl-python"
export COORDL_PYTHON="$coordl_build/coordl-python"
```

Check the installed interface:

```bash
"$COORDL_PYTHON" -c 'from nvidia.dali import backend; assert "cache_size" in backend.GetSchema("FileReader").GetArgumentNames(); print("CoorDL FileReader available")'
```

`--coordl-cache-size` counts JPEGs, not bytes. The reader persists across repeats:
the first fills the cache and later repeats reuse it. The adapter uses the CPU
JPEG decoder, followed by the workload's GPU preprocessing. This measures the
single-node cache, not distributed coordinated preprocessing.

CoorDL stores data in `/dev/shm/cache`; provide enough shared memory and do not
run simultaneous readers of the same source files. After an interrupted run,
remove only the cache entries identified for that run once its processes have
stopped. Its file-list format does not support whitespace in paths.

## Results and tests

`validation.json`, `results.json` and `report.md` record semantic checks,
throughput and storage work separately. Native byte, vector, source-block,
rowgroup and pread counters describe the work performed; file-range bytes do not
measure physical SSD traffic. Reduced-frequency results retain their own oracle
and accuracy checks. Model-only timings are separate workload controls.

```bash
python -m unittest discover -s fastlens/benchmarks/system_dct_major/tests -v
```
