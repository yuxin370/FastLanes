# Direct-DCT data loading

[Documentation index](../README.md)

The Python API is available through `fastlens.torch`. Run commands from the
repository root with a compiled Torch extension and a working CUDA device.
Prepare at least four 512×512 RGB, 4:2:0 JPEGs using the
[Converter](CONVERSION.md) for these examples. A compatible DCT
model consumes these tensors; an arbitrary RGB model does not.

## Reader and pipeline

```python
from fastlens.profiles.rgbnomore import VALIDATION
from fastlens.torch import DirectDctReader

reader = DirectDctReader(
    "fastlens/data/example-fastlens/manifest.bin",
    module_path="build/fastlens/torch",
)

with reader.iter_batches(
    [[0, 1], [2, 3]],
    profile=VALIDATION,
    coefficients=None,
) as batches:
    for batch in batches:
        print(batch.sample_ids, batch.y.shape, batch.cbcr.shape)
        # A compatible DCT model consumes batch.y and batch.cbcr on CUDA.
```

`module_path` points to the compiled PyTorch extension. The profile determines the input geometry, normalization, and tensor representation expected by the model.

The `VALIDATION` profile used here supports subsets, reordered sample IDs and
batches crossing shards. For the `VALIDATION_CENTER_CROP_512` scheduled path,
submit one complete shard per batch with image IDs in ascending order. That
batching requirement applies specifically to this scheduled profile.

By default, `coefficients=None` reads all 64 source DCT coefficients. You can also request specific frequencies:

```python
coefficients=[5, 0, 2]
```

For CNN workloads, `DctModelProfile` defines the output channels, geometry, and normalization. `DirectDctPipeline` then exposes the projected NCHW tensor as `batch.projected`.

The same store also supports the public asynchronous pipeline with a Direct-DCT profile:

```python
from fastlens.profiles.rgbnomore import VALIDATION
from fastlens.torch import DirectDctPipeline

with DirectDctPipeline(
    "fastlens/data/example-fastlens/manifest.bin",
    profile=VALIDATION,
    batch_size=2,
    module_path="build/fastlens/torch",
) as pipeline:
    for batch in pipeline.start([[0, 1], [2, 3]]):
        print(batch.sample_ids, batch.y.shape, batch.cbcr.shape)
```

Training uses `TrainingPolicy` with a dataset mapping. Call `pipeline.start_epoch(epoch)` to start an epoch; training batches provide labels through `batch.targets`.

Batches retain their underlying CUDA storage. If a batch is consumed on another CUDA stream, call `batch.record_stream(consumer_stream)` before using it there.

A complete example with a small CUDA classifier is included:

```bash
python3 fastlens/examples/direct_dct_pipeline_demo.py \
  fastlens/data/example-fastlens/manifest.bin \
  --module-path build/fastlens/torch \
  --profile validation --batch-size 2 --steps 3
```

See the [architecture guide](../../ARCHITECTURE.md) and
[DCT augmentation semantics](DCT_AUGMENTATIONS.md) for details.

## Labels and classification

The existing tiny classifier demo can use real integer labels for evaluation
or training smoke checks (the input geometry must satisfy the chosen DCT profile):

```bash
python fastlens/examples/direct_dct_pipeline_demo.py ./train_store/manifest.bin \
  --module-path ./build/fastlens/torch --annotations --num-classes 1000 \
  --batch-size 32 --steps 3
# Add --train-smoke to run CrossEntropy backward and optimizer steps.
```

This demonstrates label alignment with a freshly initialized tiny model; its
loss is not a pretrained ImageNet accuracy result. In application code, validate
the sidecar once, then gather labels by the physical IDs returned by the reader:

```python
from pathlib import Path
import torch
from fastlens.annotations import load_annotations

samples, metadata = load_annotations(Path("train_store"))
# For classification, first check integer labels against your model's class range.
labels_by_id = torch.tensor([sample["label"] for sample in samples],
                            dtype=torch.long, device="cuda")
for batch in batches:  # Your DirectDctReader iterator; IDs may be nonsequential.
    ids = torch.tensor(batch.sample_ids, dtype=torch.long, device="cuda")
    targets = labels_by_id[ids]
    loss = torch.nn.functional.cross_entropy(model(batch.y, batch.cbcr), targets)
```

The sidecar format is documented in [Annotations](ANNOTATIONS.md). Native
manifest inputs and the published Store boundary are described in
[Store format](../spec/STORE.md#standard-store-and-native-manifest-apis).
