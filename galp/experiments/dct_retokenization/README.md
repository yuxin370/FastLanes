# DCT retokenization experiment

This package evaluates whether spatial retokenization can reduce the token
count of the RGB-no-more DCT ViT without changing GALP's production runtime.
It is an experiment package: it does not add a stable GALP API and does not
own scheduling, storage, CUDA lifetime, or physical I/O.

## Boundary

The experiment applies a zigzag-prefix coefficient mask before dequantization
and frequency-mixing transforms, then evaluates one of three model shapes:

- post-projection spatial merge;
- DCT super-patch projection;
- delayed latent merge after an early Transformer block.

The implementation imports shared coefficient-mask definitions from
`galp.experiments.coefficient_mask_evaluator` and the published training
schedule from `galp.benchmarks.system_dct_major.training_pls`. It does not
import private benchmark executors.

## Dependencies and inputs

In addition to GALP and PyTorch, the experiment requires an external
RGB-no-more checkout containing `dct_manip`, `utils.dct_ops`, and
`models.plainvit`. Pass it with `--rgbnomore-root` or set
`RGBNOMORE_ROOT`. Model evaluation also requires the original DCT ViT
checkpoint and an ImageNet manifest whose samples are 512x512 JPEG 4:2:0.

The external checkout is loaded only at the experiment boundary. It is not a
production dependency of GALP.

## Entrypoints

Run modules from the repository root so package imports are deterministic:

```bash
python -m galp.experiments.dct_retokenization.evaluate --help
python -m galp.experiments.dct_retokenization.train_superpatch --help
python -m galp.experiments.dct_retokenization.benchmark_model --help
python -m galp.experiments.dct_retokenization.summarize_results --help
```

Frozen experiment contracts live in `configs/`. The small unit suite is:

```bash
python -m unittest discover -s galp/experiments/dct_retokenization/tests
```

## Outputs

Evaluation and training entrypoints write JSON/CSV evidence, predictions, and
checkpoints beneath a caller-selected output directory. `runs/`, checkpoints,
NumPy dumps, JSONL logs, and Python caches are generated artifacts and are not
source-controlled. Existing historical results remain outside this package's
Git history.
