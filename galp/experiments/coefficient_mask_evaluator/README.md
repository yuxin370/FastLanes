# Raw DCT coefficient-mask evaluator

This directory is independent of the RGB-no-more model implementation.  It
imports the existing `ViT` class and loads the existing K=64 checkpoint without
changing either one.

## Experimental contract

For every JPEG, the evaluator performs this sequence:

1. call `dct_manip.read_coefficients()`;
2. apply the same binary 8x8 frequency mask to every spatial block of Y, Cb,
   and Cr while the tensors are still raw quantized JPEG coefficients;
3. dequantize with the JPEG quantization tables and clamp to `[-1024, 1016]`;
4. perform RGB-no-more's validation `ResizedCenterCrop_DCT(32, 28)` frequency
   mixing and `ToRange` normalization;
5. run the unchanged K=64 DCT ViT.

The startup self-check proves that discarded K=1 coefficients are zero before
dequantization.  It also requires the K=64 CPU path to be bit-exact with the
official RGB-no-more validation transform.  CUDA preprocessing is allowed to
differ from that CPU reference by at most one integer DCT level because CPU and
CUDA float32 `einsum` can round a tie differently; the observed maximum is
written to `preprocess_audit.json`.

The 76 conditions are:

- `prefix_k01` through `prefix_k64`: the first K coefficients in the JPEG
  zigzag order;
- `high_k{04,08,16,32}`: the last K zigzag ranks;
- `mid_k{04,08,16,32}`: a centered contiguous K-rank zigzag window;
- `random_k{04,08,16,32}`: one deterministic size-K subset per budget.  The
  default seed is `20260816`, and every chosen rank/index is recorded.

## Run

Use a stable GPU UUID on hosts where CUDA ordinals and `nvidia-smi` indices do
not agree, and use the device-name guard:

```bash
CUDA_VISIBLE_DEVICES=GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76 \
  /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m galp.experiments.coefficient_mask_evaluator.evaluate \
  --output-dir galp/experiments/coefficient_mask_evaluator/runs/imagenet_val_k1_64_20260816_h100 \
  --expected-device-name 'NVIDIA H100 80GB HBM3' \
  --batch-size 64 \
  --condition-chunk-size 16 \
  --workers 2
```

An interrupted run can be continued with the identical command plus
`--resume`.  Prediction arrays are memmapped and `progress.json` is updated
atomically, so only the last uncommitted source batch can be repeated.

Run the mask-definition tests with:

```bash
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python -m unittest discover \
  -s galp/experiments/coefficient_mask_evaluator -p 'test_*.py' -v
```

## Outputs

- `top5_classes.npy`, `top5_probabilities.npy`: condition-major arrays with
  shapes `[76, 50000, 5]`;
- `per_sample_top1.csv.gz`: one row per validation sample and top-1 class,
  probability, and correctness columns for all conditions;
- `accuracy_curve.csv`, `prefix_accuracy_curve.csv`,
  `matched_budget_controls.csv`: aggregate top-1/top-5 results;
- `accuracy_curve.svg`: prefix curves with matched-budget top-1 controls;
- `samples.csv`, `conditions.json`: exact row/condition mappings;
- `preprocess_audit.json`, `run_signature.json`, `run_metadata.json`, and
  `hardware_{before,after}.json`: reproducibility and hardware audit;
- `progress.json`: resumable completion state.
