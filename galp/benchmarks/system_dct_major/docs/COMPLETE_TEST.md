# Complete DCT-major test

## What “best block-major sequential read” means

The logical sample order is ascending, contiguous `galp_image_id`, with
`shuffle=false` and `drop_last=false`. For the version-1
`spatial-major-image-minor` manifest this is the locality-preserving order: the
planner sorts selected shard rowgroups by physical rowgroup index, the native
reader prefetches those ordered rowgroups, and the adapter restores ordinary
model batches without changing sample order.

The six-image smoke used only four measured images after warmup, so it did not
fill a block-major segment and cannot establish the best segment size. A CPU
planning sweep on the current manifest produced:

| Images/segment | Selected/full vectors | Coverage | Planning |
| ---: | ---: | ---: | ---: |
| 50 | 4,704 / 7,967 | 59.0436% | 2.78 ms |
| 250 | 13,612 / 16,146 | 84.3057% | 62.89 ms |
| 500 | 14,735 / 16,146 | 91.2610% | 64.37 ms |
| 1,000 | 14,902 / 16,146 | 92.2953% | 68.66 ms |
| 1,024 | 14,904 / 16,146 | 92.3077% | 68.41 ms |

Therefore contiguous order is correct, but the best operational segment is not
known from the smoke. Segment 1,024 has the best physical coverage; segment
1,000 is an exact multiple of batch 50 and avoids cross-segment model batches.
These are the corrected compact-planless measurements; the sweep now requires
the access descriptor and aborts if it falls back to the eager planner. The
complete suite measures both leading candidates on the GPU and selects the largest
end-to-end p50 throughput, using the smaller segment only for an exact tie.

## Recommended publication run

First run only the bounded decision gates; this processes 23 GPU requests and
cannot continue into model, locality, 5K, or 50K work:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run_suite.py \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-planless-gates-20260731 \
  --gates-only
```

Only after `suite_results.json` reports `ok: true` should every complete-suite
command be inspected without starting GPU work:

```bash
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run_suite.py \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-complete-dryrun-20260730 \
  --dry-run
```

Run the complete publication suite in a fresh directory:

```bash
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run_suite.py \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-complete-20260730
```

After an interruption, resume completed phases without overwriting them:

```bash
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_dct_major/run_suite.py \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --output-dir /tmp/galp-dct-major-complete-20260730 \
  --resume
```

An incomplete phase is intentionally not overwritten. Inspect or move that
single phase directory, then resume.

## Exact default volume

`--gates-only` has zero model invocations: one CPU metadata planning sweep and
five GPU audits containing 2 + 8 + 8 + 4 + 1 = 23 requests.

| Phase | Data volume |
| --- | ---: |
| Feature + evaluation smokes | 84 pipeline-images |
| Five locality candidates | 5 × 5,000 × 3 = 75,000 pipeline-images |
| Crop ABBA | 4 × 1,000 = 4,000 pipeline-images |
| Feature formal | 6 paths × 50,000 × 5 = 1,500,000 pipeline-images |
| Evaluation formal | 6 paths × 50,000 × 5 = 1,500,000 pipeline-images |
| Four synthetic model ceilings | 4 × 50 × 300 = 60,000 model-images |
| **Total** | **3,139,084 model invocations** |

Each formal path traverses all 50,000 validation images five times. The six
formal paths are legacy DCT-major pushdown, planless DCT-major pushdown,
image-major pushdown, RGB-no-more, DALI, and standard PyTorch. The ABBA legs
interleave legacy and planless crop-pushdown in separate processes so memory
peaks and order effects are directly comparable. Full decode remains in both
seven-pipeline semantic smokes and is omitted from formal runs by default.

For exhaustive seven-pipeline 50K feature and evaluation, add:

```text
--include-full-in-formal
```

That increases the formal volume to 3,500,000 pipeline-images and the total to
3,639,084. Based on the six-image smoke rate, the added full-decode work alone
is roughly 22.5 GPU-hours, so this option is intentionally explicit.

## Acceptance gates

The suite is successful only when all phase validators pass:

- the sequential, random, duplicate explicit-crop, cross-shard and grayscale
  planless GPU gates pass semantic, deterministic, strategy, workset-budget and
  zero-cache checks before any 5K/50K phase starts;
- exact canonical sample trace, no shuffle, and complete partial-tail handling;
- strict DCT cross-layout input/feature or logit semantic gates;
- evaluation Top-1 agreement gates;
- strict physical-byte, decoded-vector, and source-block reductions against the
  full control;
- planless Host RSS, pinned, native-GPU, and Torch-GPU peaks no higher than the
  same-crop legacy eager control, with compact-plan peak bytes reported separately;
- hot throughput CV and endpoint drift no greater than 5%;
- unchanged runtime source fingerprints.

The primary output is `suite_results.json`. The chosen segment and all locality
candidate measurements are recorded there. Crop speedup is in
`05_crop_abba/abba_results.json`; formal DALI/PyTorch comparisons are in the two
`results.json` files under phases `06` and `07`.

DCT-major planless versus legacy, image-major, and full are strict same-domain
comparisons. DCT-major versus DALI/PyTorch is deployment-level
context because the former uses the DCT checkpoint and the latter use the RGB
checkpoint.
