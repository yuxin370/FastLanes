# Integration map and evidence boundaries

Inspected repository revision: `13fabad2b337945ea32d50dd6b6cbebc5563b031`.
The worktree contained existing inference-metrics changes when work began;
this implementation leaves those files untouched.

| Actual repository path | Role in this implementation |
| --- | --- |
| `fastlens/benchmarks/dct_models/online_crop.py:read_source_jpeg,SourceReference` | Existing full-source CNN/eFUN CPU oracle; all64 dependencies and source512 center crop |
| `fastlens/benchmarks/dct_models/backend.py` | Four DCTNet model/checkpoint/channel/normalization contracts |
| `fastlens/benchmarks/dct_models/efun_backend.py` | eFUN model and 192-channel unnormalized output |
| `fastlens/benchmarks/dct_models/dct_geometry.py:load_upsample_dct` | Reuses upstream spectral resize without importing conflicting global `utils` packages |
| `fastlens/benchmarks/baselines/system_rgbnomore/inference/model_factory.py` | ViT-Ti/SwinV2-T registered model builders and checkpoint names |
| `$HOME/RGB-no-more/datasets.py`, `utils/custom_transforms.py` | Independent Transformer CPU validation oracle, including resize rounding |
| `$HOME/RGB-no-more/dct_manip/dct_manip.cpp:read_coefficients` | Installed direct libjpeg q/Q extraction; no pixel decode/re-encode |
| `fastlens/benchmarks/dct_models/dctnet_storage.cpp`, `storage.py` | Existing IPC bridge extended with `read-source-manifest` for the actual block-major source integer gate |
| `fastlens/torch/direct_dct.py`, `fastlens/profiles/dct.py` | Reused public native pipeline with explicit occurrence crop/flip and all192 intermediate CNN channels |
| `fastlens/benchmarks/dct_models/training_pls.py`, `training_jpeg.py` | Existing model reset, precision, augmentation geometry and tail/accumulation semantics reused |
| `fastlens/benchmarks/training_pls/run_matrix.py`, `train.py`, `contracts.py` | IC/FS use the original ViT-Ti/Swin matrix, immutable recipe, checkpoint lifecycle and epoch metrics |
| `fastlens/benchmarks/baselines/system_rgbnomore/training/pipeline.py` | IC/FS coefficient acquisition feeds the original Transformer JPEG preprocessing adapter |
| `fastlens/benchmarks/dct_models/report_cnn_suite.py`, `fastlens/benchmarks/training_pls/report_equal_image_performance.py` | Existing result tables accept additional matched store paths and retain concrete storage identities |
| `fastlens/benchmarks/training_pls/published_optimizer.py`, `published_augmentation.py` | Reused AdamW/independent decay/update schedule and grouped/scalar augmentation; random choices are frozen before replay |
| `fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json` | Present, 50,000 original validation source records |
| `fastlens/data/compressed/imagenet512_val_block_major/manifest.bin` | Present, canonical online FastLens validation source |
| `fastlens/data/system_rgbnomore/e2e_v2/checkpoints/` | Present, CNN/eFUN/ViT-Ti/SwinV2 checkpoints; runtime hashes come from actual selected files |
| `fastlens/benchmarks/baselines/system_rgbnomore/training/prepare_imagenet512_v3_train.py` | Training source generator documents Pillow bilinear 512x512 and default JPEG options; this alone is not proof of validation source provenance |

Observed Python environment: `$HOME/miniconda3/envs/fastlanes-cuda/bin/python`,
PyTorch `2.11.0+cu128`; NumPy/Pillow/jpeg2dct/dct_manip present. The default
base Python lacks PyTorch. The user confirmed host CUDA and three GPUs
(4090, H100, RTX PRO6000 Blackwell); the sandbox still reports CUDA unavailable.
The sandbox also mounts `$HOME/tmp` read-only. Automatic permission review for
temporary-file tests timed out. The user later ran the standard CPU suite and
H100 numerical checks successfully on the host; no CUDA performance measurements,
full ImageNet conversion or full epoch runs were made.

Evidence labels refer to the supplied implementation plan: **P** paper
mechanism, **D** derivation, **E** explicit engineering choice, **U** unresolved
author/environment detail. This work uses that supplied specification;
it makes no new claim of discovering or executing author source code.

| Level | Decision / scope |
| --- | --- |
| P | IC numeric NPZ/DEFLATE, int32 native mosaics, triangular `u+v<k`, B/sampling/scalar candidate domain |
| D | B8,k8 keeps 36 entries; B32,k3 on 256x256 produces three 24x24 planes; FS has 2B−1 columns |
| P/E | FS group256 and int8/int16 are specified; exact data-dependent smallest signed width is the plan's explicit range-safe rule |
| E | Archive keys: matched `q` and `tables`; native IC `mosaic`, `shapes`, `offsets`, UTF-8 contract stored as numeric uint8 |
| E | Store v1 JSON defines group/image/component geometry, labels, source IDs, masks and offsets; Q is numeric uint16 in NPZ. Round-2 requires every payload's size/SHA-256 and a complete content fingerprint; older stores must be rebuilt |
| E | FS order is image → component → raster block → diagonal ascending u; one file per group/diagonal, separate group Q archive |
| E | IC-b has one concatenated q member, not separately compressed per-image members |
| E | NumPy `savez_compressed` defaults determine DEFLATE settings; exact ZIP/NPY headers are counted; no alternate codec, bit plane or spatial seek index |
| E | Native RGB uses [0,255] FP64 BT.601-style Y/Cr/Cb (`.299,.587,.114`, `.713`, `.564`), shifts every plane by 128; edge padding, orthonormal DCT-II and ties-to-even rounding |
| E | Native normalization uses training-only per-plane population statistics over all mosaic positions, including structural zeros; zero variance is an explicit error |
| E | Bounded decode/scatter chunks, default 512 MiB active decoded arrays / 1 GiB per reader, zero LRU cache, one next preparation pool. Manifest objects, inflate/transform workspace and model/optimizer memory are outside retained-array budgets; no total RSS claim |
| E | CNN label mapping is sorted class names from the full selection, matching existing `evaluate.samples`; Transformer labels retain the source manifest mapping |
| E | Transformer CPU adapter preserves upstream int16 q×Q arithmetic, clamp, resize-to-int16 rounding and ToRange order; CNN uses its existing FP32 dequantize/resize/round/saturate/normalize order |
| E | Exact tolerance is the default for identical CPU reference operations. An explicit nonzero diagnostic tolerance is not automatically calibrated or paper-derived |
| U | Original array schema, native color/level shift/tie/statistics conventions and IC-b exact member layout are incompletely disclosed; the E rules above are not attributed to recovered author code |
| U | Actual historical validation resize/encoder recipe must be supplied with provenance before formal experiments; no guessed production recipe is written into the repository |
| E | Training trace freezes RandAugment operation/magnitude, Cutout center/pad per component, translation blocks, ChromaDrop channel, both Mixup weights and actual partner occurrences; replay does not resample seeds |
| E | Calibration and verification reserve disjoint image IDs, share the complete initial model/AdamW/decay/scheduler/RNG snapshot and preserve pool/epoch tails; predeclared numerical bounds are checked and frozen per backend/device |
| E | Training correctness uses eager execution with the existing FP32 ViT / BF16 CNN/eFUN/Swin precision and actual-sample accumulation; small diagnostic microbatch sizes remain explicit contract choices |
| E/U | Ten paired CPU and H100 updates on three actual architectures passed for IC/IC-b/FS with synthetic JPEG inputs; H100 calibration and verification use zero limits and observe zero differences. FastLens on the same real source/trace and the full cross-backend CUDA G3 remain unverified |
| U | Matched resource calibration, shared multi-consumer measurements and G5 full runs remain outstanding |

Round-2 replaces formal per-sample coefficient reads with merged per-microbatch
reads in both existing DataLoader adapters. Consumed-batch counters include
worker initialization; unconsumed lookahead remains outside their accounting.
Store preflight hashing runs before measured windows and warms the file cache;
the same bytes are hashed again on actual decode. No cold-cache or G4 claim is
made. The replay schema is `0.2-diagnostic`, separate from existing formal
experiment schemas, which retain their normal result and checkpoint lifecycle.

Implemented correctness checks cover integer/Q round trips, ragged and odd
JPEG geometry, negative width boundaries, full-member read amplification,
tail groups/batches, duplicate/ordered requests, bounded cache eviction,
corrupt members, native scalar DCT oracle and IC/IC-b/FS native equality.
Real-source checkpoint smoke checks are separate from full validation;
neither their sample count nor their timings are described as a full run.
