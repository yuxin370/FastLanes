# RGB-no-more 四管线训练 Benchmark 运行指南

v2/v3 共用 reader、严格 manifest preflight、v3 canary 与 block-major 暂缓边界见
[`TRAINING_LAYOUT_COMPATIBILITY_2026-07-31.md`](TRAINING_LAYOUT_COMPATIBILITY_2026-07-31.md)。

## 1. 数据 manifest

训练集和验证集各使用一个 JSON 文件。`split` 分别为 `train` 和 `val`；标签固定为 ImageNet-1K 的零基 `[0,1000)` 标签。相对路径相对于 manifest 所在目录解析。

```json
{
  "split": "train",
  "label_mapping": "imagenet-1k-zero-based",
  "samples": [
    {
      "logical_sample_id": "n01440764/ILSVRC2012_val_00000293.JPEG",
      "path": "/data/imagenet/train/n01440764/example.JPEG",
      "label": 0,
      "width": 500,
      "height": 375,
      "payload_sha256": "可选但建议填写",
      "galp_image_id": 123
    }
  ]
}
```

`galp_image_id` 仅是 GALP 路径的原生 image-major ID，但四条路径仍以 `logical_sample_id` 审计样本身份。runner 会校验文件内容 hash、重复 ID、非法标签，以及训练/验证的 ID 或文件路径污染。

正式复现 RGB-no-more 的 512×512 输入数据并报告官方 ImageNet accuracy 时，train
与 validation 必须来自两个独立的官方 split，也必须分别绑定两个 GALP manifest。
单 manifest held-out 模式只保留给 canary/布局 A/B，不能报告官方 ImageNet
validation accuracy。

先准备完整 512×512 train JPEG 和 Compact-v3：

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
set -o pipefail
PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python

"$PY" -B \
  galp/benchmarks/system_rgbnomore/training/prepare_imagenet512_v3_train.py \
  --phase all \
  --resize-workers 32 \
  --layout-threads 32 \
  --shard-decode-threads 4 \
  --shard-workers 16 \
  --encoding-workers-per-shard 2 \
  --verify-workers 16 \
  2>&1 | tee \
  galp/data/system_rgbnomore/e2e_v3/prepare_imagenet512_v3_train.full.log
```

`--compress-threads N` 仍作为兼容参数，同时映射 layout scan 和 shard JPEG
decode；不要将它与取值不同的 `--layout-threads` 或
`--shard-decode-threads` 混用。准备脚本会在 `plan.json`、`commands.json` 和
completion artifact 中记录上述五个阶段的 effective values。正式全量重跑前应先用
canary 比较 `24 shard × 2 encoding`、`32 × 1` 和 `16 × 2` 的 wall time、RSS
与内存带宽；当前 100-JPEG canary 的保守候选是 `16 × 2`。

当 train 和现有官方 validation 的 v3 manifest 均完成后，生成两个 manifest-local
`galp_image_id` 的训练 JSON：

```bash
"$PY" -B \
  galp/benchmarks/system_rgbnomore/training/prepare_imagenet512_v3_train.py \
  --phase manifests
```

该步骤严格要求 train=`1,281,167`、validation=`50,000`、manifest-v3、
`image-major-vector-rowgroups` 和 `tiled-z32`。默认产物为
`e2e_v3/training_manifests_official_v3/{train,val}.json`。

## 2. 只生成 contract

这一步不创建模型，也不要求 GPU：

```bash
PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python

"$PY" galp/benchmarks/system_rgbnomore/training/run.py \
  --enabled-pipelines galp,rgbnomore,dali,pytorch \
  --required-comparison-groups dct,rgb \
  --phase smoke \
  --train-manifest /path/to/train.json \
  --val-manifest /path/to/val.json \
  --galp-manifest /path/to/train/manifest.bin \
  --galp-validation-manifest /path/to/validation/manifest.bin \
  --output-dir /tmp/galp-training-contract \
  --dry-run-contract
```

## 3. execution mode

`--execution-mode audit|runtime` 是不可变 contract 字段，默认是 `audit`。

- `audit` 保留逐 stage 同步、逐 step 梯度扫描、首个 measured step 参数快照等完整审计行为，适合语义核验，不能当作生产吞吐。
- `runtime` 仍先在 fresh clone 上执行同一个 audit first-step probe，但 probe 不进入 warmup 或 measured window。正式 measured loop 不调用逐参数统计或逐 step `.item()`；CUDA wall-clock 只在 measured region 前后各同步一次，stage 用 CUDA event 在 region 结束后解析。
- `--resume-run` 只能恢复相同 mode；显式传入不同 mode 会立即失败。

`--workers N` 是训练 workload 的外层数据 worker 参数，不再转发为 native
rowgroup-prefetch worker 数；native I/O 并行度由 production profile 固定。
`N=0` 表示不创建外层 DataLoader worker。batch lookahead 使用冻结的 production
配置；训练入口不再接受 `--prefetch-depth`。GALP 将整个逻辑 schedule 交给
native-owned 有界 Pipeline，Python 不维护 FIFO、future、backpressure 或 cancel/drain。

模型固定为 RGB-no-more ViT-Ti，augmentation 固定为 published recipe，precision
固定为 FP32。因为三者都只有一个生产值，`--model-architecture`、
`--augmentation-recipe` 和 `--precision` 已从训练入口删除。

下列短命令使用当前正式数据路径：

```bash
PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
E2E=/home/tangyuxin/gfastlanes/FastLanes/galp/data/system_rgbnomore/e2e_v3
TRAIN="$E2E/training_manifests_official_v3/train.json"
VAL="$E2E/training_manifests_official_v3/val.json"
GALP_TRAIN="$E2E/compact_v3_tiled_z32_rgbnomore512_train/manifest.bin"
GALP_VAL="$E2E/compact_v3_tiled_z32_rgbnomore512/manifest.bin"

CUDA_VISIBLE_DEVICES=0 "$PY" galp/benchmarks/system_rgbnomore/training/run.py \
  --enabled-pipelines galp,rgbnomore,dali,pytorch \
  --required-comparison-groups dct,rgb \
  --execution-mode runtime --phase smoke \
  --train-manifest "$TRAIN" --val-manifest "$VAL" \
  --galp-manifest "$GALP_TRAIN" --galp-validation-manifest "$GALP_VAL" \
  --expected-manifest-version 3 \
  --expected-physical-layout image-major-vector-rowgroups \
  --expected-spatial-order tiled-z32 \
  --expected-image-count 1281167 --expected-validation-image-count 50000 \
  --device cuda:0 --batch-size 64 --workers 4 \
  --warmup-steps 3 --measured-steps 10 --repeats 3 \
  --output-dir /tmp/galp-training-runtime-smoke
```

`smoke` 默认仍执行一次；显式传入 `--repeats N` 会真正执行并保存 N 个独立
repeat。v2/v3 acceptance 至少要求三次，单次 smoke 不再足以支持相对吞吐结论。

## 4. 四管线 GPU smoke

```bash
PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python

CUDA_VISIBLE_DEVICES=0 "$PY" \
  galp/benchmarks/system_rgbnomore/training/run.py \
  --enabled-pipelines galp,rgbnomore,dali,pytorch \
  --required-comparison-groups dct,rgb \
  --phase smoke \
  --train-manifest /path/to/train.json \
  --val-manifest /path/to/val.json \
  --galp-manifest /path/to/train/manifest.bin \
  --galp-validation-manifest /path/to/validation/manifest.bin \
  --device cuda:0 \
  --batch-size 64 \
  --workers 4 \
  --warmup-steps 10 \
  --measured-steps 100 \
  --output-dir /tmp/galp-training-smoke
```

GALP 正式运行要求已有 `manifest.bin.payload_fingerprints.json`。仅在数据准备或显式刷新时添加 `--refresh-galp-payload-fingerprints`；正式 measured run 复用并校验该缓存，不隐式重哈希全部 FLS shard。

默认初始化模式是 `random`。同域 pair 从同一个已落盘、带 SHA-256 的初始 model/optimizer/scheduler/RNG 状态克隆；first-step 语义 probe 在任何 warmup 前运行，不污染正式 repeat。

## 5. Step 性能和短收敛

正式 step 测试默认运行 5 个 repeat，保留 repeat 0 但只聚合 repeat 1–4，并独立报告 throughput CV 性能门。每个 repeat 在 warmup 前恢复模型、优化器、scheduler、RNG、增强状态和样本游标。

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" \
  galp/benchmarks/system_rgbnomore/training/run.py \
  --enabled-pipelines galp,rgbnomore,dali,pytorch \
  --required-comparison-groups dct,rgb \
  --phase step \
  --execution-mode runtime \
  --train-manifest /path/to/train.json \
  --val-manifest /path/to/val.json \
  --galp-manifest /path/to/train/manifest.bin \
  --galp-validation-manifest /path/to/validation/manifest.bin \
  --device cuda:0 \
  --batch-size 64 \
  --warmup-steps 10 \
  --measured-steps 500 \
  --repeats 5 \
  --output-dir /tmp/galp-training-step
```

短收敛实验会明确标为 `from_scratch_short_convergence`、`fine_tuning` 或 `resumed_training`，并声明不能替代完整 ImageNet 最终精度：

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" \
  galp/benchmarks/system_rgbnomore/training/run.py \
  --pipeline pytorch \
  --phase convergence \
  --seeds 11997733,11997734,11997735 \
  --train-steps 10000 \
  --eval-interval 1000 \
  --train-manifest /path/to/train.json \
  --val-manifest /path/to/val.json \
  --device cuda:0 \
  --output-dir /tmp/pytorch-short-convergence
```

正式性能运行前先完成上面的 10-step stability smoke。不要直接从 `--phase all` 开始。runtime artifact 会记录 native pipeline ownership、producer busy、consumer wait、完成后的 native read/decode/transform、同步原因、event-derived compute-only upper bound、cold repeat 0、hot repeat 1--4 与 CV。compute-only 数值只是诊断上界，不是第五条 pipeline，也不能替代 end-to-end images/s。

历史 K=2/4/8 sweep 已删除：这些值不能改变冻结的 native runtime policy，继续比较只会把噪声误标为配置收益。

## 6. 初始化和精确恢复

- `--init-mode random`：不接受 checkpoint。
- `--init-mode weights`：使用 `--rgb-init-checkpoint` 和/或 `--dct-init-checkpoint`，严格加载 model state dict；该模式标为 fine-tuning。
- `--init-mode full-checkpoint`：严格要求架构、输入域、完整模型配置、optimizer/scheduler recipe、scheduler horizon、model/optimizer/scheduler/RNG、global step、epoch、无状态增强说明和样本游标。recipe 不匹配会在加载 state 前失败；游标必须位于已完成 batch 边界。
- `--resume-run /existing/output`：恢复未完成的 pipeline。它从原 contract 恢复所有运行参数并复用已审计的初始状态文件，不重新生成或覆盖初始状态。

## 7. 产物和状态

输出至少包含：

- `contract.json`、`initial_states.json`、`initial_state_<domain>_seed<seed>.pt`；
- `manifest_preflight.json` 与 `manifest_preflight_validation.json`（启用 GALP 时）；
- `pipeline_<name>.json`、`semantic_comparison.json`；
- `pipeline_progress_<name>.json`、`first_step_probe_<name>.pt`；
- `sample_order.json`、`augmentation_contract.json`、`repeat_resets.json`；
- `training_curves.json`、最终/最佳 full checkpoint（收敛 phase）；
- `commands.json`、`run_metadata.json`、`artifact_hashes.json`；
- `results.json`、`validation.json`。

每条 pipeline 和 comparison group 都分别记录 `correctness`、`semantic`、`performance`、`convergence`、`artifact` 和 `overall`。未启用的 pipeline 必须是 `not_run`；性能 CV 失败不会改写数值 correctness 状态。

每个 smoke/step repeat 还固定输出跨布局可比较的 `common_statistics`；GALP
原生计数器位于可选的 `native_execution_stats` 命名空间，缺失或新增字段不参与正确性判定。
独立 validator 还会校验公共统计数值有限且非负，并核对
`samples == processed_images`、`batches == measured_steps`、吞吐/墙钟时间副本、
loss summary count、GALP queue peak 不超过硬容量以及 CUDA peak memory 非空。

独立复核命令：

```bash
"$PY" galp/benchmarks/system_rgbnomore/training/validate.py \
  /tmp/galp-training-step --no-write
```

## 8. 公平性边界

训练增强由 `(seed, epoch, logical_sample_id)` 唯一决定，不使用 worker RNG。DCT crop 固定 1:1 且按 16 个源像素对齐；从源像素 crop 到各 JPEG 分量 padded block grid 的换算与 native reader 一致：起点向下取整、终点向上取整，然后各分量独立 resize。水平翻转使用 block 列反序并对奇数水平频率系数取反。DCT 默认绝对误差门允许一个整数系数在 `[-1, 1]` 归一化后的步长（`1/1020`），覆盖 native/CPU resize 在整数舍入边界上的单步差异，但不会掩盖 crop/block 选择错误。RGB decoder/interpolation 的可接受差异以 warning/failure 两级门报告。DCT 与 RGB 不声明 tensor 或权重等价，只比较各自同域 pair。

本 runner 的默认 recipe 是为了四管线可审计、公平计时而固定的
RRC+hflip、无 Mixup/RandAugment、batch=64、warmup=500、统一 AdamW 配置；它不等同于
RGB-no-more 发布训练脚本的完整 300-epoch recipe。后者还包含 global batch=1024、
10,000-step warmup、gradient clipping、Mixup、域专用 RandAugment、域专用学习率和
从 ImageNet-train 划出的 1% minival。使用本指南的官方 validation split只能声明
“输入数据/模型/基础 crop 语义一致的四路训练 benchmark”，不能声明逐项复现发布
checkpoint 的训练 recipe 或最终精度。

## 9. manifest v2/v3 性能验收

v2 和 v3 必须分别运行到独立目录；除 GALP manifest 版本、物理布局及其指纹外，两个 `contract.json` 应完全一致。正式对比使用 `runtime` mode、相同 batch/worker/prefetch、相同 warmup/measured steps、相同训练/验证 manifest 和 seed。

GALP repeat 现在分别输出 `native_execution_stats_by_phase.warmup` 与 `.measured`。`galp_native_*` allocation 字段是进程全局累计快照，不能按 batch 求和；`native_allocation_stability` 使用 warmup high-water 到全部已消费 batch high-water 的 CUDA allocation 增量，并独立合计 measured batch 的 output-arena、chunk-arena、compact-buffer growth 和 pageable fallback。output/chunk arena 还分别报告计划容量、实际请求、当前容量、增长次数和增长字节。只有该对象的 `verifiable` 和 `stable_after_warmup` 都为 true，才可声称 warmup 后稳定复用。

相对性能门还要求 v2/v3 各至少三个 smoke repeat，并要求两边 compute-only
upper-bound 的均值相差不超过 10%、各自 CV 不超过 10%。条件不一致时
`v3_not_slower_than_v2` 保持 `unverified`，不会把 GPU 降频或后台竞争误判为布局性能。

下面的命令把独立 planner、受控 crop reader 和 v2/v3 训练结果合并为逐门的机器可读报告。缺失的 GPU 结果保持 `unverified`，不会被当作通过：

```bash
GALP_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0 \
  ./build/galp/tests/galp_tests \
  --gtest_filter='JpegDct.ManifestV3PlanlessMatchesLegacyAcrossRaggedShardsAndSampling:JpegDct.PlanlessDeviceMatchesLegacyAcrossGeneralityMatrix' \
  --gtest_output=xml:/tmp/galp-v3-planless-gpu-tests.xml

python3 galp/benchmarks/system_rgbnomore/training/v3_acceptance.py \
  --baseline-dir /tmp/galp-training-v3-baseline \
  --v2-dir /tmp/galp-training-v2-final \
  --v3-dir /tmp/galp-training-v3-final \
  --planner-v2 /tmp/galp-planner-v2.json \
  --planner-v3 /tmp/galp-planner-v3.json \
  --reader-v3 /tmp/galp-reader-v3.json \
  --gpu-gtest-xml /tmp/galp-v3-planless-gpu-tests.xml \
  --output /tmp/galp-training-v3-acceptance.json
```

返回码 `0` 表示所有 blocking gates 通过，`1` 表示已有证据违反 blocking gate，`2` 表示仍缺 required evidence。`--allow-incomplete` 只允许 incomplete 报告返回 `0`，不会掩盖已失败的门。报告不会相加可能重叠的 planner/read/decode/transform stage 时间，也明确禁止把 GALP 与 DALI 的不同模型路径解释为纯 codec 差异。
