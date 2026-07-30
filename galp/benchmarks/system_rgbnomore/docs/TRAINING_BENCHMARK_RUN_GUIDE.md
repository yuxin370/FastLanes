# RGB-no-more 四管线训练 Benchmark 运行指南

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
  --galp-manifest /path/to/manifest.bin \
  --output-dir /tmp/galp-training-contract \
  --dry-run-contract
```

## 3. execution mode

`--execution-mode audit|runtime` 是不可变 contract 字段，默认是 `audit`。

- `audit` 保留逐 stage 同步、逐 step 梯度扫描、首个 measured step 参数快照等完整审计行为，适合语义核验，不能当作生产吞吐。
- `runtime` 仍先在 fresh clone 上执行同一个 audit first-step probe，但 probe 不进入 warmup 或 measured window。正式 measured loop 不调用逐参数统计或逐 step `.item()`；CUDA wall-clock 只在 measured region 前后各同步一次，stage 用 CUDA event 在 region 结束后解析。
- `--resume-run` 只能恢复相同 mode；显式传入不同 mode 会立即失败。

GALP 的 `--workers N` 明确表示一个有序 batch producer 内部的 native rowgroup-prefetch worker 数，不表示 N 个 batch producer。`N=0` 会失败。`--prefetch-depth D` 对应最多 `D+1` 个 in-flight batch（当前 batch 加 D 个 ahead batch），队列保持 FIFO、施加硬 backpressure，并在 close 时 cancel 或 drain 全部未消费任务。

下列短命令使用当前正式数据路径：

```bash
PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
TRAIN=/tmp/galp-training-manifests/train.json
VAL=/tmp/galp-training-manifests/val.json
GALP=/home/tangyuxin/gfastlanes/FastLanes/galp/data/imagedataset_dct/ImageNet-train/manifest.bin

CUDA_VISIBLE_DEVICES=0 "$PY" galp/benchmarks/system_rgbnomore/training/run.py \
  --enabled-pipelines galp,rgbnomore,dali,pytorch \
  --required-comparison-groups dct,rgb \
  --execution-mode runtime --phase smoke \
  --train-manifest "$TRAIN" --val-manifest "$VAL" --galp-manifest "$GALP" \
  --device cuda:0 --batch-size 64 --workers 4 --prefetch-depth 2 \
  --warmup-steps 3 --measured-steps 10 \
  --output-dir /tmp/galp-training-runtime-smoke
```

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
  --galp-manifest /path/to/manifest.bin \
  --device cuda:0 \
  --batch-size 64 \
  --workers 4 \
  --prefetch-depth 2 \
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
  --galp-manifest /path/to/manifest.bin \
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

正式性能运行前先完成上面的 10-step stability smoke。不要直接从 `--phase all` 开始。runtime artifact 会记录 queue hit/miss、producer busy、consumer wait、native read/decode/projection、同步原因、event-derived compute-only upper bound、cold repeat 0、hot repeat 1--4 与 CV。compute-only 数值只是诊断上界，不是第五条 pipeline，也不能替代 end-to-end images/s。

建议依次用 `--prefetch-depth 0`、`2`、`4` 做三个独立短目录；不要在同一目录跨 depth 或跨 mode resume。

## 6. 初始化和精确恢复

- `--init-mode random`：不接受 checkpoint。
- `--init-mode weights`：使用 `--rgb-init-checkpoint` 和/或 `--dct-init-checkpoint`，严格加载 model state dict；该模式标为 fine-tuning。
- `--init-mode full-checkpoint`：严格要求架构、输入域、完整模型配置、optimizer/scheduler recipe、scheduler horizon、model/optimizer/scheduler/RNG、global step、epoch、无状态增强说明和样本游标。recipe 不匹配会在加载 state 前失败；游标必须位于已完成 batch 边界。
- `--resume-run /existing/output`：恢复未完成的 pipeline。它从原 contract 恢复所有运行参数并复用已审计的初始状态文件，不重新生成或覆盖初始状态。

## 7. 产物和状态

输出至少包含：

- `contract.json`、`initial_states.json`、`initial_state_<domain>_seed<seed>.pt`；
- `pipeline_<name>.json`、`semantic_comparison.json`；
- `pipeline_progress_<name>.json`、`first_step_probe_<name>.pt`；
- `sample_order.json`、`augmentation_contract.json`、`repeat_resets.json`；
- `training_curves.json`、最终/最佳 full checkpoint（收敛 phase）；
- `commands.json`、`run_metadata.json`、`artifact_hashes.json`；
- `results.json`、`validation.json`。

每条 pipeline 和 comparison group 都分别记录 `correctness`、`semantic`、`performance`、`convergence`、`artifact` 和 `overall`。未启用的 pipeline 必须是 `not_run`；性能 CV 失败不会改写数值 correctness 状态。

独立复核命令：

```bash
"$PY" galp/benchmarks/system_rgbnomore/training/validate.py \
  /tmp/galp-training-step --no-write
```

## 8. 公平性边界

训练增强由 `(seed, epoch, logical_sample_id)` 唯一决定，不使用 worker RNG。DCT crop 固定 1:1 且按 16 个源像素对齐；水平翻转使用 block 列反序并对奇数水平频率系数取反。RGB decoder/interpolation 的可接受差异以 warning/failure 两级门报告。DCT 与 RGB 不声明 tensor 或权重等价，只比较各自同域 pair。
