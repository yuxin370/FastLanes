# GALP Image ID 顺序端到端实验

这个目录专门回答两个问题：

1. GALP 当前 `component -> spatial block -> image` 物理布局下，batch 内 image ID
   全局随机与 rowgroup 对齐连续时，完整 JPEG-Ti 推理吞吐相差多少；
2. 连续 batch 是否改变推理结果，以及它造成的类别聚集是否会让训练顺序敏感。

2026-07-13 的 RTX 4090 正式实测、统计检验和原始 artifact 摘要见
[`RESULTS_2026-07-13_RTX4090.md`](RESULTS_2026-07-13_RTX4090.md)。

## 三个性能条件

正式运行固定 `seed=11997733`、`batch=64`、`5` 个 warmup batch、`20` 个
measurement batch，并交错执行：

| 条件 | 定义 | 用途 |
| --- | --- | --- |
| `current_random` | 对全部 48,615 个受支持 ID 做与现 benchmark 完全相同的全局 shuffle，取前 1,600 张 | 实际迁移参考 |
| `paired_scattered` | 选出 25 个 rowgroup 对齐连续 batch，再分别在 warmup/measured cohort 内打散 | 同样本因果对照 |
| `contiguous` | 与 `paired_scattered` 完全相同的图片集合，但每个 batch 是一个 64-image 物理 rowgroup | 同样本因果对照 |

`paired_scattered` 和 `contiguous` 的 warmup 集合、measured 集合分别严格相同，
不会把图片难度、尺寸或类别构成差异算成布局收益。`current_random` 与
`contiguous` 的差值更接近从当前 benchmark 直接迁移时看到的数值，但由于图片集合不同，
不能单独作为因果结论。

每个 batch 的计时包含 Direct-DCT 读取、解压、融合预处理、JPEG-Ti forward、
top-1/top-5 统计和 CUDA 同步；模型/reader 创建及 warmup 不计时。repeat 顺序轮转，
正式汇总排除 repeat 0。

## 精度协议

推理语义保存 measured cohort 的全部 logits，并按 `galp_image_id` 重新对齐后比较：

- top-1/top-5 accuracy delta；
- top-1 预测与 top-5 集合一致率；
- logit max/mean absolute difference 和 cosine similarity。

这能直接检测 batch 组成/顺序是否改变 eval 模式下的模型输出。

训练部分是一个受控的短程 fine-tune 顺序敏感性探针：从同一 DCT checkpoint 出发，
在完全相同的 transformed-DCT 样本上比较“每 epoch 全局 shuffle”和“只 shuffle 连续
batch、batch 内保持连续”两种顺序，并在同一个未参与训练的全局随机集合上评价。
它量化短程敏感性和 batch 类别熵，不冒充 300 epoch ImageNet 全量重训结论。

## 运行

正式配置：

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/examples/run_image_order_benchmark.py \
  --output-dir /tmp/galp-image-order-e2e
```

只生成 selection、检查 48,615 eligibility 和连续 rowgroup 候选，不启动 GPU：

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/examples/run_image_order_benchmark.py \
  --output-dir /tmp/galp-image-order-selection \
  --selection-only
```

可用 `--skip-training` 只运行端到端性能和推理精度部分。

## 输出

- `contract.json`：输入文件 SHA-256、环境、参数和 git 状态；
- `selection.json`：全部 image ID、配对 invariant、物理 rowgroup 与类别统计；
- `performance.json` / `performance.csv`：每 condition/repeat 的原始结果；
- `semantic_*.npz`：按 measured batch 捕获的 image ID、label 和 logits；
- `training_probe.json`：逐 seed/epoch 的训练与 held-out 评价；
- `training_order_stats.json`：全量 ImageNet train 连续/随机 batch 的类别混合统计；
- `summary.json` / `report.md`：最终量化结论与解释边界。
