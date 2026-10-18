# IC/FS 实验执行计划（2026-10-15 修订）

执行原则：先覆盖每个配置的一次有效结果，再统一补第二、第三次。正式推理仍为完整 50K；matched training 按原图采用 65,536 张不同图片、64 次参数更新的窗口。已通过的内容验证按身份复用，性能重复不重新全量解码审计。已有有效且配置相同的结果直接抵扣；旧 FastLens 结果保留，但受新运行路径影响的结果不进入当前汇总。

## 与原图的关系

已核对 `galp/docs/paper/scripts/plot_matched_dct.py` 及其结果来源。原推理图采用完整 50K 单次遍历，模型 warmup 20 batches 在计时外；读取器创建、读取、处理、H2D、模型和预测收集在计时内。不能用 60 秒预热、180 秒回放的吞吐直接补入该图。

原训练图的实际记录是 **65,536 张不同图片的训练窗口、64 次参数更新**，不是完整训练 epoch。按用户后续确认，新增 matched training 采用同一窗口，不安排完整训练集建库或 full-epoch 测量。旧 PDF 的 full-epoch 表述须改为 training window；配置相同的新结果可补入原训练图。

## 实验矩阵

| 实验 | 数据及工作量 | 模型 | 路径 | 首轮及后续 |
|---|---|---|---|---|
| all64 matched 推理 | ImageNet-1K 验证集 50,000 张，1,000 类；匹配原图单次遍历协议 | 首批 MobileNet24、ResNet24、ViT-Ti；随后 MobileNet32、ResNet64、eFUN、SwinV2 | CPU-DCT、最新 FastLens、IC-shared、IC-batched、FS | 每配置先一次；原 CPU/FastLens 仅在配置和版本兼容时复用 |
| all64 matched 窗口训练 | 原图 B6/epoch=2 的 65,536 张不同图片；microbatch=64、accumulation=16，64 次参数更新 | 首批 MobileNet24、ResNet24、ViT-Ti | 同上；共享 GPU 处理的 IC/FS 明确标注 | 32 次预热更新后恢复模型/优化器/调度/decay/RNG；同一 seeded 初态、顺序和 basic crop/flip；先一次，不声明收敛精度 |
| Partial-DCT 推理 | 同一完整 50K；源量化系数置零，保持模型输入形状 | MobileNet24、ResNet24、ViT-Ti | CPU-DCT、FastLens、IC-StaticK shared、IC-batched-StaticK、FS | D4/D6/D8，即每块 10/21/36 系数；45 配置各一次，再补重复 |
| compact 训练吞吐 | 已有 8,194 张真实训练图片；固定裁剪、无随机增强、真实参数更新；持续窗口，不称 full epoch | 普通三通道 ResNet50、MobileNetV2、ViT-Ti | CPU-DCT、IC-prepared shared、IC-prepared batched、FS-prefetched、最新 FastLens | D3 为 `3×84×84`，D15 为 `3×224×224`；复用 ResNet50/4090/D3 的三次结果，新配置各一次 |

推理图片源：`/home/tangyuxin/tmp/dct-50k-20261013-152452/contract/source.json`。源 JPEG 为 RGB 512×512、BILINEAR resize、quality 75、4:2:0；不转换成 B32/Q50。训练窗口从完整训练集的 premixed `ordered_mapping.csv` 按原图 B6 调度选择，建库包括实际预取触及的额外图片；这些额外图片不计入 65,536 张测量样本。不能按 manifest 的任意前缀替代原窗口顺序。CNN 训练为 BF16 autocast，ViT-Ti 为 FP32；保持原图 compile/no-CUDA-Graph、优化器与预热缓存策略。

Partial-DCT 的 FastLens 和 FS 复用完整系数存储，不重新压缩。IC 两种布局分别构建 D4/D6/D8 剪枝存储，转换和容量单列。旧 IC-Full 结果仅保留为完整存储后清零的对照。各预算分别核对输入、logits 和预测；不同预算允许精度变化。D10/D12 不在本轮范围内；D15 仅为已有 all64 参考。

Compact 的 IC-prepared 存储已经完成固定几何处理的拼接图，归一化保留在线；离线成本必须报告。D3/D15 是同模型尺寸对照，不扩展选频推理预算。ViT-Ti 的 84×84 输入只为 patch 对齐补零到 96×96，使用 16×16 patch；224×224 保持原尺寸。位置编码规则各路径一致，补零和位置编码处理计入训练时间，不将图像放大到 224。

## 执行顺序

1. 核对原图配置及已有结果身份，抵扣兼容结果；保留当前已启动的源验证。
2. 并行准备 StaticK 六套存储、原图训练窗口 IC/IC-b/FS、compact D15 prepared 数据及新模型入口。性能测量避开建库的 CPU、存储和内存竞争。
3. H100 首批三个模型的 all64 推理、同原图窗口训练各一次。
4. 三模型 D4/D6/D8 推理各一次。
5. Compact 新模型和尺寸对照各一次；4090 在资源可隔离时并行。
6. 补 all64 推理剩余四模型及跨硬件缺项，各一次。
7. 所有首轮配置覆盖后，才统一补第二、第三次。

H100 以 UUID `GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76` 选择；4090 以 `GPU-40c637bd-acf5-ea1a-0df8-617138228467` 选择。GPU 编号变化不影响选择。沿用实际 MPS 客户端检查和连续空闲 60 秒规则，不杀 MPS Server，不改变 Compute Mode。

## 汇总口径

一次结果明确标为“单次测量”；三次有效结果齐备后报告中位数及离散程度。未完成重复的配置不与三次中位数混称。OOM、资源竞争或不完整运行保留日志且不计有效次数。

随主实验收集 all64、StaticK、prepared 的实际存储、转换时间、构建并行度及内存成本；选一个 CNN 和一个 Transformer 单独 profiling，报告真实读取、解压、H2D 和模型阶段。Profiling 时间不作为主表吞吐。FS 冷缓存、共同后处理、额外布局消融排在本轮覆盖之后。
