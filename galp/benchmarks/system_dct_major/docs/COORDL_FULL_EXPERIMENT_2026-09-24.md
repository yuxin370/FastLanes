# CoorDL 七管线 50K 实验结果（2026-09-24）

已运行完整七管线的 feature extraction 和 evaluation，各 50,000 张图、5 轮，共 350 万次图像推理；另完成四组系数语义检查、两组 smoke、四组模型计算上限测试。对首测稳定性失败的三项测量各做一次同配置复测，额外推理 75 万张，保留全部首测结果。

**结论：CoorDL 已可运行，热轮吞吐与 DALI、FFCV 接近。整套实验未全部通过验收：首测有吞吐稳定性失败，K32 全量 raw-mask oracle 预测一致率也未达标。不能将本次结果描述为所有语义检查通过的正式加速结论。**

## 配置与口径

- GPU：RTX 4090，CUDA_VISIBLE_DEVICES=0；运行前无其他 GPU 计算进程。
- 数据：固定 512×512 ImageNet val，50,000 张、49 个 GALP shard；batch size 50、8 workers、顺序读取、不丢尾批。
- 模型：RGB-no-more ViT-Tiny，RGB checkpoint `imgnetRGBViTTi_ep300_74.1.pth`，DCT checkpoint `imgnetDCTViTTi_ep300_75.1.pth`；FP32。
- GALP K64 与 K32 共享 DCT 模型和 crop profile。K32 在解量化和频率变换前保留 zigzag 前 32 个原始系数，其余置零；底层存储和模型输入仍为 dense-64。
- RGB 使用 512 输入的 224 中心裁剪；DCT 使用 448 中心区域缩至 224 的现有 profile。RGB/DCT checkpoint 和预处理不同，跨域仅按已有部署口径比较，不能用两域准确率差异判断 loader 正确性。
- CoorDL：上游 `bcde72da21781aab0661eacdedbbd1dcca5f4cfe` + 本目录兼容补丁，DALI 0.20.0dev、Python 3.11、CUDA toolkit 12.8；普通 DALI 2.2.0 独立运行。
- CoorDL cache_size=25,000 JPEG，跨 5 轮保留 reader；上游日志显示缓存进入已填充状态，完成后清理本次缓存。FFCV 复用同一份离线 raw `.beton`，转换时间不计入吞吐。
- 热轮为后 4 轮完整 batch-loop 吞吐的算术平均；CV 为总体变异系数。首轮另按进程启动口径统计。稳定性门槛：热轮 CV ≤5%，首尾漂移 ≤5%。
- 采用既有 application-overlapped 协议，未清空系统页缓存。RGB 管线记录的首轮/热轮 storage_read_bytes 均为 0；这里不衡量磁盘受限条件下的缓存收益。

## 吞吐与准确率

单位：张/秒。`*` 表示该项引用唯一一次复测；其余为首测。Top-1/Top-5 为完整 50K evaluation，五轮一致。

| Pipeline | Feature 热轮 | CV | Evaluation 热轮 | CV | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| GALP K64 | 4,777.5* | 0.08% | 4,676.5* | 0.09% | 75.140% | 92.446% |
| GALP K32 | 2,099.4 | 0.25% | 4,702.2 | 0.06% | 75.128% | 92.476% |
| PyTorch DCT | 1,770.4 | 0.62% | 1,816.2 | 0.94% | 75.140% | 92.446% |
| DALI | 5,037.1* | 0.39% | 4,932.3 | 0.20% | 66.622% | 86.640% |
| FFCV | 5,028.2 | 0.31% | 4,965.8 | 0.70% | 66.648% | 86.644% |
| PyTorch RGB | 2,674.0 | 4.22% | 2,735.3 | 3.49% | 66.666% | 86.652% |
| CoorDL | 5,067.0 | 0.14% | 4,946.6 | 0.15% | 66.658% | 86.640% |

首测与复测存在明显时段差异。K32 feature 首测虽在组内稳定，但与 K64 复测不处于同一运行时段；不能用这两个数值直接推导系数下推的因果收益或退化。

| CoorDL / RGB baseline | Feature 吞吐比 | Evaluation 吞吐比 |
|---|---:|---:|
| DALI | 1.006× | 1.003× |
| FFCV | 1.008× | 0.996× |
| PyTorch RGB | 1.895× | 1.808× |

CoorDL 相对 DALI/FFCV 的差异均不足 1%，不能据此宣称显著胜出；相对现有 PyTorch RGB 的观测吞吐约为 1.90×（feature）和 1.81×（evaluation）。这些是本配置的端到端比值，不能单独归因于 CoorDL 的缓存策略。

## 首测与复测

| Workload / Pipeline | 首测热轮 | 首测 CV | 首测首尾漂移 | 复测热轮 | 复测 CV | 复测首尾漂移 |
|---|---:|---:|---:|---:|---:|---:|
| feature / GALP K64 | 2,337.1 | 10.96% | 21.65% | 4,777.5 | 0.08% | 0.19% |
| feature / DALI | 3,481.9 | 40.95% | 142.10% | 5,037.1 | 0.39% | 0.98% |
| evaluation / GALP K64 | 3,542.3 | 31.16% | 49.30% | 4,676.5 | 0.09% | 0.23% |

三项复测都通过原有稳定性门槛。原始 feature/evaluation 的 `results.json` 仍保留 `ok=false`；未覆盖或修改首测结果，未调整阈值。

## 正确性检查

- K64/K32/K16/list:0,2,5,9 的 32 样本 raw-mask 检查通过；两个 workload 的 K64 vs PyTorch DCT 严格语义检查通过。
- **K32 全量 oracle 检查未通过**：Top-1 预测一致率 99.814%（93/50,000 个预测不同），低于 99.9% 门槛。native Top-1=75.128%，oracle=75.122%，准确率差 +0.006 个百分点，在 0.05 个百分点门槛内。总体准确率接近不能代替逐样本预测一致性。
- CoorDL vs PyTorch RGB：32 样本特征余弦 0.999129，logits 余弦 0.998967；后者低于 0.999。像素级诊断未通过。现有 RGB 对比将这些差异记为 diagnostic，不声明逐像素或 logits 严格等价。
- DALI 的像素级诊断同样未通过；FFCV 的 RGB 诊断通过。所有管线的完整样本顺序检查通过。

## 模型计算上限

同一 GPU、batch size 50，预热 20 steps、计时 300 steps；不包含输入加载。

| 输入域 | Workload | 张/秒 |
|---|---|---:|
| dct | feature-extraction | 5,545.9 |
| rgb | feature-extraction | 5,510.2 |
| dct | evaluation | 5,541.6 |
| rgb | evaluation | 5,505.7 |

RGB 三种加速 loader 的 evaluation 吞吐约为纯 RGB 模型上限的 90%。这支持当前配置下模型计算占比较高的解释，但不是对缓存收益的隔离测量。

## 首轮进程吞吐

下表保留原始七管线实验的首轮，包含进程内初始化；不是清空操作系统缓存后的磁盘冷读。

| Pipeline | Feature 张/秒 | Evaluation 张/秒 |
|---|---:|---:|
| GALP K64 | 2,152.0 | 3,285.9 |
| GALP K32 | 1,741.2 | 3,346.4 |
| PyTorch DCT | 1,523.4 | 1,531.7 |
| DALI | 1,671.6 | 3,262.8 |
| FFCV | 2,654.9 | 2,640.4 |
| PyTorch RGB | 2,161.0 | 2,049.9 |
| CoorDL | 2,629.2 | 2,532.2 |

## 产物与复现

- [原始 suite 参数与命令](/home/tangyuxin/tmp/dct-major-coordl-full-20260924/suite_plan.json)
- [完整日志](/home/tangyuxin/tmp/dct-major-coordl-full-20260924.log)
- [Feature 首测](/home/tangyuxin/tmp/dct-major-coordl-full-20260924/06_formal_feature_extraction/results.json)
- [Evaluation 首测](/home/tangyuxin/tmp/dct-major-coordl-full-20260924/07_formal_evaluation/results.json)
- [Feature 复测](/home/tangyuxin/tmp/dct-major-coordl-full-20260924/09_stability_repeat_feature/results.json)
- [Evaluation 复测](/home/tangyuxin/tmp/dct-major-coordl-full-20260924/10_stability_repeat_evaluation/results.json)
- [系数语义检查](/home/tangyuxin/tmp/dct-major-coordl-full-20260924/01_coefficient_semantics.json)
- [GPU 监控（evaluation 中途开始，至复测结束）](/home/tangyuxin/tmp/dct-major-coordl-full-20260924/gpu-monitor.csv)
- [CoorDL 构建与运行说明](../README.md#coordl-baseline)

全量启动时修复了两个脚本入口问题：FFCV 数据目录入口使用正确的 manifest 字段及关键字参数；系数语义脚本按正式 adapter 启用 native physical 模式并提交完整 shard，只比较前 32 个样本。4 项相关回归测试通过，未改变模型、计时、数值阈值或 CoorDL 缓存算法。
