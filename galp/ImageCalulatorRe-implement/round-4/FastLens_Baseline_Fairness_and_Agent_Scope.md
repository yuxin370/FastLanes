# FastLens 基线公平性与 agent 任务范围修订

本修订依据用户最新明确的实验目的，对上一轮“共用 GPU 后处理”的优先级和公平性表述作更正。它替代第三版原始报告与复核包中将共同后处理列为主线必做项的表述；原有源码检查、数值证据、调度反例和内存计量发现保留。

## 核心决定

**IC 主 baseline 保留其 CPU 数据增强方式是公平的，不需要为了公平性接入 FastLens 的 GPU 增强。** 相同任务、源数据、模型、checkpoint、预处理/增强语义以及硬件资源条件，可以由不同 CPU/GPU 分工实现；系统比较允许这些实现差异。GPU 数据增强也不等于 GPU 模型推理，二者应分别说明。

**复用 FastLens 处理的版本作为单独消融，用于隔离数据格式及其读取机制的影响。** 共同后处理不再是主 baseline 的实现完成条件，也不应把未使用 GPU 增强列为缺陷。

## 实验分组

| 分组 | 执行方式 | 可以支持的结论 |
|---|---|---|
| IC-style matched 系统基线 | 接入 IC-style 存储，保留 CPU 预处理/增强执行方式；模型推理使用同一 checkpoint | 相同任务和输入语义下，完整输入管线的差异 |
| FastLens 系统 | 使用自身优化的读取、GPU 处理与融合路径 | 完整系统收益，包含 CPU/GPU 分工和流水线收益 |
| 共同处理消融 | 将 IC/FS reader 接入同一后处理实现，模型与执行边界一致 | 在共同处理条件下，存储格式及读取机制的差异 |

IC/FS 均应按各自公开方案说明和实验适配记录执行分工；不能由“IC 没有 GPU 增强”自动推断 FS 的每个实现细节。

主比较不要求每个系统执行相同数量的 CPU 或 GPU 算子，不要求为 baseline 补上 FastLens 的专用优化；应完整计入各自的 I/O、解压、预处理、数据传输、同步和模型计算成本。独立调优可以做，但须有一致的资源上限与明确预算，不为 baseline 设置无依据的低效参数。

已有 checkpoint 的推理实验主要对齐确定性的 crop、resize、通道选择和 normalization；若不启用测试时增强，训练 RandAugment/Mixup 不应进入 inference 路径。训练扩展才需要对齐增强规则与随机决定。语义一致不要求使用同一设备或同一实现。

## 第三版当前实现的准确分类

当前训练接入已经使用共享 GPU 增强：

- CNN：StoreDctDataset 在 CPU 读取 q/Q 并完成几何处理；JpegPool 将输入移到 CUDA 后调用共享 ProjectedDctTrainingAugment，包含 GPU 增强和 Mixup。
- Transformer：CPU worker 完成 q/Q、crop/resize 等；训练 runner 将 batch 移到目标 device 后执行共享 RandAugment/Mixup。目标是 CUDA 时，这部分张量运算在 GPU。

因此当前版本属于 **IC/FS 存储适配 + CPU 几何 + 共享 GPU 增强**。它可以作为已实现的共享部分处理实验保留；不能描述成原生 IC 的完整 CPU 增强管线。另一方面，IC/FS 几何仍在 CPU、FastLens 几何在 native CUDA，所以它还不是完全隔离数据格式的消融。

### 源码定位

- [CNN training_jpeg.py](sandbox:/workspace/scratch/22ddd5f2b436/review_round3_20261010/submission/current_repo/galp/benchmarks/dct_models/training_jpeg.py)：第59–84行 CPU 几何；第147–149行先 x.cuda 再 augment.apply；第182行创建共享 native augmentation。
- [Transformer 训练 runner](sandbox:/workspace/scratch/22ddd5f2b436/review_round3_20261010/submission/current_repo/galp/benchmarks/training_pls/train.py)：第2117–2135行移动 batch 后执行 RandAugment/Mixup。
- [Transformer CPU pipeline](sandbox:/workspace/scratch/22ddd5f2b436/review_round3_20261010/submission/current_repo/galp/benchmarks/system_rgbnomore/training/pipeline.py)：第513–594行 store 与 CPU transform；第646行 on_device=False。

以上是已有源码的只读核对；本次没有修改产品实现，也没有重新运行 GPU 或性能实验。

## agent 任务调整

### 主线任务

1. 明确每个结果的实验类型和执行分工。IC-style matched 主 baseline 保留 CPU 增强方式；已经使用共享 GPU 增强的现有结果注明为共享部分处理的适配版本。共享模型加载、推理入口和结果统计可以保留，不由此声称整个输入管线也是原方案。
2. 优先完成已有 checkpoint 的正式推理验证与性能测量。固定源内容、样本、输入语义和模型契约，在相同全局资源条件下测量各系统自己的完整路径。推理时不无故加入训练增强。
3. 保留并处理第三版审查中与实验类型无关的问题：projected 完成映射、split-source transient 计量、当前源内容绑定、源码和二进制身份。按报告所述分支做定向验证。
4. 若扩展到训练系统比较，再落实同一增强语义的 CPU IC baseline 路径；FastLens 保留自己的 GPU 增强。无需以移植 FastLens GPU 增强作为公平性门槛。

### 可选的数据格式消融

将原“任务 D 做共同后处理的存储对照”改为可选实验，目的明确为隔离存储格式及读取机制。只有在这个分组中才统一后处理实现与 device 交接边界；每条路径实际发生的 packing、dtype/layout 转换、同步和 H2D 都纳入成本。原任务 D 中的当前源内容绑定仍是主线要求，应放在正式启动/preflight 中。

可以把结果命名为“IC-style + shared processing”或“IC-format ablation”，并明确复用的 FastLens 模块。单纯共用增强、但几何执行不同，只能称为共享部分处理，不能自动宣称已经隔离格式。

完整 IC-native 搜索依然是可选的另一类实验；它不是本次公平性修订后的新增主线要求。

## 可用于论文方法说明的表述

主系统比较保留各方案的预处理与数据增强执行方式，IC-style matched baseline 使用 CPU 处理，FastLens 使用其优化的 GPU 路径。各方案遵循相同任务、源数据、模型输入语义和资源条件，并计入完整输入管线成本。为分析存储格式及读取机制的独立影响，我们另设共同处理消融，将 IC/FS 的存储读取接入相同后处理流程。该消融中的共享 GPU 处理是实验控制，不视为原 IC 的实现能力。
