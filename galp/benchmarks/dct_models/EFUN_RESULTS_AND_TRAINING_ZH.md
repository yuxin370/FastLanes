# eFUN 家族、结果汇总与训练配置

本汇总区分作者公布的模型成绩、我们已经完成的推理实测及两轮训练实验。模型源码固定为 FUN revision `6c2b5f4a43a2b514163ff1f3f114d4feeb174d3c`。

## 1. eFUN 家族包含哪些模型

四个变体均使用 EfficientNet 风格的 MBConv 倒残差模块、SE 通道注意力和 Swish 激活，但采用各自的 stage 配置。它们不是 ResNet 或 MobileNetV2 的 DCT-24/32/64 变体。四者都接收 **192×28×28** DCT 输入：Y/Cb/Cr 各保留全部 64 个频率；S/L 的区别主要是网络内部宽度、深度和扩展比例，而非减少输入 DCT 通道。

| 模型 | 作者注册名 | 源码实际参数量 | 各 stage 的输出通道 × 模块数 | head 通道 | 作者 Top-1 | 作者 FPS |
|---|---|---:|---|---:|---:|---:|
| eFUN | `efun` | 4,233,448 | 128×3 → 160×6 → 192×1 | 1280 | 77.0% | 124 |
| eFUN-L | `efun_l` | 6,209,180 | 144×3 → 180×2 → 180×5 → 216×2 | 1280 | 78.8% | 101 |
| eFUN-S | `efun_s` | 3,394,390 | 120×3 → 140×5 → 192×1 | 1280 | 75.6% | 132 |
| eFUN-S+ | `efun_s_plus` | 2,540,728 | 96×3 → 120×4 → 192×1 | 960 | 73.3% | 145 |

参数量来自实际实例化作者四个模型。作者 FPS 使用 **V100、batch 1**，不能与下面的 RTX 4090、batch 64 测量直接比较。来源：[作者模型定义](FUN/timm/models/eFUN.py)、[作者 README](https://github.com/kfirgoldberg/FUN#pretrained-models)。README 中 S 与 S+ 的权重链接相同，尚未验证其分别对应正确架构，不能将该链接当作两个已确认的 checkpoint。

当前完成 GALP 接入和 50K 实测的是 **base eFUN**；L/S/S+ 仅核对了源码结构及参数量，没有我们的准确率、吞吐或训练结果。RGB 对照是 Torchvision **EfficientNet-B0 / IMAGENET1K_V1**，5,288,548 参数，输入 **3×224×224**；它是独立的 RGB 网络，而非 eFUN 的同构 RGB checkpoint。

## 2. 完整推理结果

共同条件：RTX 4090、相同 ImageNet-512 验证集 50,000 张、相同标签映射、batch 64、FP32、TF32 关闭。JPEG/PyTorch 使用 16 workers；RGB DALI 使用经测试选择的 4 线程、预取深度 4。DALI 使用三次交叉确认的中位数，其余为既有单次在线测量，均未强制冷缓存。

| 模型与路径 | 在线耗时（秒） | Top-1 | Top-5 | 模型 forward 累计时间（秒） |
|---|---:|---:|---:|---:|
| eFUN，作者 JPEG→DCT 参考路径 R | 41.873 | 75.428% | 92.622% | 14.273 |
| eFUN，GALP grid / pushdown off | 11.800 | 75.428% | 92.622% | 10.729 |
| eFUN，GALP projected / pushdown off | 10.730 | 75.428% | 92.622% | 10.021 |
| eFUN，GALP projected / pushdown on | 10.738 | 75.428% | 92.622% | 10.027 |
| RGB EfficientNet-B0，PyTorch/PIL | 22.338 | 76.774% | 93.238% | 12.989 |
| RGB EfficientNet-B0，DALI（4 线程 / 预取 4） | 11.733 | 76.770% | 93.262% | 10.332 |

结果根目录：[rtx4090_20260915](../../data/system_rgbnomore/e2e_v3/runs/efun/rtx4090_20260915/)。eFUN JSON 在 `full/`，原 RGB JSON 在 `rgb_efficientnet_b0/`；DALI 主结果已替换为 [交叉确认的中位运行](../../data/system_rgbnomore/e2e_v3/runs/efun/dali_tuning_20260916/inference_confirmation/w4_q4_r1/RGB_dali_50000.json)，原始结果保留。每个结果保留全部预测；阶段时间存在并行和等待，不能简单相加或相减来推导节省的工作量。

可支持的结论：

- GALP projected 相比 eFUN 的作者在线参考路径约快 **3.90×**，三条 GALP 路径的全部预测与 R 一致。
- RGB DALI 相比同 checkpoint 的 RGB PyTorch 约快 **1.90×**，预测一致率 **99.244%**，Top-1 差 **-0.004 个百分点**。DALI 与 PIL 的解码/缩放实现并非逐元素相同。
- 加入 RGB DALI 后，eFUN/GALP 的速度优势约为 **1.09×**，同时 Top-1 低 **1.342 个百分点**。这是不同架构与数据路径的权衡，不能把整个差值归因于 pushdown。
- eFUN 的 pushdown off/on 都保留全部频率，各路径请求的压缩 payload 均为 **5,274,114,488 字节**，因此这里没有频率裁剪收益。输入尺寸和模型算术工作没有减少；forward 计时变化本身不能证明模型工作减少。

离线生成 50K 目标数据耗时 **533.47 秒**，共 49 个分片、约 **5.35 GB**；访问索引增加约 4.86 MB。这些离线成本未计入在线推理时间。GPU 抽查的存储往返输入与参考输入一致，独立 projected 检查通过。

**推理的处理边界：这批 GALP 数据是预先物化的模型目标，不是原图 DCT 的在线裁剪实验。** `generate.py` 调用 `efun_backend.Reference.coefficients()`，离线完成作者的 Resize(256)、CenterCrop(224)、Q100 JPEG 重编码和 DCT 提取，再存储 28×28 系数。在线 `evaluate_shards.py` 的几何变换为 identity，保留全部频率；因此 3.90× 包含将固定预处理移到离线的收益，不能当作在线 crop pushdown 或频率 pushdown 的独立加速证据。RGB PyTorch/DALI 则在线读取 JPEG、解码、缩放和裁剪，双方在线输入起点不同。

## 3. 已有训练结果

| 测试 | 实际工作量 | 训练窗口耗时 | 状态与含义 |
|---|---|---:|---|
| eFUN model-only | 5 次 warmup，120 次测量更新；122,880 个图像实例 | 42.609 秒，2,883.87 images/s | BF16/compile；反复使用 64 张驻留 GPU 图像；参数更新通过，不包含数据加载 |
| eFUN，GALP B6 | 8,192 个唯一样本，2 个池，8 次更新 | 7.984 秒 | BF16、未 compile；有限值检查及 8 图验证通过 |
| eFUN，JPEG A0 | 8,192 个唯一样本，2 个池，8 次更新 | 13.630 秒 | BF16、未 compile；有限值检查及 8 图验证通过 |
| eFUN，native A0 全局随机顺序 | 首池 4,096 图，4 次更新 | 首池 423.951 秒 | 两池测试中止，未产生成功的完整训练结果 |
| RGB EfficientNet-B0，PyTorch 配置验证 | 1,024 个唯一样本，1 次更新 | 34.979 秒，其中 epoch 准备 32.715 秒 | BF16、未 compile、4 workers；初始与更新后 8 图验证通过 |
| RGB EfficientNet-B0，DALI D2 配置验证 | 1,024 个唯一样本，1 次更新 | 44.059 秒，其中 epoch 准备 42.292 秒 | BF16、未 compile、4 workers；初始与更新后 8 图验证通过 |

B6/JPEG 诊断运行时另有 CPU 离线生成工作，且短窗口包含启动成本、顺序及裁剪策略不同，不能把上述时间当作正式训练加速比或收敛证据。model-only 中有 95 次严格检查和 25 次延后检查的测量更新。训练 DCT 输入与未取整参考的最大差异为 0.50005，符合既有整数取整约定；grid/projected 输出相同。

新增 RGB 检查的时间主要用于建立完整训练集的顺序与增强计划；仅一个更新的总耗时不适合比较 PyTorch 与 DALI 的稳定吞吐。其结果保存在同一结果根目录的 `training_rgb_config_check/`。

上述均为早期诊断。2026-09-15 已启动下述四臂完整训练矩阵；四臂 probe 均通过，每臂覆盖 8,192 张图、完成 8 次参数更新和 1K 验证。启动前 PRO 6000/H100 忙、4090 空闲，因此统一使用 RTX 4090 串行运行。

原四臂矩阵于 **2026-09-15 20:16（北京时间）全部完成**。受干扰的 JPEG 两轮随后从相同种子、相同配置重新训练，于 **21:27 完成**，退出码为 0。下表采用 JPEG 重跑结果、调优后的 DALI 重跑结果（†），以及原矩阵 GALP/PyTorch 结果。每臂每轮均覆盖 1,281,167 张唯一样本；每臂累计 2,504 次更新，完成初始及每轮 50K 验证，保存两个 epoch checkpoint。完整覆盖与有限值检查全部通过。

| 路径 | epoch 1 耗时（秒） | epoch 2 耗时（秒） | epoch 1 Top-1 | epoch 2 Top-1 | epoch 2 Top-5 | epoch 2 验证 CE |
|---|---:|---:|---:|---:|---:|---:|
| eFUN JPEG A0（重跑） | 1,434.106 | 1,386.967 | 8.012% | 19.372% | 41.744% | 4.040935 |
| eFUN GALP B6 | 557.899 | 529.811 | 7.666% | 19.464% | 41.654% | 4.050337 |
| RGB EfficientNet-B0 PyTorch | 1,830.422 | 1,761.882 | 7.434% | 20.924% | 44.068% | 3.889383 |
| RGB EfficientNet-B0 DALI D2（16 线程 / 预取 4）† | 949.099 | 938.800 | 7.352% | 21.234% | 44.218% | 3.872174 |

† DALI 两轮重跑于 **2026-09-16 17:04** 完成。GPU 采样仍发现约 21 秒和 4 秒的额外进程占用；这些时间是带已知并发占用标记的观测结果，不能视为独占 GPU 下的受控调优收益。准确率、训练 CE 及验证 CE 与旧配置完全一致。旧 4 线程 / 预取 2 的结果 **1294.526 / 1234.752 秒** 仍保留，完整调优和干扰记录见第 6 节。

初始 Top-1 在 DCT 两臂均为 0.104%，RGB 两臂均为 0.100%。第二轮训练吞吐依次为 923.72、2,418.16、727.16、1,364.69 images/s。表中训练耗时包含 epoch 准备、首次编译及训练循环，不含单独验证；两轮训练耗时合计依次为 47.02、18.13、59.87、31.46 分钟（DALI 带 † 标记）。

原 JPEG 两轮在约 17:30–18:07 间出现额外 GPU 计算进程；第一轮 150/313、第二轮 92/313 个池边界采样出现超过该轮最低总显存 256 MiB 的占用。旧耗时 1,419.592 / 1,400.919 秒保留在原目录，不再用于主表和加速比。

JPEG 重跑的 20:39:23–21:27:15 期间，定向 4090 的每秒采样共 2,871 条，仅记录到训练 PID 889490；既有每 100 ms 的全卡采样也只记录到该训练进程及其他两张 GPU 的任务，未观察到同卡额外计算进程。两轮的训练 CE、验证 Top-1/Top-5/CE 均与原运行一致；两轮耗时分别比原运行变化 +1.02% / -1.00%，不能据原来的并发占用断言出现了明显减速。原矩阵其余三臂同样未观察到同卡额外计算进程；这不适用于本次替换的 DALI 调优运行，其同卡额外进程已单独标记。即使没有同卡额外进程，也不排除共享主机 CPU/I/O 和运行间波动。

采用重跑结果后，GALP 相对 JPEG 的第一轮训练吞吐为 **2.57×**，第二轮为 **2.62×**，两轮合计为 **2.59×**；第二轮 DALI 相对 RGB PyTorch 的观测吞吐比为 **1.88×**，但 DALI 带已知同卡占用，不能当作受控加速比。GALP 第二轮 Top-1 比 JPEG 高 0.092 个百分点，DALI 比 RGB PyTorch 高 0.310 个百分点；单种子、两轮 warmup 结果不足以证明收敛等价或精度优势。DCT 两臂还同时改变了顺序和裁剪策略，不能将差异单独归因于 pushdown；两臂均使用全部 DCT 频率，不能据此证明频率减少或模型算术工作减少。

结果目录：JPEG 主结果使用 [training_v1_jpeg_rerun_20260915/jpeg_full](../../data/system_rgbnomore/e2e_v3/runs/efun/training_v1_jpeg_rerun_20260915/jpeg_full/)；GALP/PyTorch 使用 [training_v1](../../data/system_rgbnomore/e2e_v3/runs/efun/training_v1/)；DALI 使用 [调优后完整重跑](../../data/system_rgbnomore/e2e_v3/runs/efun/dali_tuning_20260916/rgb_d2_clean_full/)。目录中的 `clean` 是启动前命名，不表示采样证明无干扰。各臂 `initial_validation.json`、`epoch_0.json`、`epoch_1.json` 和 `training.json` 均已落盘。定向 GPU 采样在重跑目录的 `gpu_4090_processes.csv`，原始全卡采样在各臂 `gpu_process_memory.csv`。

**训练的处理边界不同于上述推理物化。** GALP B6 从完整源图 DCT 存储在线按 epoch/PLS 生成裁剪与翻转请求，随后规划读取、解码、DCT resize、RandAugment/Mixup；计时在 `pipeline.start_epoch()` 之前开始，包含这些在线准备和执行。两轮运行统计中 selected/full vector 比分别为 **72.00% / 72.40%**，说明裁剪请求确实减少了读取路径选择的压缩向量，不是提前保存每轮裁剪后的模型输入；这些计数不是物理磁盘字节或隔离的性能加速比。eFUN 保留全部 192 通道，训练也没有频率剪枝。源 JPEG 系数提取、GALP 编码、离线物理重排和索引构建仍是计时外的存储准备成本。B6 与 JPEG A0 还改变了裁剪共享及样本顺序，当前对照不隔离 crop pushdown。RGB DALI D2 在线通过 `decoders.image_slice` 做解码裁剪，再 resize/flip/normalize；每轮增强决策也在计时内生成，没有预先物化裁剪图像。

**尚无 eFUN 家族最终收敛结果。** 作者 450-epoch 配方保留在 `efun.py official_train`，与下面的系统对照配方分开。

### 3.1 RGB DALI CUDA Graph 补测（2026-09-22）

沿用RTX 4090、16线程/预取4、两轮完整1,281,167张训练及初始/每轮50K验证，开启模型 `reduce-overhead` CUDA Graph。第二轮训练由历史 938.80 秒变为 **552.35 秒**（2319.47 img/s），Top-1为21.234%。旧结果曾记录同卡并发任务，因此跨日期时间比不能解释为纯Graph收益。eFUN B6仍是旧default结果，本次补测没有重跑B6，也没有改变DCT频率或模型计算量。

完整计时、数值对照及GPU breakdown见[补测报告](../../benchmarks/system_dct_major/docs/DALI_CUDA_GRAPH_TRAINING_SUPPLEMENT_2026-09-22.md)。

## 4. 本轮训练配置：base eFUN 与 RGB EfficientNet-B0

先验证实际已接入的 base 模型，L/S/S+ 不纳入本轮默认矩阵。复用 `cnn_training_baselines.sh`、`training_pls.py` 和已有 PyTorch/DALI 训练适配器：

| 测试臂 | 模型 | 输入及顺序 | 用途 |
|---|---|---|---|
| `efun:jpeg` | eFUN | JPEG→DCT，每图独立裁剪、全局 shuffle，A0 | DCT 训练控制组 |
| `efun:native` | eFUN | GALP，按物理组共享裁剪、封闭池 shuffle，B6 | GALP 目标系统 |
| `efun:rgb_pytorch` | RGB EfficientNet-B0 | 既有 PyTorch RGB 训练适配器，固定种子的每图裁剪/翻转 | RGB 系统基线 |
| `efun:rgb_d2` | 同一 RGB EfficientNet-B0 | DALI D2，使用与 RGB PyTorch 相同的裁剪/翻转决策及顺序 | 对齐的 DALI 基线 |

默认排除已表现出高读取成本的 native A0；DALI D3 自行生成增强，不作为本轮对齐控制组。DCT 两臂和 RGB 两臂分别共享初始化、优化器、更新计划和各自验证预处理；RGB 与 DCT 的增强及标签形式不同，不能把跨模型收敛差值解释为单一调度效果。

共同训练配置：

- 每臂使用全部 **1,281,167** 张训练图，先运行 **2 个完整 epoch**。物理映射与已有 ImageNet-512 数据复用。
- 从零初始化，种子 **11997733**；虽然构造过程严格加载 checkpoint，随后所有含参数层都重新初始化，训练不保留预训练权重。
- **BF16 autocast + torch.compile**；microbatch **64**，累积 **16**，有效 batch **1024**；尾批按实际图像数归一化，不丢样本。梯度裁剪范数 **1.0**。
- 复用 AdamW：基础 LR **0.003**、betas **(0.9, 0.999)**、epsilon **1e-8**、AdamW 内置 decay 为 0；另用既有独立 weight decay，系数 **1e-4**。
- 保留 **300-epoch** 总调度和 **10,000 update** warmup，之后 cosine；2 epochs 共 **2,504 次更新**，仍处于 warmup，不能证明最终收敛。
- B6 使用 **G=1024、M=4**；DCT transform 使用注册的默认配置，不沿用其他 CNN 手工设置的 32768 blocks。JPEG/PyTorch 用 16 workers，DALI D2 原结果使用 4 workers、预取深度 2；调优后复现脚本使用 16 workers、预取深度 4，完整训练复测状态见第 6 节。
- eFUN 训练复用 DCT 裁剪/缩放/翻转、取整、RandAugment 和 Mixup；验证使用 `imagenet512_val_efun28_block_major` 的作者参考目标。RGB 训练与验证均复用系统配方的 **bilinear、mean/std=.5**，使用 hard labels，不使用 DCT RandAugment/Mixup。这不是 RGB 预训练推理的 bicubic/ImageNet 归一化，也不是作者 eFUN 的 450-epoch 配方。
- full 模式在随机初始化时、epoch 1 后和 epoch 2 后分别进行 **50K 验证**。保存初始验证 JSON、每轮 Top-1/Top-5/CE、覆盖样本数、训练耗时与 epoch checkpoint；重启 full 模式会复用已完成 epoch 的 checkpoint。

从项目根目录预览四臂完整命令（不运行 GPU、不建立输出目录）：

```bash
EXP=galp/benchmarks/dct_models
SPECS='efun:jpeg efun:native efun:rgb_pytorch efun:rgb_d2'
bash "$EXP/cnn_training_baselines.sh" plan training_v1 "$SPECS"
```

真正运行前检查 GPU。优先空闲 PRO 6000/H100；4090 仅确认空闲后使用。eFUN 运行模式要求调用者显式指定 GPU：

```bash
nvidia-smi
# 确认选中卡空闲后，只设置其中一个 UUID：
# PRO 6000: GPU-e796262d-3449-6af1-586d-8460d8836d1b
# H100:     GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76
export CUDA_VISIBLE_DEVICES=GPU-40c637bd-acf5-ea1a-0df8-617138228467
bash "$EXP/cnn_training_baselines.sh" probe training_v1 "$SPECS"
bash "$EXP/cnn_training_baselines.sh" full training_v1 "$SPECS"
```

`probe` 为每臂 2 个池、1K 验证，启用 compile；`full` 为每臂 2 个完整 epoch，串行运行四臂。输出在 `e2e_v3/runs/efun/training_v1/{backend}_{probe|full}/`。本轮 probe 与 full 均已完成；脚本会跳过已有 `training.json` 的输出目录。

## 5. 训推 Nsight Systems 检查与 breakdown

2026-09-16 已完成 **6 条推理、4 条训练**采集，最后一份原始报告于 14:04:31 写出。目录名仍为 [nsys_20260915](../../data/system_rgbnomore/e2e_v3/runs/efun/nsys_20260915/)。共有 10 份 `.nsys-rep`（合计 90.71 MiB）、10 个 SQLite、10 份 breakdown JSON、6 组推理 PNG/SVG 图，以及 4 组训练 PNG/PDF 时间线。

检查结果：所有日志均有 `profile_complete: true`；10 个 SQLite 的完整性检查通过，每个文件均有一个闭合的 `dctnet.capture` 窗口。推理各含 64 次 forward / 4,096 张图；训练各含 256 个 microbatch / 16,384 张图、16 次更新。训练从第一轮 checkpoint 恢复，首池预热后捕获全局更新 1,257–1,272，处于 update 100 之后的检查策略。GPU 的 100 ms 进程采样所记录的 10 个 PID 与十份 trace 的训练/推理进程一一对应，未发现同卡外部进程；这不排除共享主机 CPU/I/O 竞争。

下表均为 **启用 profiler 的局部窗口**，单位为秒；H2D 单位为十进制 GB。模型、输入、拷贝可能重叠，不能直接相加。GPU 空闲表示本进程没有 kernel/memcpy/memset 活动的时间，不是 SM 利用率。传输量按窗口统计，预取可能跨越样本窗口边界，不能当作每张图的精确存储成本。正式速度仍采用前文无 profiler 的完整运行结果。

### 5.1 推理：每条 4,096 张图

路径名称直接链接原始 `.nsys-rep`；同目录包含 SQLite、`breakdown.json` 和图表。

| 路径 / 原始报告 | 窗口耗时 | 模型 kernel | 输入 kernel | GPU 空闲（占比） | H2D GB |
|---|---:|---:|---:|---:|---:|
| [eFUN JPEG R](../../data/system_rgbnomore/e2e_v3/runs/efun/nsys_20260915/inference/jpeg/dct_reference.nsys-rep) | 2.725 | 0.777 | 0 | 1.835（67.3%） | 2.466 |
| [eFUN GALP grid/off](../../data/system_rgbnomore/e2e_v3/runs/efun/nsys_20260915/inference/grid_off/galp_native.nsys-rep) | 0.953 | 0.774 | 0.114 | 0.057（6.0%） | 0.448 |
| [eFUN GALP projected/off](../../data/system_rgbnomore/e2e_v3/runs/efun/nsys_20260915/inference/projected_off/galp_projected.nsys-rep) | 0.866 | 0.774 | 0.031 | 0.057（6.6%） | 0.448 |
| [eFUN GALP projected/on](../../data/system_rgbnomore/e2e_v3/runs/efun/nsys_20260915/inference/projected_on/galp_projected.nsys-rep) | 0.867 | 0.775 | 0.032 | 0.058（6.7%） | 0.448 |
| [RGB PyTorch](../../data/system_rgbnomore/e2e_v3/runs/efun/nsys_20260915/inference/rgb_pytorch/rgb_pytorch.nsys-rep) | 1.527 | 0.787 | 0 | 0.627（41.1%） | 2.466 |
| [RGB DALI（4/4）](../../data/system_rgbnomore/e2e_v3/runs/efun/dali_tuning_20260916/profiles/inference/rgb_dali.nsys-rep) | 0.952 | 0.802 | 0.075 | 0.101（10.6%） | 3.232 |

四条 eFUN 路径的模型 kernel 数均为 **12,352**，GPU 执行时间约 0.775 秒。JPEG 的主线程输入等待为 1.374 秒，projected/off 为 0.000509 秒；窗口内 H2D 降低约 **81.8%**，GPU 空闲明显缩短。证据支持输入供给与传输路径改善，不能解释为模型算术工作减少。

grid→projected 的输入 kernel 数从 52 降至 28，输入 GPU 时间从 0.114 秒降至约 0.031 秒，主线程输入整理从 25.27 ms 降至 0.136 ms；模型 kernel 数保持不变。projected off/on 结果接近，符合 eFUN 保留全部频率、没有频率裁剪的设定。

RGB 两路的模型 kernel 数均为 **19,264**。调优后 DALI 的模型与拷贝重叠为 0.140 秒，PyTorch 为 0；DALI 的总 H2D 字节数更高，不能用“传输更少”解释这组 RGB 改善，观察到的收益主要表现为更少的 GPU 空闲和更多重叠。

### 5.2 训练：每条 16,384 张图

模型 kernel 分类包含 forward、loss、backward、optimizer 和相应数值检查。路径名称链接原始报告；同目录有 `breakdown.json` 和 `timeline.png` / `timeline.pdf`。

| 路径 / 原始报告 | 窗口耗时 | 模型 kernel | 输入 kernel | GPU 空闲（占比） | H2D GB |
|---|---:|---:|---:|---:|---:|
| [eFUN JPEG A0](../../data/system_rgbnomore/e2e_v3/runs/efun/nsys_20260915/training/jpeg/capture.nsys-rep) | 19.065 | 3.798 | 0.118 | 14.694（77.1%） | 9.868 |
| [eFUN GALP B6](../../data/system_rgbnomore/e2e_v3/runs/efun/nsys_20260915/training/native/capture.nsys-rep) | 8.799 | 3.832 | 1.322 | 3.970（45.1%） | 1.224 |
| [RGB PyTorch](../../data/system_rgbnomore/e2e_v3/runs/efun/nsys_20260915/training/rgb_pytorch/capture.nsys-rep) | 23.738 | 5.496 | 0 | 17.782（74.9%） | 9.865 |
| [RGB DALI D2（16/4）](../../data/system_rgbnomore/e2e_v3/runs/efun/dali_tuning_20260916/profiles/training/capture.nsys-rep) | 14.252 | 5.361 | 0.147 | 8.220（57.7%） | 6.512 |

eFUN 两路模型 kernel 数均为 **210,768**，GPU 执行时间接近。GALP 的输入 GPU 工作增加，但窗口内 H2D 减少约 **87.6%**，输入与模型重叠 0.382 秒，GPU 空闲从 14.694 秒降至 3.970 秒。它支持系统数据路径收益，不能单独归因于频率 pushdown，也不支持模型工作减少；A0/B6 还改变了裁剪和顺序策略。

检查时发现训练分析器只识别 `dali::` 后台线程，遗漏了独立的 `nvjpeg::` 解码线程，将其误计入模型。已在 `analyze_training_capture.py` 中修正，并基于原始 trace 重新生成 DALI breakdown 和 PNG/PDF 时间线。**32,768 个 nvJPEG kernel** 已归入输入：DALI 输入 kernel 共 34,048 个，模型 kernel 为 **344,368**，与 RGB PyTorch 完全一致。独立 SQL 核对通过，总 GPU 活动并集保持不变。

调优后的两份 DALI 报告于 2026-09-16 16:25 完成，替换上表的 DALI 行，原十份报告仍保留。两份新 SQLite 完整性检查通过；GPU 采样分别只记录到对应推理/训练进程。新训练报告仍为 344,368 个模型 kernel、34,048 个输入 kernel，输入与模型重叠 0.066 秒。

**以下为旧 DALI 配置的历史异常诊断，39.536 秒并非上表新报告。** 旧 DALI 训练采集窗口存在明显放大，不能据此断言 DALI 比 PyTorch 慢。 与既有第二轮中相同四个池的无 profiler 时间比较：

| 路径 | 原无 profiler 同池时间（秒） | 本次采集窗口（秒） | 跨运行比值 |
|---|---:|---:|---:|
| eFUN JPEG | 17.303 | 19.065 | 1.10× |
| eFUN GALP | 6.941 | 8.799 | 1.27× |
| RGB PyTorch | 21.164 | 23.738 | 1.12× |
| RGB DALI D2 | 15.670 | 39.536 | 2.52× |

这些比值来自不同运行，不是受控测得的纯 profiler 开销。DALI 的主线程 forward/backward NVTX 区间为 35.150 秒，PyTorch 为 20.630 秒，而模型 GPU kernel 分别只有 5.536 / 5.496 秒，且数量相同。因此异常增长主要体现在主机区间及 GPU 空隙，不能解释为模型算术工作增加。当前采集未启用 CPU sampling/context-switch tracing，无法进一步确定是 profiler、CPU 调度竞争还是其他主机开销。DALI trace 可用于检查 kernel 组成和传输活动；原配置无 profiler 完整训练中 DALI/PyTorch 为 **1.43×**；调优后的完整结果见第 3 节及其并发标记。GALP/JPEG 第二轮为 **2.62×**。

训练的 `input.pool_wait` 只覆盖取池，不覆盖所有 microbatch 获取过程，不能把该 NVTX 区间接近零解释成没有输入等待。


## 6. DALI 配置调优（2026-09-16）

扫描线程数 4/8/16 与预取深度 2/4 的六种组合，固定 checkpoint、数据顺序、增强、batch 和计算精度。推理每种配置三次完整 50K；训练先各运行一次，从同一 epoch-1 checkpoint 恢复，排除首池后测量 49,152 张图，再对旧配置和最快候选各补两次。配置选择仅覆盖上述范围，不声称全局最优。

| 线程 / 预取 | 推理扫描三次中位数（秒） | 训练首测（秒） |
|---|---:|---:|
| 4 / 2 | 14.223 | 44.398 |
| 4 / 4 | 11.732 | 61.394 |
| 8 / 2 | 11.790 | 36.174 |
| 8 / 4 | 15.585 | 35.136 |
| 16 / 2 | 19.865 | 55.118 |
| 16 / 4 | 12.017 | 31.443 |

早期推理扫描存在明显运行间波动，不能用 19.865 / 11.732 宣称调优收益。完整训练之后进行了三组交叉确认：旧 16/2 为 **11.929、11.941、11.904 秒**；新 4/4 为 **11.719、11.733、11.782 秒**。两组中位数分别为 **11.929 / 11.733 秒**，新配置耗时减少 **1.64%**。全部六次预测与原 DALI 完全一致；这支持较少线程下的接近吞吐和小幅改善，不能宣称显著或普遍的提升。

训练旧 4/2 三次为 **44.398、88.300、46.889 秒**；候选 16/4 为 **31.443、70.300、42.033 秒**。中位数耗时减少 **10.36%**，选择 16/4 进行完整训练；其他四个配置只有筛选测量，不能据此排除它们在更多重复下更快。

首次调优后完整训练已完成，epoch 时间为 **1293.667 / 906.917 秒**；训练 CE 及两轮 50K 验证 Top-1/Top-5/CE 与旧配置完全一致，每轮覆盖 1,281,167 张唯一样本，累计 2,504 次更新。但 100 ms GPU 采样发现额外 PID 14094（15:53:56–15:54:21）和 151728（16:22:20–16:22:37），分别涉及两轮。因此该次运行仅保留为历史记录。随后从相同种子重跑，于 **17:04** 正常完成，输出到 `rgb_d2_clean_full/`；两轮耗时 **949.099 / 938.800 秒**，样本覆盖、2,504 次更新、初始及每轮 50K 验证和两个 checkpoint 均齐全，训练及验证数值与旧配置一致。

重跑的 19,628 条主进程 GPU 样本记录之外，仍检测到 PID **253019**（16:37:38–16:37:59，208 条）与 **296074**（16:48:41–16:48:45，40 条）的短时同卡占用。因此第 3 节以 † 明确标记这次替换结果。相对旧结果，第二轮耗时减少 **23.97%**、吞吐为 **1.315×**，两轮合计耗时减少 **25.36%**、吞吐为 **1.340×**；这些是跨运行观测值，不能把全部变化归因于线程/预取配置，也不能因占用短暂就假定影响为零。两次完整重跑均有额外进程，未再自动发起第三次；如需无同卡干扰的训练时间，需要先确保一个完整的独占测试窗口。

原始调优数据：[dali_tuning_20260916](../../data/system_rgbnomore/e2e_v3/runs/efun/dali_tuning_20260916/)。`measurements.json` 保存扫描与训练复测，`inference_confirmation/` 保存推理交叉确认，`selected.json` 保存选定配置及推理代表运行；旧结果不删除。新增预取参数通过 DALI 的 17 图尾批顺序 GPU 测试。
