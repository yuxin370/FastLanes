# FastLens matched DCT 基线复现与 agent 开发计划

版本：2026-10-08。目标：为 FastLens 增加可信、可解释、可复核的 Image Calculator 和 Frequency-Store 对照。本文是实施规格与实验计划；不包含已经完成的基线实现或新测量结果。

## 1 结论与交付范围

建议采用两条实现线、三类实验。首先复现两篇工作的核心表示与存储机制，建立小规模原生验证；其次将其存储策略显式适配到 FastLens 的同一 JPEG 源系数、同一模型输入与同一请求轨迹，加入现有 matched DCT 主表。如果论文需要进一步声称优于两项工作的完整部署方案，再做原生表示下的准确率约束比较。

最重要的命名要求是：现有七个 FastLens DCT workload 对应的 IC/FS 行应写为 **paper-guided reimplementation with a JPEG-DCT adapter**。IC 原生会搜索分块、采样、剪枝和量化；FS 原生采用 32×32 DCT、统一量化因子 50。将这些选择冻结到已有 JPEG 8×8 系数后，比较隔离的是存储和输入路径，不能声称覆盖了原系统全部设计空间。[IC §4.1–4.5；FS §4–5；FL §3.4.1、§4.1]

| 实验层 | 保持相同 | 允许变化 | 能支持的结论 |
|---|---|---|---|
| 现有 workload 的严格 matched DCT | 源 q/Q、几何、请求轨迹、最终模型输入、checkpoint、batch、精度、训练更新 | 无损编码、布局、读取、调度、CPU/GPU 分工 | 同一逻辑工作负载下的输入系统效率 |
| 原生机制验证 | IC/FS 原生 B、量化、mask、模型输入；横向对比时使用同一份量化系数 | 每图、按 batch、按反对角线组织 | 复现是否保留论文机制；压缩与重建行为 |
| 准确率约束下的部署对比 | 原始数据、数据切分、质量预算、调优预算和选择规则 | 表示、输入形状、checkpoint、必要的模型适配 | 给定质量约束下的空间与速度 Pareto |

建议最终方法名如下。`adapted`、`native`、`enhanced` 应同时出现在配置和图表说明中，而不只出现在代码注释里。

| ID | 名称和用途 | 主要边界 |
|---|---|---|
| cpu_dct | 已有 CPU-DCT | 保留现有经过验证的逻辑 reference |
| ic_m_shared | IC-style matched，完整源系数库 | 每图 NPZ，保留 JPEG q/Q；原生表示搜索关闭 |
| ic_m_fixedk | IC-style matched，固定源频率配置 | 离线保留所需 source-K，保留所有空间位置；为不同 K 计副本 |
| ic_m_batched | IC-style matched，batch256 对照 | 排除小文件和普通 batching 的影响；结构属于显式工程选择 |
| fs_m_b8 | FS-layout matched，B8 | 15 条反对角线，默认存储 batch256；原 JPEG q，不做 Q50 再量化 |
| fastlens | FastLens 当前实现 | 同一 contract 和 trace；只增加可验证的接入接口 |
| ic_native / fs_native | 论文原生机制实现 | 独立配置、独立模型输入；不混入上述主表 |

本次核查作者 publication 页面、CIDR 页面和公开标题/作者检索，未找到可验证的官方 IC/FS 源码发布。官网可找到论文、slides、IC technical report 和后续教程入口。这个结论表示本次检索的覆盖范围，不证明不存在未索引或未公开仓库。[W1–W3]

## 2 证据等级和版本锁定

每个实现决定在 `ASSUMPTIONS.md` 中标记以下来源等级：

- **P**：论文明确给出的机制、参数或 API。
- **D**：根据公式、图示或明确机制推导出的结果。
- **E**：为完成复现而选定的工程规则，不属于作者已披露实现。
- **U**：仍缺少证据的细节；必须给出处理规则、影响和验证方式。

优先级为附件原文、可访问的作者官方材料、依赖库官方文档。后续综述和教程不能自动覆盖论文版本。IC 的 2024 论文给出 6048 个候选，而作者 2025 教程页面写 4104；本计划以 2024 附件为 IC 版本锚点，不把不同数字混成一套搜索域。[IC Table 1、§4.2.5；W2]

需要补齐的环境输入包括 FastLens 仓库、实际 commit、数据 manifest、checkpoint、profile、训练配置和可用硬件。附件给出了匿名仓库入口，但本次未能通过可用读取工具获取其内容。因此后文目录和接口是建议规格，实际接入文件由首个 agent 在仓库中定位，不能声称已经检查了源码。

## 3 从论文恢复的核心规格

### 3.1 Image Calculator

IC 的原生写入路径是 RGB → YCrCb → 对所有分量使用相同采样率 → B×B 分块 DCT → 左上三角剪枝 → scalar quantization → NumPy 无损压缩。读取直接恢复量化后的 DCT 表示供网络使用，不在 inference hot path 做反量化、IDCT 或 RGB 重建。[IC §4.1、§4.5，pp.5、12]

| 内容 | 规格 | 证据状态 |
|---|---|---|
| 分块 | B ∈ {8,16,32,64,128,256} | P：Table 1，p.6 |
| 采样 | 不采样、隔行、隔列、行列各隔一；作用于所有颜色分量 | P：§4.2.1，pp.6–7；原文“4×4”不是 stride4 |
| 剪枝 | 对零基坐标保留 u+v<k，k ∈ {1,…,B} | P/D：§4.2.3、Fig.6，pp.7–8 |
| 输出 | 每个 block 裁为含结构零的 k×k，再按原 block 网格拼接成三个平面 | P/D：Fig.6 |
| 量化 | 所有被保留系数除同一个标量，再舍入到最近整数；标量 ∈ {20,50,100} | P：§4.2.4，pp.8–9 |
| 数值类型 | 存 int32，训练使用 float32 | P：§5.1，p.13 |
| 压缩 | NumPy 无损压缩；FS 后文将与 IC 共用的编码明确为 savez_compressed | P：IC §4.1.2；FS §6.2.1 |
| 原生文件 schema | array key、轴序、端序、header、精确压缩级别没有完整公开 | U；每图 numeric NPZ 是建议默认，且与 FS 对 IC 的 file-by-file 描述一致 |

具体例子：256×256 输入、B=32、k=3 时，每块仅保留六个系数，但输出块为有三个结构零的 3×3；最终 tensor 为 `3×24×24`。不能改成把六个频率都放进 channel 维的 tensor 后仍称忠实 IC。原文候选域中的 `k=B` 也仍是三角形：B8、k8 保留 36 个系数，不是全部 64 个。因而 matched all64 或 arbitrary K 是显式的候选域外适配。[IC Table 1、Fig.6、§4.2.3]

IC 搜索的最低复现要求如下：

1. 枚举理论域 `4×(8+16+32+64+128+256)×3=6048`。主实验曾固定 no-sampling、q=50，再约取 20% 的 B/k 组合，共 96 个；精确 96 个组合未公开，新的子集必须存为显式 manifest，不能说与原列表一致。[IC §4.2.5、§5.1]
2. Accuracy model 按压缩程度将候选分为 16 个 bucket，评估关键的 6 个 bucket，取每个 sampled bucket 内最大 accuracy，再线性插值。未披露通用的 critical bucket 选择规则，图示推断与新策略必须标注。[IC §4.3.1、Fig.7–9]
3. 从压缩最少的配置开始训练，然后向压缩更多的配置 transfer，使用 20 或 40 epochs；首模型从头训练。候选间优化器状态、统计量迁移等补充规则要单列。[IC §4.3.1]
4. Latency 和 storage 在目标设备实际 profiling。不要虚构一个“论文规定的 learned latency regressor”。所选 bucket 未评估时再实际评估，最终报告实测 accuracy。[IC §4.3.2–4.4]
5. 完整搜索是可选的部署实验任务，不是现有 matched throughput 的前置条件。仅完成 NPZ reader 或稀疏 grid search 时，不宣称复现了完整 self-design optimizer。

原生训练方面，论文在 RGB 上做 RandomResizedCrop/RandomHorizontalFlip 后再做 GPU DCT，并按 DCT 表示的 mean/std 标准化；并非直接复用了 FastLens 的 JPEG 系数域增强。原文使用 SGD、210 epochs from scratch，并提及对 TorchJPEG 做过未披露优化。[IC §5.1，p.13]

### 3.2 Frequency-Store

FS 的最终方案是每条 `d=u+v` 反对角线一列，一个存储 batch 内全部图像、全部 block 的该对角线值存进一个独立压缩数据文件。原文先比较了每系数一列、每 bit-width 一列，最后选中每反对角线一列。[FS §5.2、Fig.7，p.5]

| 内容 | 规格 | 证据状态 |
|---|---|---|
| 原生表示 | B=32，scalar quantization factor=50，无采样搜索 | P：§4、§4.1–4.2，pp.3–4 |
| 原生列数 | 2B−1=63；B8 matched 适配则为 15 | D：由反对角线定义得出 |
| 存储分组 | 256 幅图像/存储 batch | P：§6.1，p.6 |
| 列编码 | 每条 diagonal 采用 8 或 16 位整数，savez_compressed | P：§5.2.3、§6.2.1，pp.5–6 |
| 完整列元素顺序 | 颜色分量处理、内部遍历次序、ragged geometry 编排未完整披露 | U；本计划给出确定性实现 |
| 共享 | 相同 B、变换、量化，不同 pruning budget 复用一份底层列数据 | P：§3–4，pp.2–3 |
| 低带宽图 | /dev/shm profile 的 server 时间加上基于 50 Mbps 的传输估计；FS/JPEG 假定 perfect sharing | P：§6.2.2，p.7 |

不能把以下内容写入“原生 FS 默认实现”：bit-plane、bitshuffle、Zstd、LZ4、Parquet/Zarr、FastLanes GPU 解码或二维空间索引。按 bit-width 进行 bitpacking 是论文比较后放弃的方案。需要这些增强时建立单独 enhanced variant。[FS §5.2]

`savez_compressed` 对应 ZIP_DEFLATED 的 NPZ，内部成员为 NPY 数组。一个普通压缩成员没有任意元素的独立 seek 能力；在未建立额外压缩检查点的情况下，应按完整成员/列解压，再做 gather。[W4]

## 4 冻结 matched contract

### 4.1 SourceManifest

所有写入器从同一个 canonical source 读取，只允许重排、无损类型编码和无损压缩。canonical source 必须由 JPEG coefficient API 提取已有量化整数 q；不能 RGB decode 后重新 DCT，不能把 JPEG quality 值当原量化表，也不能额外统一除以 50。[FL §3.1；W5]

每幅图像记录：稳定 image_id、label、split、JPEG SHA256、宽高、component id、sampling factor、block grid、有效边界、padding 规则、每个分量的量化表 Q 和引用 id、系数实际类型、zigzag 与 (u,v) 映射版本。系数可以保存为 `q[c][block_y,block_x,u,v]`；components 允许不同形状，不能无声明地 upsample chroma。JPEG bitstream 中的 DC 差分编码不等于 coefficient API 输出的 DC 值，应以 API 恢复后的绝对 q 为基准。

“无损”限定为保留当前 ImageNet-512 JPEG 的 q/Q；不代表 ImageNet 先 resize/再 JPEG 编码这一过程无损。必须记录共同源数据制作的 resize、JPEG encoder、quality 和 subsampling recipe。[FL §4.1]

可以保留各方案的原生无损存储位宽：IC int32、FS int8/int16、FastLens 的实际列类型。但需另做共同 int16 的控制实验来隔离位宽贡献。另允许 IC 在其整个 image/batch 编码单元上选择 range-safe int8/int16，作为强化对照；不得只给 FS 安全缩窄的机会，再把全部增益归因于列布局。任何缩窄必须检查范围，不能 clip、wrap 或静默改变符号。超过已声明输入类型范围时 fail closed，或以另一个明确命名的 fallback variant 处理。

### 4.2 ModelProfile 和源依赖

每个 profile 冻结输出 tensor 的数量、shape、dtype、layout、颜色/频率次序、normalization 数值和统计来源，以及 dequantization、clamp、resize/mixing、round、saturation、affine conversion 的顺序。CNN 的 projected NCHW 与 Transformer 的 Y/C grids 不能被统一 flatten 成一种输入。[FL §3.3.3、§3.4.1]

分别记录 `source_frequency_set` 和 `output_channel_map`。FastLens 稿明确指出 resize 可能依赖输出中不存在的源频率，§4.4 的 MobileNetV2-DCT24 实验仍使用 source K=64。不能从 DCT24/DCT32 等模型名推断源 K。[FL §3.4.1、§4.4]

依赖集合最稳妥的形式为 `(image_id, component, block_y, block_x, u, v)`。共同矩形 `G×K` 只是便于聚合的特例；不同分量、变换、边界可产生不同需求。先用全源、独立 reference 验证 dependency closure，再启用读取裁剪。未解决的 dependency 不得被假定为一个漂亮的小 K。

### 4.3 RequestTrace

每个输入 occurrence 记录 `occurrence_id, image_id, epoch, view_id, logical_slot, microbatch_id, crop_box, resize_geometry, flip, augmentation_parameters, mixup_partner_occurrence, mixup_lambda, target_parameters, profile_id`。trace 存为不可变文件并计算 hash。

同一个 seed 不足以证明 matching：worker 数、prefetch 和物理排序改变随机数消费顺序。因此所有 backends replay 相同 trace，而不分别调用随机 augmentation API。

FastLens 默认的 bounded-pool shuffle、每 virtual segment 共享 crop 也是 trace 的一部分。主表让所有 loaders 使用同一默认 trace；再用所有 loaders 共享的 independent crop/global shuffle trace 做敏感性实验。不要把不同采样策略的吞吐差异混为存储优势。[FL §3.4.2]

允许按物理局部性重排读取、合并重复请求、共享已解压 source，但最终 tensor 必须回到 logical slot。相同 image 的两个 occurrence 可以共享读取，却不能共享各自不同的 augmentation 结果。Mixup 的伙伴、顺序与 lambda 要匹配，不只样本集合匹配。

## 5 IC 的实现路线

### 5.1 原生 reference

建立独立 `ic_native_codec`，支持显式 B、k、采样和 scalar q。以 FP64 正交 DCT 作为小张量 oracle，生产路径使用被验证过的批量实现。实际 TorchJPEG `block_dct` 可以处理可变 B，但 `batch_dct` 和高层 normalize 等 convenience function 中存在固定 B8 逻辑；不要直接用于 B32/B64。以实际源码和正交性测试核对 scale，不能仅抄 docstring。[W6]

颜色矩阵、Cr/Cb 次序、输入范围、是否 level-shift 128、padding、舍入 tie rule、standard deviation 为零的处理和统计轴属于必须显式冻结的补充规则。对每个需要独立统计的 B/k/q/颜色约定，只使用训练 split 计算并冻结 mean/std，记录统计轴、结构零是否计入和 stats hash；不得使用最终 validation/test 数据或默认的外部 B8 统计量替代。两个 anchor 后端复用同一份统计量。建议默认像素尺度使用 0…255、按所选颜色 convention 做明确 level shift，并用独立参考验算；这些是 E，不能声称是未披露的作者原值。尤其不能将 0…1 范围直接除以 50 后把大量全零结果误当成功复现。

压缩前构造与论文一致的三平面 mosaic，三角形外的 structural zeros 保留在 k×k block 内；在训练中按冻结的统计量处理。使用 numeric NPZ，禁止 object arrays/pickle。schema 定义 array 名称、shape、dtype、端序、版本和预处理 hash。原生 inference 直接消费量化系数，不乘回 scalar q、不做 IDCT。

### 5.2 JPEG-DCT matched writer

`ic_m_shared` 每图保存完整 source q 和必要 metadata，沿 image-major 顺序组织成一个 numeric payload，或按 component 保存确定的数组。未公开的容器细节必须写入 assumptions。该 full mask 不是原生 k=B 的别名。

`ic_m_fixedk` 只在 profile 的源依赖闭包明确后启用。静态 mask 为所有合法 online requests 的 source-K 并集；保留全部源空间 block，不能预先裁掉某一条评测 trace 恰好没有访问的区域。每个不同静态配置有独立 manifest 和转换/空间账本。动态 resize 导致并集为 all64 时，fixedk 退化是合理结果。

如果静态 K 不属于 IC 的三角候选域，则标记 `mask_family=matched_arbitrary_adapter`。只有该 mask 恰好等于某个合法 k 对应的 `{(u,v):u+v<k}` 才可记录 `native_triangle`，仅为其子集不满足条件。该标记只描述 mask 形状，JPEG q/Q 管线仍是 adapted；所有标记都必须经实际坐标验证。为节省大小而直接存最终 model-ready tensor，属于 `static_precomputed` 对照，另表且给所有方法相同机会。

`ic_m_batched` 默认每 256 图一个 batch archive。为检测简单 batching 的作用，建议将 image-major payload 拼成一个连续 numeric member，并给出 offsets；这提供真实跨图压缩机会。若使用每图独立压缩成员的 envelope，单独命名为 bundle，不把这两个版本混为同一 IC-b。原文 IC-b 的具体成员布局未披露。[FS §6.2.1]

### 5.3 Reader 与合理优化

Reference 路径：读取完整需要的 image/batch 压缩单元 → native DEFLATE 解压 → 恢复原始 q → 调用共享的语义适配器 → pinned H2D → 相同模型。

CPU workers、批量 gather、pinned staging、异步 H2D、跨 microbatch prefetch 都应合理实现和调优。禁止逐系数 Python 循环作为最终热路径。主比较保留 NPZ/DEFLATE；native C inflate 等保持格式语义的优化可以使用，记录实现和版本。

建议额外测一个强化版本：CPU 解压后传输 compact signed q，在 GPU 做普通、共享的语义变换。标记 `gpu_transform_adapter`，不要称为原论文管线。它能检验 FastLens 的优势是否仍超出“竞争者在 CPU 上过早展开 float32”。不能将 FastLens 专属 compressed planner/decode/fusion 直接搬进 IC 再仍用原方法名。

## 6 FS 的实现路线

### 6.1 Writer

先实现 native B32/Q50，再使用同一个布局模块接入 canonical JPEG B8 q。两种 source mode 使用独立配置，不能共享含义模糊的 `quality=50` 参数。

对每个存储 group 和 diagonal d，将所有图的相关系数放入一个连续 signed array。推荐确定性顺序为 `image → component → raster block → diagonal 内 u 升序`；这是 E，不是作者已公开内部顺序。每个 `(group_id,d)` 一个独立 NPZ 文件，包含一个系数 member，geometry/offsets 放入有版本的 metadata。所有 component 共用同一 diagonal 文件可保持 B8 的 15 个列对象；若按 component 再拆文件，必须报告这个额外物理设计。

对每列实际 min/max：可放入 [−128,127] 则 int8，否则在 signed int16 范围内用 int16；超范围报错或显式 fallback。metadata 必须记录 dtype，禁止 decoder 按 diagonal 序号猜宽度。默认 group256；尾 group 保留实际图数。

原生完整 B32 source 可存 63 条 diagonal；只物化一个最大预算前缀也可，但 manifest 必须声明可支持范围。B8 matched shared store 保存 all64 q，即全部 15 条 diagonal。不能为每个请求 k 重新生成裁好的图片文件，然后称为共享列存。

### 6.2 Planner 与 Reader

```text
for each preparation pool:
    map requested images to storage groups
    for each group:
        D_required = union(u+v for every requested source (u,v))
        for each d in D_required:
            if decoded cache contains this column: reuse decoded values
            else:
                reuse compressed cache or fetch full NPZ column
                decompress full column once
        gather exact image, component, spatial block and source frequency
    apply the common logical transforms
    restore occurrence logical slots and deliver model batches
```

必须计入：被需要的 diagonal 上未被请求的频率、同 group 未被请求的图片、同图 crop 外的 block，以及完整 NPZ header/central directory。没有额外索引时不允许声称 DEFLATE 内部任意空间跳读；不允许把 15 diagonal 偷换成 64 个精确 frequency column。

缓存 key 至少包含 dataset version、group、diagonal、format hash。compressed cache 与 decoded cache 分开计量、按 bytes 有界；pending 相同列请求应合并为一次实际读取。多个消费者读取同一 group 时共享 in-flight fetch 和必要列，完成后维持各自 occurrence/target 语义。

### 6.3 覆盖 FS 原本有利的场景

单模型测试不足以覆盖 FS 的贡献。加入完整 group/较大 batch、顺序读取、无 crop 或共享 crop、嵌套频率预算、多消费者的场景；同时报告随机小 batch/独立 crop 下的成本。

采用 1/2/4 个消费者作为最低版本，必要时扩到 8/12。至少包含完整 diagonal-prefix 预算（例如 B8 的 D=1/2/4/8/15）和同时请求同一存储 group 的场景，另测不同步消费者；不能只用触及 partial diagonal 的 zigzag prefix 代表 FS 的优势场景。固定总 CPU、GPU 和缓存资源，给 FS、FastLens 和可共享的原 JPEG 路径相同的共享机会。IC 每种静态专属格式的空间和物理文件读取按实际入账；IC shared variant 可以共享已有 source，不能人为禁用。输出实际 union bytes、sum bytes、每应用吞吐/完成时间和共享缓存命中。

## 7 公共接口与代码组织

以下是建议接口，不是对现有 FastLens 代码路径的声明。先定位当前 writer、CPU-DCT reference、profile resolver、训练循环与 counter，再以适配层接入，避免复制模型和 optimizer 代码。

| 接口 | 输入 | 输出与要求 |
|---|---|---|
| build_store | SourceManifest、StorageSpec | 有版本 StoreManifest；编码时间、字节和完整性报告 |
| plan | RequestTrace slice、ModelProfile | 真实物理 ReadPlan；必须包含读取放大和 output mapping |
| read | ReadPlan、IOContext | 压缩 buffers、metadata 和实际 I/O counters |
| decode | EncodedWorkset | 原始 q、index/geometry、类型、device 和生命周期 |
| materialize | q、trace、profile | 正确 model-ready tensor；记录变换设备与临时内存 |
| iterate | StoreManifest、frozen trace | 与现有训练/推理循环兼容的 ModelBatch |

建议模块划分为 `common/source_manifest`、`common/request_trace`、`common/profile`、`reference/jpeg_coeffs`、`reference/dense_transform`、`ic/native_codec`、`ic/matched_store`、`fs/diagonal_layout`、`fs/reader`、`runtime/io`、`runtime/staging`、`bench/runner`、`bench/counters`。公共 contract 和 reference 由一个 owner 维护，IC/FS agent 不同时修改。

基准命令的目标形态可以是 `dct_baselines build/verify/bench/report --config ...`，但包内没有实现这些命令。开发 agent 应先落实 contract，再创建 CLI；不能仅凭文档命令就记录“已运行”。

## 8 正确性验收门

任何 matched gate 失败的结果不能进入速度主表；保留失败日志，不自动放宽容差。

| Gate | 必须验证 | 通过标准 |
|---|---|---|
| G0 输入定位 | 仓库、source/profile/trace/checkpoint、版本、权限范围 | 所有必需输入真实存在；schema 中 null 已解决 |
| G1 格式和整数 | JPEG q/Q、component grid、所有支持类型、重排/压缩往返 | 原始整数逐项/流式 hash 完全相同，无 clipping |
| G2 依赖和变换 | 全源 reference vs 选择路径；crop/resize/flip、clamp/round 顺序 | 所有 required source 存在；整数阶段 exact，浮点阶段符合预先冻结容差 |
| G3 模型与训练 | final input、logits、loss、梯度、短期参数更新 | 相同模型状态与随机性，误差在既有/校准后冻结的门限内 |
| G4 计时和计量 | I/O bytes、读取单元、缓存、H2D、异步完成事件 | 小型可手算 trace 与日志吻合；计时包含首请求和末尾 drain |
| G5 正式实验 | 校准配置被冻结、实际 workload、重复运行和结果 schema | 无错误/丢样本/隐藏预计算；生成可追溯原始结果 |

必测边界：负系数、−129/−128/127/128、全零列、不同量化表、支持范围内 grayscale/4:4:4/4:2:0/odd dimensions、尾 group、partial diagonal、稀疏 K、跨 group 乱序、重复 image 的不同 augmentation、贴边/非对齐 crop、resize 依赖、cache eviction、损坏压缩成员、最后不满模型 batch。

浮点容差优先复用当前 FastLens 已有并有依据的门限。若不存在，使用不参与最终测速的 golden set，对同一个可信 reference 的 CPU/GPU实现做独立校准，冻结每个 profile 的绝对/相对误差、误差分位数和 mismatch rate。FP32 的 `atol=1e-5, rtol=1e-5` 可作为初始诊断值，不应被写成论文规定，也不保证适用于所有 resize/round 路径。

G3 至少包含一个 batch 的完整反向/更新和约 10 步复放。warm-up 后恢复 model、optimizer、scheduler 和 RNG state；Dropout、Mixup、gradient accumulation、distributed padding/tail 也必须一致。完整 50K validation 检查输入契约和预测；仅看到 top-1 接近不能代替 G1–G3。

原生小规模 gate 建议包含 `B32,k3` 的 `3×24×24` mosaic、`B8,k8` 仅 36 个有效系数、B32 的 63 列/B8 的 15 列、每个 (u,v) 恰好属于一条 diagonal，以及 IC/IC-b/FS 对相同 native 系数和 mask 得到相同模型 tensor。

## 9 实验矩阵和计时边界

### 9.1 最小主结果

优先选三个互补 workload：MobileNetV2-DCT24、ViT-Ti 或 SwinV2-T、ResNet50-DCT24。跑通后补齐当前稿的七个 DCT variants，分别在 RTX 4090 和 H100 上扩展已有表，不改变原模型/checkpoint。[FL Table 1、§4.2]

| 实验 | 规模和变量 | 主要回答 |
|---|---|---|
| 原生 sanity | 同一 B32/Q50；IC、IC-b256、FS；显式公布若干 k，如 1/4/8/16/32 | 格式复现是否保持相同内容和原生机制；这些 k 是本计划选择 |
| 严格 matched inference | 全部 50K validation；固定现有模型 batch；默认 warm compressed-file cache | IC/FS/FastLens 的端到端输入系统差异 |
| 严格 matched training | 完整一个训练 epoch；完全相同 trace 和起始状态 | throughput、更新数和训练语义一致性 |
| 存储与转换 | 全数据集；共同 JPEG 分母；profile 数增长 | 持久化代价、多副本、编码成本和摊销 |
| 频率/空间选择 | 输入微基准和真实 profile 分开 | 选择是否减少实际 bytes/decode/H2D，而非只减少最终 tensor |
| 局部性 | prep pool64/1024；共享/独立 crop；完整 group/随机小 batch | 跨图片布局的受益条件和退化条件 |
| 多消费者 | 最少1/2/4；嵌套频率预算；统一资源和有界缓存 | FS 的共享优势是否被覆盖 |
| 受控 I/O | warm pipeline、可核验冷读、125/500/2000 MB/s aggregate throttle 分表 | 计算/读取瓶颈改变时优势如何变化 |

原生 sanity 同时给两条空间账本：特定 mask 所需 FS 列文件字节与 IC/IC-b 专属文件比较；支持整个 budget family 的总持久化字节单列。若原生对照只含 k≤32，可用最大 diagonal 前缀32作为共同容量 anchor，完整63列库另报支持范围；不能以一方支持更多数据的整库大小和另一方单个 pruned 文件冒充同任务对比。

频率微基准可用显式 zigzag prefix `{8,16,32,64}`，另加同 diagonal 内不完整的 irregular mask；前提是不依赖全频 resize，或只测试 coefficient fetch。它不代表现有 checkpoint 在改变 source-K 后仍有相同准确率。真实模型空间实验可沿稿中 full/448/160 crop，用该 profile 的真实 source dependency，保持最终模型输入尺寸一致。[FL §4.4–4.5]

不要一次做所有参数的笛卡尔积。先主结果，再做能区分机制的角点：native/default 与共同位宽、per-image 与 IC-b、FS diagonal 与 per-coefficient 消融、full source 与选择读取、CPU 变换与 GPU 变换适配。

### 9.2 三个计时区间

1. **Input-only**：第一个被测逻辑请求进入系统，直到所有最终 CUDA model tensors ready。包含 planning、metadata/file read、decode、gather、H2D、变换、重排和依赖同步。
2. **Inference E2E**：第一个被测请求发起之前开始，到最后一个模型输出 CUDA completion；不得预先在计时外准备首个被测 pool。
3. **Training E2E**：同一完整 epoch 的上述路径，加 forward、loss、backward、optimizer 和最后异步工作 drain。warm-up 不应改变正式运行的起始状态。

共同 trace 的解析和通用索引可以统一在计时前准备；backend 特有的 group/diagonal 规划、请求合并、byte-range 构造必须在计时内。metadata 缓存状态统一，初始化成本另计；不能仅为某个 backend 免费预编译所有物理请求。

以完成同步后的 wall time 计算 throughput。串行 stage 时间可以用于诊断，但异步执行时不能将 stage 累计时间相加当 E2E，也不能相乘多个独立消融的 speedup。可另报告 steady-state，但必须与含启动/收尾的 closed-window 测量分开。

至少三次正式运行，提供 median、原始 run 值和离散程度；交替/随机执行方法顺序。只对接近 1× 且影响结论的差异增加重复，直到能合理判断或明确无显著优势。profiler 单独运行，不把其开销混入正式吞吐。

### 9.3 缓存和资源

区分 execution warm、OS page cache、软件 compressed cache、decoded coefficient cache、GPU tensor cache。主 warm 条件建议压缩文件已热，但仍完成正式 read/decode/transform；decoded 数据跨窗口缓存默认关闭或设共同有界预算。in-flight 合并和准备池内合理复用保留并计量。

冷读必须有实际 I/O 证据；若无法清缓存，采用受控工作集/文件级 eviction 并说明验证范围。单纯 pollute cache 不能自动标为冷盘。页缓存命中的 application read bytes 不能声称为 disk-device bytes。

固定总 CPU core、NUMA/affinity policy、pinned memory、host/GPU staging budget、模型 microbatch 与模型资源。存储 batch、preparation pool 和 model microbatch 分开记录；不能让 FastLens 准备 1024 样本却只准 IC/FS 看到 64。各方法可以在同一预算、同一 calibration split 上调 workers/prefetch/group size，发布 paper-default 和 best-valid-tuned 两种结果，最终选点不能偷看 test 性能。

### 9.4 必须保存的计量

- 身份：代码/build/dependency、source/profile/trace/checkpoint hash、方法/variant、GPU/CPU/NUMA、precision、batch、tail、worker、pool 和 cache 配置。
- 输入需求：occurrence 数、唯一图像数、逻辑 coefficient 请求数、去重后的 coefficient 集合大小。
- I/O：payload 与 metadata/index 字节、实际 application read bytes、可用时的 device/wire bytes、open/read/seek/request 数、cache hits/misses。
- Decode：实际解码 coefficient 数和 bytes，FS diagonal overread、空间 overread、跨图 overread，FastLens vector population/dependencies。
- H2D：实际系数、metadata、quantization tables、indices、labels、padding 的全部传输字节，另可列 payload-only。
- 资源：worker 合计 RSS/PSS、pinned peak、GPU allocated/reserved peak、输入管线增量显存、CPU core-seconds、模型等待输入的时间。
- 存储：logical file bytes 与 allocated disk bytes 分开，全部 index/metadata/headers、所有 per-profile 副本、临时构建峰值。
- 结果：elapsed、throughput、TTFB、batch latency 分位数、quality、correctness gate、原始日志路径。

所有 read/decode amplification 必须写明分母。建议 `A_decode=实际解码 coefficient 数/去重后实际需要的 coefficient 数`；同时报告基于 occurrence 的逻辑需求，避免共享数据使分母偷偷变化。compressed read bytes 与 logical int16 bytes 的比值同时受压缩影响，不要直接称纯读取放大；可分列“编码单元覆盖放大”和“压缩率”。

50 Mbps 的 FS 原论文网络图可以单独恢复为 `T_est=T_server_profile+8×bytes/bandwidth_bits_per_second`，标为 modeled，并给各可共享方法同样假设。FastLens 的 125/500/2000 MB/s 方法则是 aggregate completion throttle，标为 emulated。二者都不等同于真实 remote measured E2E；50 Mbps 是 6.25 MB/s，也不能直接与 125 MB/s 当同带宽比较。[FS §6.2.2；FL §4.5.2]

## 10 加强原生匹配证据的可选小实验

若担心 IC-style matched 仅保留 NPZ 输入路径，增加一个 **native IC anchor workload**：选择 IC 合法 B8、q50、k4 或 k8 配置，训练一次三通道 mosaic 模型；将同一批 native 量化系数分别交给 IC-NPZ 与 FastLens 存储，两边重建同一 mosaic、使用同一 checkpoint。FastLens 需要新增一个消费量化系数的 mosaic profile，不做 Q 乘回；该输入是 native IC 的契约，不是原 JPEG-DCT 七模型之一。

系数注入可以使用实际可用的 writer 中间接口；若必须通过 JPEG 容器桥接，要验证写入/读出 q 完全一致，且不能让默认 profile 自动再反量化。该实验须标明 FastLens 增加的 profile 与接入层。最低版本是 inference-only anchor：模型按原生 RGB augmentation 后再 DCT 的规则训练一次，再由两种存储生成相同验证输入。不能把离线 mosaic 当像素图直接 RandomResizedCrop 后称为原生 IC 训练；若扩展训练 anchor，必须重新验证整条原生增强顺序。它能提供更直接的原生 IC 表示匹配证据，但不复现 IC 搜索的全部效果。

对原生 FS B32，不能把 32×32 基底中的 1024 系数重新解释为 JPEG B8 的 64 个频率。最小方案是在 IC-native-B32/IC-b/FS-native 间验证相同 tensor 和压缩机制。若要将 FastLens 加入 B32 原生 matched 表，必须明确扩展列数或进行有证据的无损字段映射，增加 metadata 和 profile，并验证全部 1024 个系数；否则这一行不应存在。

## 11 准确率预算和离线成本

现有 matched 表因输入和 checkpoint 固定，不要求每个 loader 重新训练 300 epochs。充分的 source/tensor gate、全 validation 和短训练更新匹配即可检查输入等价性；完整 epoch 用于真实吞吐。FastLens 的共享 crop/局部 shuffle 是否影响最终质量是另一个问题，需要与独立采样策略的训练对照。[FL §3.4.2、§4.2.3]

只有要声称完整 IC/FS 部署也被超过时，再为原生方案训练/适配相应模型，按预先选定的质量损失预算（例如 0.5 或 1.0 个百分点，是建议值）比较 Pareto。训练集内留出搜索/校准集，最终 50K validation 不用于挑选存储/模型点。报告 architecture、params、input shape、FLOPs、训练 recipe、search GPU-hours 和最终 checkpoint。

离线成本分别记录共同 source preprocessing 和方法特定的 encode/search/train；包括 wall time、CPU core-hours、GPU-hours、读写字节、临时空间及输出副本。对于固定的复用次数 R，可报告 `C_method+R×T_method`。只有在单次在线确实节省时间时，才计算转换成本差额除以每轮节省时间的 break-even；不能混用 wall time 和资源费用。

## 12 Agent 分工与停止条件

| 工作包 | 主要产物 | 前置与完成条件 |
|---|---|---|
| A0 仓库和证据审计 | integration_map、版本锁、ASSUMPTIONS、数据/checkpoint 清单 | 不猜路径/参数；确认当前可运行 CPU-DCT reference |
| A1 Contract 和 oracle | SourceManifest、ModelProfile、RequestTrace、独立 dense reference | G0/G1；schema 和 trace 冻结后其余任务并行 |
| B1 IC 实现 | native codec、shared/fixedk/batched stores、reader | B/k/mosaic 单元验算；JPEG q roundtrip；明确 adaptations |
| C1 FS 实现 | diagonal writer/planner/reader、group cache、native/matched | 不加空间跳读；G1；partial diagonal 和读放大正确 |
| D1 Harness | IOContext、bytes counters、计时、资源限制、结果 schema | 与 B/C 并行；用手算 trace 通过 G4 |
| A2 语义集成 | 四/多个 backend 接入同模型循环 | G2/G3；明确 CPU/GPU transform variant |
| D2 校准与主表 | 冻结 tuning manifest、50K inference、完整 epoch | G4/G5；先三类模型后补齐七个 |
| C2 机制实验 | IC-b/位宽、选择粒度、低局部性、多消费者 | 共用同 budget；记录 default/tuned、实际/模拟 |
| R 研究复核 | 结果包、方法说明、偏差账本、可复跑命令 | 无错误 gate；数字从原始 results 自动生成 |

并行时最多让四个 agent 分别维护 contract、IC、FS、harness；先冻结公共接口再并行实现。后续语义集成和正式运行有依赖，不能跳 gate。完整 native search、B32 FastLens 扩展、真实远程服务是独立扩展任务，不能在 P0 尚不正确时优先投入。

停止条件是必需产物、正确性和计划内实验已经完成；不以“达到论文 10×/11×/14.2×”为验收，也不以“FastLens 必须赢”为调参目标。任何原方法优于 FastLens 的条件都应保留并解释。

## 13 建议补入论文的方法描述

以下内容应按实际完成的 variant 和实验填写，不能在实现前声称已经做到：

> We implement paper-guided versions of Image Calculator and Frequency-Store and adapt their storage organizations to a common JPEG-derived DCT input contract. All matched runs use identical source coefficients and quantization metadata, model checkpoints, logical request traces, and transformation semantics. The IC adapter uses image-oriented compressed arrays, with separately reported fixed-frequency materializations; the Frequency-Store adapter groups coefficients by DCT anti-diagonal over batches of images. We preserve each method's physical read granularity and account for metadata, overfetch, conversion costs, and materialized copies. Native representation experiments are reported separately from the matched input-system comparison.

同时补齐 ImageNet-512 制作 recipe、profile/源 K、workers/prefetch/cache 定义、checkpoint 来源和尾 batch 规则。当前稿 Table 1 的 checkpoint accuracy 与后续从头训练曲线的数值不能混作一个质量预算；共享 crop 的可收敛曲线也不能单独证明与独立策略同质量。若保留 time-to-target 主张，应给出 wall-clock 对照。[FL §4.1–4.2.3]

## 14 来源和可追溯信息

### 附件版本

| 标记 | 附件及定位 | SHA256 |
|---|---|---|
| FL | FastLens 稿；原附件 YuxinTang (27).pdf；16 页 | 53fd4ef193b84e9e5833b57936b244fe296620a54538de5954dc17e084d3832f |
| IC | The Image Calculator；SIGMOD 2024；31 页；DOI 10.1145/3639307 | 6dd24eb9a7840d520f86d870846b8c269b2e6dc3af818ff3534c622fa182c143 |
| FS | Frequency-Store；CIDR 2025；原附件 p27-sirin.pdf；8 页 | f28025c27771ce7a0e8ad0d1308d25d453ccb02afe89bdb70aff17a527804771 |

### 已读取的公开原始资料

- W1：[作者 publications](https://www.utkusirin.com/publications)，本次核查 IC/FS 的论文、slides、tech report 入口和公开源码链接状态。
- W2：[作者 SIGMOD 2025 tutorial](https://www.utkusirin.com/sigmod25-image-ai-tutorial)，用作后续公开说明；与2024候选数不一致，未覆盖原论文。
- W3：[CIDR 官方 FS 页面](https://vldb.org/cidrdb/2025/frequency-store-scaling-image-ai-by-a-column-store-for-images.html)；[FS 官方 PDF](https://vldb.org/cidrdb/papers/2025/p27-sirin.pdf)。正文实现依据为附件。
- W4：[NumPy savez_compressed 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html)，NPZ、NPY、ZIP_DEFLATED 语义。
- W5：[TorchJPEG coefficient API](https://torchjpeg.readthedocs.io/en/latest/api/torchjpeg.codec.html) 与 [读取量化系数说明](https://torchjpeg.readthedocs.io/en/latest/dct_rw.html)，直接取已有 JPEG q/Q，避免 RGB 往返。
- W6：[TorchJPEG DCT 实际源码](https://torchjpeg.readthedocs.io/en/latest/_modules/torchjpeg/dct.html)，可变 block_dct 与固定 B8 convenience functions。
- W7：[IC ACM DOI 页面](https://dl.acm.org/doi/10.1145/3639307)，原论文版本入口。
- W8：[合作者 Florian Klein CV](https://florianklein.me/cv)，相关 core components 有 Rust/C++ 开发的作者自述；不能据 NumPy/PyTorch 描述断言整个原系统纯 Python。

作者链接的 IC TechRep、slides 和 2025 DEBull 综述已定位，但本次没有取得可读全文，不将其作为未披露实现细节的依据。后续若取得原始代码或 technical report，先更新证据/偏差账本，再决定是否改变配置；不同版本保留结果身份，不能静默覆盖。
