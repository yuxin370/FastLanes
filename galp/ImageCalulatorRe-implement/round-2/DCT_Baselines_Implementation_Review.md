# FastLens DCT baselines 实现审查

## 结论

上传版本已经实现 IC/FS 的核心存储格式、读取参考路径、原生机制检查，以及相当完整的数值验证框架。现阶段适合继续开发和做小规模诊断，尚不具备进入论文 matched 性能主表的条件。

本次在原样存储代码上执行的 9 组 NumPy/SciPy 检查全部通过。另确认了两个应优先修复的问题：验证证据没有绑定实际 NPZ 内容；单次请求池的活跃解压工作集不受缓存字节预算限制。训练 schedule、CPU 计时和结果聚合也存在确定的逻辑问题。

README 本身将全部当前计时标记为 `headline_eligible=false`，并明确披露 CUDA、G4/G5、完整验证集和 epoch 尚未完成。应保留这一边界，不能把下文的局部测试通过解释成完整模型、JPEG 或 GPU 验收通过。

## 审查对象和证据范围

- 输入：`dct_baselines.zip`，164620 bytes。
- 输入 SHA-256：`bf477c5f5a6309e756948d6e2c2598c743ecc95a47bd0fc87239f0de405f26cd`。
- 有效源文件：14 个 Python 文件、2 个 Markdown 文件；归档内另有平台元数据和字节码。
- 源码中声明了 25 个测试方法。原测试套件在导入两个测试模块时因缺少 PyTorch 而失败，没有执行这些测试体。
- 当前环境还缺少完整 `galp` 父仓库、RGB-no-more/DCTNet/FUN 源树、JPEG 扩展、模型与 checkpoint。本附件不是独立可运行的完整仓库。
- 审查保留上传源码原样。复现脚本和结果单独放在 `review_20261009/probes_*`。
- 对照标准为上一轮 FastLens matched DCT 基线计划。审查针对上传字节，不评价未随包提供的其他 worktree 修改。

### 本次实际执行的检查

| 检查 | 结果 | 能证明什么 |
| --- | --- | --- |
| 全部 14 个 Python 文件的 AST 解析 | 通过 | 当前解释器能解析源码 |
| 原始 unittest discovery | 两个测试模块导入失败，缺少 torch | 原始 25 个测试未在本环境运行 |
| 4 种 matched store 的独立往返 | 通过 | 合成 q/Q、ragged grid、部分灰度、尾组、重复乱序、空间 slice、频率顺序正确 |
| 反对角线分区和原生 mask 精确识别 | 通过 | B8/B32 的 15/63 列划分和三角形边界正确 |
| FS 部分反对角线及空间请求的读放大 | 通过 | whole-column 解压与需求统计符合该 API 的语义 |
| decoded LRU 命中与淘汰 | 通过 | LRU 自身的字节预算、命中和淘汰正常 |
| signed dtype 边界与溢出 | 通过 | 已覆盖值安全窄化，溢出明确失败 |
| 原生 DCT 与 SciPy 独立公式、IC/IC-b/FS mosaic | 通过 | 已选 B8/B32 小样例和原生格式机制一致 |
| NPZ 内容更改反例 | 缺陷复现 | q/Q 可改变而 store.json 身份不变，reader 接受 |
| 零缓存活跃数组生命周期 | 缺陷复现 | 多个完整解压组同时保留；不是 RSS 或 OOM 测量 |
| schedule 和 trace 纯函数探针 | 缺陷复现 | 更新数公式与 pool 边界规则不一致，推理标签校验缺失 |
| runner 控制流与 report 聚合探针 | 缺陷复现 | CPU 窗口错位、身份合并/丢失及精度契约问题 |

存储探针只为缺失的 `galp.benchmarks.common.sha256_file` 提供标准 SHA-256 辅助函数，实际 `contracts/store/native` 代码不变。runner 探针从原文件 AST 提取函数，并用明确的 stand-in 隔离外部 pipeline；它不是实际 PyTorch/CUDA 测试，schema 也只检查必需字段。100/900 images/s 等数值是聚合反例的人工输入，不是 baseline 性能结果。

## 已完成的机制

### IC

`store.py` 已实现单图 `ic_matched_shared`、显式依赖 `ic_matched_fixedk` 和默认 256 图像的 `ic_matched_batched`。批量版本使用一个连续的 q 数值成员，能够利用跨图像压缩；Q 使用 uint16。`paper`、统一 int16 和范围安全 `safe` 策略存在。

FixedK 会要求非空依赖说明，拒绝请求未存储频率；`CoefficientView.dense()` 不会将缺失频率擅自补零。`mask_family` 通过集合精确相等识别原生三角形，all64 正确归为适配扩展。

### FS

`store.py` 实现了跨整个存储 group 的反对角线列，B8 为 15 列，B32 为 63 列。每列分别 NPZ 压缩，按该列完整数据范围选择 int8/int16，Q 放在独立组文件中。读取完整触及列后再 gather，没有加入每个频率一个文件、空间 seek、bitpacking 或其他压缩器。

独立样例中，两图一组，对图 0 的三个分量各请求一个空间块的频率 1，并重复请求：实际解压 80 个 q，occurrence 需求为 6，去重需求为 3，只打开一个列文件和一个 Q 文件。此行为保留了 FS 的整列读取语义。

### Native 和 matched 的区分

`native.py` 保留 B、sampling、标量量化、`u+v<k`、k×k 结构零和三平面 mosaic。候选枚举为 6048；B8,k8 为 36 个有效系数。B8 与 B32 小样例的量化 DCT 与独立 SciPy DCT-II 相同，IC/IC-b 与 FS-native 重组的同 mask mosaic 相同。

颜色变换、level shift、padding、rounding 等作为工程选择明确记录，没有冒称恢复了作者源码。没有完整 IC self-design 优化器这一点也已披露。

### Matched 契约与数值框架

现有模型 profile 明确固定 source512、4:2:0 和所有 64 个源频率；不会把输出 DCT24/32 名称当作源 K。Trace 包含 occurrence、逻辑位置、crop/flip、具体增强参数、Mixup 伙伴和双权重。代码还实现了完整 model/optimizer/decay/scheduler/RNG 快照和逐更新比较。

这些是可继续使用的实现基础。真实 JPEG 提取、上游模型接口以及 CPU/GPU 数值结果仍需要在完整环境中验证。

## 应优先修复的问题

### R01 P1 正确性证据没有绑定实际 NPZ 内容

位置：`store.py` 的 `build_store` 第 120、135、140 行；`_decode_file` 第 263 行；`runner.py` 第 168、207 行；`unified.py` 第 59 行。

Manifest 记录文件名、dtype 和元素数，但没有各 NPZ 内容散列。`verify` 的 store 身份和 `benchmark` 的过期证据检查都只使用 `artifact(root / "store.json")`。

已实际复现：构建合法 store 后，以相同 shape/dtype 重写 NPZ，把一个 q 从 -129 改为 -128，把一个 Q 从 32526 改为 1。NPZ 散列改变，store.json 散列不变；原样 StoreReader 接受并返回更改后的数据。

StoreReader 行为已执行验证；旧 evidence 仍满足 runner 的 store 身份条件，是读取第 207 行判断式得出的静态结论。本次没有运行依赖 PyTorch 的完整 benchmark。训练 `verify_training` 每次会重新执行 G1，不应把它说成完全跳过内容检查，但其持久 backend identity 也不能代表所有 payload。

影响：文件被重新生成、覆盖或错误恢复后，旧 passing evidence 无法保证被计时数据仍是已验证内容。ZIP CRC 只验证当前压缩流是否自洽，合法重写的 NPZ 会有新的 CRC。

修复：构建时保存全部 q/Q 文件的内容散列和大小，生成覆盖完整 store 的 fingerprint；verify、bench 和结果表使用同一个内容身份。测量前检查身份，散列核验成本单独报告或按统一规则计时，并保持数据在测量期间不可变。

验收：同 shape/dtype 重写任一 q 或 Q 文件时，旧证据必须被拒绝；未修改文件的正常运行继续通过。

### R02 P1 缓存预算没有限制活跃解压工作集

位置：`store.py` 的 `StoreReader.read` 第 284 行；全部 misses 解码阶段第 296 行；gather 阶段第 311 行。

一次 read 会先将请求池触及的全部 group/column 解压到局部 `decoded` 字典，再进行所有请求的 gather。`cache_bytes` 只约束 `self.cache`，局部解压数组和已完成 future 的结果不受其限制。

已实际复现：IC-b group=4，共四组，cache_bytes=0，每组只请求一个 q，总需求 4 个 q。用弱引用观察 `_decode_file` 的实际返回数组，四组完整 q 同时存活，共 16384 bytes；LRU 大小仍为零。这是数组生命周期证明，不是实测 OOM。

按 source512、4:2:0 推导：单图 q 数为 `(64*64 + 2*32*32)*64 = 393216`；IC-b int32 每图 1.5 MiB，group256 的 q 为 384 MiB。随机池触及许多组时，活跃内存可能达到数十乃至更多 GiB，实际还会有输出、metadata、future 和下一池的开销。这些是容量推导，不是已测峰值。

修复：以有界 group/column 批次执行 decode → scatter 到本次所需输出 → 释放；限制 in-flight futures 和活跃 decoded bytes。LRU、活跃解压、最终输出与预取分别统计，共同约束资源预算。保持整列读取和读放大计数，不需要为 FS 添加空间索引。

验收：随机跨组、cache=0 的压力测试中，活跃 decoded bytes 受显式预算约束；随着 touched groups 增多，峰值不再按全部完整组线性保留；q/Q 与统计仍正确。

### R03 P2 默认 schedule 与 pool 边界更新数不一致

位置：`__main__.py` 第 19、79 行；`training.py` 的 `update_windows` 第 133 行。

默认 schedule 用 `ceil(population / (batch_size * accumulation)) * 300`。实际窗口在 pool 或 epoch 边界提前结束 accumulation，而 CLI 只要求 pool 是 microbatch 的整数倍。

| population | batch | pool | accumulation | 默认每 epoch 更新数 | 实际窗口数 |
| --- | --- | --- | --- | --- | --- |
| 8 | 2 | 2 | 4 | 1 | 4 |
| 8 | 2 | 8 | 4 | 1 | 1 |
| 100 | 4 | 12 | 4 | 7 | 9 |

这些数值来自原 `update_windows` 纯函数执行。第一行的 300 epoch 默认 schedule 是 300 步，按实际窗口则为 1200 步。改变 pool 还改变了有效更新 batch，因此不能只按输入吞吐敏感性解释。

该缺陷适用于上传包的 recipe 生成与窗口划分组合；未提供的正式训练器是否相同，需要完整 patch 才能确认。校准/验证子集本来就可以使用完整训练计划的 schedule，不能简单改成短 trace 的长度；应基于声明的 schedule population，采用同一 pool/tail/update 规则推导，并记录显式 override。

验收：默认配置、pool 小于累积窗口、不能整除的 pool、epoch 尾部及显式 total_updates 都有一致的期望更新计数。若论文比较不同 pool，应保持目标训练语义或单独披露有效 batch 的变化。

### R04 P2 CPU core-seconds 与 wall-clock 窗口不一致

位置：`runner.py` 第 226、232、286、295、304 行；`store.py` 的 `storage_bytes` 第 349 行。

CPU 起点早于 reader 初始化；wall 起点在初始化之后。wall 结束后，代码先生成 result record，其中会遍历整个存储目录、打开 NPZ 查看成员、计算 artifact hash，最后才计算 CPU 差值。

隔离探针在计时外初始化注入 25ms CPU 工作，在计时后 storage accounting 注入 75ms CPU 工作，确认这 100ms 均进入 cpu_core_seconds，却不进入 elapsed_wall_s。探针数值只用于证明控制流边界，不能作为 baseline 性能。

影响：IC 单图格式和 FS 的文件数不同，计时外全库扫描成本也不同，CPU 利用与资源效率比较会受到不对称污染。CPU 时间大于 wall 时间本身不是错误的证据；这里的问题是已知窗口外工作被计入。

修复：在同一正式窗口捕获 CPU 和 wall 起止；reader 初始化、验证哈希和存储容量扫描另计。最终 CUDA 完成后先结束两种计时，再生成统计报告。

验收：向窗口外注入 CPU 工作不改变正式 CPU 指标，向窗口内注入工作能够被统计；原始运行仍包含首池和最终 drain。

### R05 P2 report 会混合不同实现的运行并丢失分组身份

位置：`runner.py` 的 `report` 第 368、375 行。

分组 key 不含验证证据中的实际 implementation/store fingerprint。代码更新后重新验证，但继续使用同一配置与同一 git HEAD 时，优化前后记录可能被当成重复实验。

隔离反例使用相同 config/checkpoint/trace/runtime、不同 implementation/store 证据，人工设定 100 和 900 images/s。原 report 输出一个组、repeats=2、median=500。这些数字是聚合测试输入。

另一个反例把 config/checkpoint/device 改为不同，report 内部正确分成两组，但输出 summary 丢掉了这些身份；两行除吞吐外完全无法区分。

修复：在 raw schema 中显式保存实际 store、实现和 native binding 的语义 fingerprint，按这些字段分组；summary 输出 model/profile、device、checkpoint/config/store/code identity、batch/pool/precision 和 run IDs。同步更新 schema，避免临时增加未声明字段。

验收：同一实现的真实重复能合并；代码或存储变化必须分组；不同模型和 GPU 的 summary 行可以独立识别并追溯到 raw runs。

### R06 P2 training operation_order 与实际执行顺序不一致

位置：`materialize.py` 第 45、61 行；`__main__.py` 第 38 行；`training.py` 第 81、116 行。

prepare 先生成推理顺序，其中 CNN 已含 select_channels/normalize，Transformer 已含 affine，再在尾部追加 RandAugment、model_normalize 和 Mixup。实际训练是 raw 几何 → 训练范围 round/clamp/int16 → RandAugment 及逐步 clamp → organize 中的投影/归一化 → Mixup。

该问题是冻结描述与执行不一致；没有证明当前各 backend 之间因此已经产生数值差异。它会使后续 agent 按契约实现另一条路径时选择错误的运算顺序。

修复：为 inference 和 training 分别生成真实的完整 operation_order，记录实际训练 affine、投影和 rounding/clamp 边界，避免追加推理列表。

验收：保存的操作顺序与 instrumented 执行轨迹一致，CNN 和 Transformer 均有检查。

## 次要契约问题和验证缺口

### 推理标签和校准 trace 验证不足

`contracts.py:108` 的推理 trace 验证没有 target_label 的类别范围检查，也未核验冻结 label mapping。独立反例中，source label=0，target_label=-1、1000、999 都被接受。正常 prepare 的排序映射不一定有错；问题是错误 trace 可被接受且两边共同使用同一个错误标签。

应保存带 hash 的完整 label mapping，统一验证整数类型、类别范围和 image_id 对应关系。CNN 有意进行 sorted-class 映射时，不应简单与原 source label 比较。

`gates.py:126` 的校准 trace 只调用 load_trace，没有执行 `runner.inputs` 对正式 trace 的全局 microbatch 编号、batch size、epoch tail 检查。后续用 microbatch_id 作字典 key，外部构造的重复编号可能覆盖批次。正常 freeze_trace 会生成全局递增编号；这一风险未通过真实 Torch 端到端复现。应抽取同一个 validator 用于两个 trace。

### 训练 oracle 只能独立覆盖部分环节

`unified.py:159` 的两条路径最终都调用 `finish_batch`，共享 organize、Mixup 和 soft target 构造。它能很好地检查 backend 的读取及 raw geometry 差异，但 G2/G3 通过不能独立证明这些公共变换本身正确。

现有两样本常量 Mixup 手算测试有价值，应增加非平凡 normalization、非对称通道和独立 target golden。`tests/test_baselines.py:229` 的 CNN 几何测试将 448px crop 的 56 个块输出到 grid56，Y resize 为恒等情况，Cb/Cr 又全零；应补非恒等 resize、非零 chroma 和 rounding 边界。

### 当前空间 region 能力没有接入真实 replay

`runner.py:70` 和 `unified.py:127` 的请求只传频率，不传 regions。因此 storage API 支持空间 slice，实际模型 replay 仍请求完整空间网格。

不能据此说模型输入一定错误；crop 在 materializer 中仍会执行。但这没有完成原计划中实际 G_source 的请求表达，requested_q 分母也是实际全网格请求，而非模型最小空间需求。应将两种需求明确区分，再接入经过验证的空间闭包。

### 推理 BF16 配置被接受但没有执行

`runner.py:45` 接受非训练 bf16-autocast；benchmark 只有 inference_mode，没有 autocast，结果第 326 行固定写 fp32。隔离探针确认这一接受路径。prepare 当前的推理默认 FP32 不会触发。

当前只支持 FP32 时应拒绝其他精度；若支持 BF16，则 verify 和 bench 必须使用同一策略并报告实际值。

### 两项低优先级问题

- `store.py:144` 对 IC/FS 都写 `ascending-u-within-diagonal`。IC 实际遵循 stored mask 顺序，应分别记录；reader 目前按 masks 工作，已测数值正确。
- `runner.py:268` 创建 CUDA first_event，但未将其转换为 first_batch_latency，CUDA 结果保持 null。应继续标记未测，补齐后再报告。

## 仍未交付或未完成的关键实验工作

### 正式训练接入修改未随附件提供

README 和 ASSUMPTIONS 提到 `training_jpeg.py`、CNN/Transformer 正式 runner、pipeline、报告器、`dctnet_storage.cpp`/bridge 等改动。ZIP 只有 dct_baselines 目录，没有这些 patch，也没有包外 result schema。

因此可以确认本包中的 API 和调用意图，不能确认 README 中所有外部集成都已正确落地。下一版应提交明确 base revision 上的完整 diff，连同构建命令、依赖版本和必要小型 fixture；无需提供整个 ImageNet 或大 checkpoint。

### 正式 DataLoader 仍会逐样本重复读取完整 group

README 第 248 行明确说正式 adapter 按样本读取完整 NPZ member，没有 decoded cache；`training_store_identity` 也记录 `whole-member-per-sample`。IC-b 和 FS 会重复读取/解压同一个 group。

这是已披露的诊断集成限制，应优先完成 batch/pool 级 read-many、请求合并和有界工作集后再测吞吐。不能将 pooled bench 的较优 counters 套用到这条正式 DataLoader 路径。

### 数值与性能证据仍需要真实环境闭环

README 报告三个实际模型在合成 JPEG 上完成十次 CPU 更新，以及三个真实图像的 FastLens 整数检查。但附件没有相应 trace、source manifest、evidence JSON、日志或 checkpoint identity 结果，本次无法独立复核。

本包的 bench 仅支持 input_only/inference；正式训练 benchmark 依赖未交付的外部 runner。CUDA 校准、G4 测量审计、等预算多消费者、缓存校准、完整 50K 验证与完整训练 epoch 均不能据此认定完成。

## 相对原开发任务的完成度

| 原任务 | 当前判断 |
| --- | --- |
| T01 仓库与真实配置梳理 | 有详细集成说明；父仓库和实际产物未交付，待复核 |
| T02 契约与 oracle | 有实质实现；需修内容身份、schedule、operation_order 和 trace 验证 |
| T03 IC adapter | 核心存储已实现，独立合成往返通过；正式高效训练接入待完成 |
| T04 FS adapter | 核心列存已实现，独立机制检查通过；活跃工作集和正式批量读取待完成 |
| T05 harness | 诊断推理 harness 已实现；CPU 窗口、聚合和完整计量未验收 |
| T06 数值集成 | gate 代码较完整；真实 CPU 声明缺证据包，CUDA 未验收 |
| T07 调优与机制实验 | 局部功能测试存在；资源/缓存/多消费者正式实验未完成 |
| T08 全验证集 | 未见可复核的完整运行产物 |
| T09 完整训练 epoch 和发布证据 | 未见可复核的完整运行产物 |

不建议用代码行数或单一百分比概括完成度。格式核心与论文实验的就绪程度差别很大。

## 下一轮 agent 的执行顺序

1. 提交完整集成 patch、base revision、依赖/构建说明及真实 CPU 检查的原始证据；保留 diagnostic 标签。
2. 修 R01 内容身份和 R02 有界工作集。这两项分别保护数值证据与大规模可运行性。
3. 并行修 R03 schedule、R04 CPU 窗口、R05 report 身份、R06 operation_order，以及 trace/精度校验。
4. 将同一个批量/有界 reader 接入实际正式训练路径，确认 FS/IC-b 的 group 复用，计量就在实际执行路径采集。
5. 在完整环境运行原测试和上述回归，增加非恒等 resize、非零 chroma、公共 Mixup/normalization golden；随后做目标 GPU 的校准和十次更新。
6. 通过 G4 后先跑三个代表模型的完整验证集和训练 epoch，再扩展七个工作负载、共享消费者、缓存/带宽与第二 GPU。

下一次可以进入正式性能测量的最小条件：所有被比较路径的实际内容身份固定；活跃内存有界；训练 schedule 和输入契约一致；真实 GPU 数值 gate 通过；计时及结果身份可追溯。完整 IC 搜索器与质量 Pareto 仍可作为后续独立扩展。

## 复现材料

配套 `DCT_Baselines_Review_Pack.zip` 包含本报告、未修改的被审源码、3 个独立探针、原始探针输出、环境及 SHA-256 清单，并包含 runner 聚合探针所需的原计划 schema。

探针覆盖范围在每个脚本头部和结果 JSON 中注明。它们不是正式 benchmark，不应将 fixture 中的人工吞吐或 passed 状态导入论文结果表。
