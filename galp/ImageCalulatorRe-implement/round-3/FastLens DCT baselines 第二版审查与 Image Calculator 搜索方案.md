审查日期：2026-10-09。

**结论：第二版可以认可为 JPEG-DCT matched 的 IC/FS 存储与参考输入路径。上一轮 R01–R06 已有实质修复，正式训练接入也已有完整补丁。当前适合进入共同真实源上的 GPU 集成验收与性能调优，尚不具备论文主表资格。Image Calculator 的原生表示工具已经实现，但最佳表示搜索、候选训练、预算选择与最终部署还没有实现。** [S1–S5]

建议保留现有存储主干，先修正默认实验条件和 cache 元数据，再补齐生产配置的数值与计量验证。论文保留严格 matched 存储对照，并增加一条 IC-native 的准确率预算实验，直接回应 IC 可以自行选择表示的能力。[S2–S6]

## 1 审查版本与证据范围

对象为 `DCT_Baselines_Round2_Response.zip`，SHA-256：

`a09932d2f069b9932c849909e72268243073094188d98f6746b9d32c5087a178`

提交声明的父仓库 revision 为 `13fabad2b337945ea32d50dd6b6cbebc5563b031`。本包包含完整 integration patch、CPU/H100 JSON 与日志、小型 JPEG/NPZ/trace fixtures；没有完整父仓库与大模型快照。审查从完整 new-file hunks 重建了 `dct_baselines` 文件，既有文件的修改按 patch 静态检查，没有声称已经应用并编译整个父仓库。[S1]

三类证据分别处理：

1. **独立执行**：原始 store/contracts 的 NumPy 探针；真实数组弱引用生命周期；合法 q/Q 重写拒绝；上传 NPZ 的内容与 declared source q/Q 对照；哈希链核查；原始 update 函数的纯函数检查。

2. **静态确认**：CNN/Transformer DataLoader、C++ bridge、训练/报告契约、CPU 计时窗口、独立 normalization/Mixup 组织等补丁。

3. **提交者运行记录**：34 项宿主 unittest、9 组 CPU 与 9 组 H100 verification。审查环境没有 Torch、dct_manip 和 GPU；这些运行没有在本环境复跑。[S1–S5]

## 2 上一轮问题的关闭情况

| 问题 | 第二版实现 | 本次结论与证据 |
| - | - | - |
| R01 内容身份仅覆盖 manifest | 每个 NPZ 记录 size/SHA-256；布局及 payload 清单进入 fingerprint；计时前 preflight；实际 decode buffer 再校验 | 核心问题关闭。IC/FS 的 q 与 Q 各作合法同形同 dtype 重写，preflight 和 read 共 8 次拒绝旧内容。源码 `store.py:61–69,160–164,265–271,318–325`；[S3] |
| R02 完整解压数组累计驻留 | 输出、LRU、active decode 分预算；按字节与 worker 数切 chunk；worker 内 scatter，uncached 完整数组不随 future 返回 | 核心问题关闭。8 group 弱引用探针验证并发峰值受控，返回后完整数组全部释放。源码 `store.py:343–437`；[S3] |
| R03 pool/tail 下 schedule 不一致 | 用全人口、microbatch、pool、accumulation 推导 updates；明确 total_updates override | 关闭已检查的计算缺陷。8/2/2/4→4，100/4/12/4→9，1,281,167/64/4096/16→1252。标准配置原先本来也得到1252；不能说标准实验曾必然跑错。[S4] |
| R04 CPU/wall 时间窗不同 | 同时开始，在 metric collection 和最终 CUDA drain 后同时结束，初始化另记 | 静态修复成立；提交者控制时钟回归支持。实际 GPU 首批延迟及完整计量仍待 G4。源码 `runner.py:253,311–326`；[S1,S2] |
| R05 聚合丢失或混合身份 | bundled 0.2-diagnostic schema；store/code/binding fingerprint 入分组，保留 model/device/config/checkpoint/raw refs | 诊断报告缺陷已修。正式 headline 的完整资格检查仍单独验收。源码 `runner.py:336–430`；[S1,S2] |
| R06 training 操作顺序与声明不同 | training/inference 分别生成 order；geometry 后 integer/clamp，再 augment、normalize、Mixup | 静态实现合理；独立 normalization/Mixup/soft-target 路径与提交者 goldens 支持。源码 `materialize.py:28–45`、`training.py:131–168`；[S1,S2] |

本轮也补齐了 label mapping 的身份及合法性检查、校准/验证 trace 共用 validator、真实 crop 的 component regions、推理 BF16 明确拒绝、IC/FS serialization order 的区分。[S1,S2]

### 2.1 实际内存探针

32 张图、8 个 group、每组4图，cache=0。并发用 barrier 强制两份实际解压数组同时存活，峰值来自弱引用跟踪，不只来自 reader 自己的计数器。[S3]

| 配置 | 解压准入预算 | 实际完整数组活跃峰值 | read 返回后存活 | group opens |
| - | -: | -: | -: | -: |
| 1 worker | 19,968 B | 19,968 B | 0 B | 8 |
| 4 workers，最多同时两成员 | 39,936 B | 39,936 B | 0 B | 8 |

另外验证了 IC 单图、IC-b、FS 的 ragged 分量、重复/乱序样本、空间子区域、部分且乱序频率、非平凡 uint16 Q 和并行 scatter。所有 q/Q 精确；全缓存重复读取的 file opens 与解压 q 字节为0。[S3]

这些预算约束 retained arrays。它们没有证明 Python/NumPy/Torch 临时空间与多 worker 的整机 RSS 上限；README 已明确这个边界，应该保留该限定。[S1,S3]

## 3 上传 CPU 与 H100 证据的可信范围

### 3.1 本地字节一致性核查

对18份 calibration、18份 verification 及其递归引用检查得到：[S4]

- 36条记录均声明 passed；每条引用的11个 baseline 实现文件与当前 patch 重建内容一致，共396次模块哈希比较。

- 553次可定位引用的哈希核对全部一致，没有 mismatch。

- 36条快照引用涉及6个未附的大快照；26条外部依赖引用涉及13个未附路径。这些标记为不可核对，没有计作通过。

- 三个主 evidence stores 的20个 NPZ：长度和SHA全部匹配。

- 从三个 store 读回的27个分量 q 哈希和27个 Q 均匹配 source manifest 声明。

- 每个模型的 CPU/H100 记录都使用同一个 initial-model-state SHA。[S4]

这证明附件内部的代码、输入和记录一致性较好。q 与 JPEG 的独立系数提取、本机模型执行、FastLens GPU 路径及完整父仓库构建仍不在这一哈希核查结论内。

### 3.2 H100 测试确实做了 GPU 模型更新，但规模很小

提交的 H100 verification 覆盖 MobileNet24、ResNet24、ViT-Ti 各自对 IC、IC-b、FS 的九个组合。所有组合记录10次更新，inputs/logits/loss/gradients/parameters/optimizer 最大差为0，input/model/state 的 atol、rtol 都为0。两个 CNN 是 BF16 autocast，ViT-Ti 是 FP32。CPU还有相同九组验证。[S1,S5]

真实规模由复现脚本和 trace 确认：

- 3张随机生成的512×512 JPEG；1张用于校准、2张用于verification。

- microbatch=2、pool=2；10个epoch，总20个verification occurrences、2个unique images。

- IC-b/FS 的 storage group=3；IC 单图 group=1。

- 配置 accumulation=16，但每个 pool 仅一个 microbatch，pool边界就提交更新。

- 本地执行原始 `update_windows`，三份trace均得到10个窗口，每窗口1个microbatch、2个sample。[S1,S4,S5]

因此现有结果支持“小样本实际模型在GPU上的数值gate”，未覆盖生产batch64、group256、pool4096及实际16次microbatch accumulation。没有全量ImageNet准确率、吞吐或收敛结论。

### 3.3 FastLens 尚未进入这一共同 GPU gate

IC/FS 的 UnifiedReplay 在CPU完成读取与几何变换，再将输入传到GPU执行augmentation、normalization和模型。原生 FastLens 使用另一条CUDA pipeline；它虽已有代码分支，本轮九份H100证据没有运行该分支。3张合成图没有真实 `galp_image_id`。提交者明确将“真实共同源的 FastLens CUDA 对照”列为下一步。[S1,S2,S5]

CUDA normalization 修复将 oracle 的 mean/std 显式设为设备FP32 tensor，同时保留逐通道独立计算与零容差，属于明确算术契约的合理做法。本次核查确认了代码及修复后证据一致性，没有独立重编译并验证提交说明中的 PyTorch kernel 原因。[S1,S2]

## 4 正式实验前仍须处理的事项

### 4.1 先解决默认实验条件不一致

`cnn_training_baselines.sh` 应用后第48行默认 `CONDITION=A0`，第57行仅让 native 使用B6。训练Dataset把B6映射为per-PLS crop，其他条件为per-sample crop。因此使用默认脚本直接比较 IC/FS 与 native，crop条件并不相同。[S2：integration.patch第8、17、393–394行]

这不表示适配器不支持B6；手动选择同条件可以对齐。建议增加独立的 matched launcher，并让成对实验核验 source、occurrence、crop、augmentation/Mixup、profile、model/state、precision、pool/accumulation/recipe 身份。保留原有 A0/B6 实验用于其原先研究问题，避免修改旧实验含义。

### 4.2 全库预检后不能默认标 cold

正式 Transformer入口新增 `reader.verify_content()` 全库读入，但环境元数据仍保留“epoch1 cold/order diagnostic，epoch2 warm-primary”的描述。全库读取会改变cache状态；即使数据大于RAM，也不能据此断言全库全热，更不能未经验证就称cold。[S2：integration.patch第1223、1243行]

改为“post-verification/order diagnostic；cache state按实测声明”。只有在全库校验后执行并验证同一冷缓存流程，才使用cold标签。README已经给出了正确限定，正式输出需要同步。[S1,S2]

### 4.3 需要同源生产配置的 GPU 数值验收

选择真实共同source manifest，确认JPEG producer recipe、图像ID和FastLens原始q/Q；同时运行 CPU oracle、FastLens、IC、IC-b、FS。按所选生产profile使用实际精度、batch、group、pool、accumulation。

最小新增数值fixture应包括至少一个完整pool、真实16个microbatch的累积窗口、多个窗口、尾部样本和尾部更新，并覆盖非恒等resize、非零chroma、crop/flip/augment/Mixup。保留现有小fixture作快速回归；它没有必要重复充当生产规模证据。[S2,S4,S5]

`training_store_identity()` 当前检查ID、label、path与all64 masks，不能仅靠path证明现时JPEG字节或FastLens q相同。生产实验应绑定实际共同源内容验证及校准证据；当前所有结果保持 `headline_eligible=false` 是合理的。[S2：store.py30–46]

### 4.4 让 FS 和 IC-b 获得合理调优条件

两条正式 `__getitems__` 现在按真实microbatch合并读取，这是有效改进。但每个DataLoader worker仍创建独立reader，默认cache=0、内部1个decode worker。同一个256图group在跨microbatch/worker时会再次解压；没有实现跨消费者共享。[S2]

性能实验需要同时包括随机散布、顺序/完整group、冻结trace下的pool合并，以及明确的共享消费者工作负载。对IC、IC-b、FS、FastLens采用同一全局CPU与内存预算，测有界cache和预取参数后选最强合法配置。不能只测 FS 不利的随机稀疏组占用，然后把结果推广为完整 FS 的最佳表现。

### 4.5 全局资源与真实I/O计量仍待验收

512MiB active decoded、1GiB reader arrays 是每reader预算；CNN默认16个DataLoader workers、prefetch_factor=2。因此不能把单reader限额解释为全管线内存上限。[S1,S2]

正式结果当前明确只统计 consumed microbatches；未消费lookahead不在计数内。建议worker保存lifetime counters，主进程在测量起止获取同步快照或结束回报；区分消费与丢弃预取。整机/进程组RSS、pinned memory和CUDA峰值独立测量；各worker局部max的max不能替代同时驻留总峰。[S2]

实际read内的SHA-256是额外CPU工作。需要统一各backend的完整性策略，或者在不可变数据快照上独立报告验证成本；不能回退R01的内容身份保证。当前没有测量证据表明SHA已经造成显著性能差异，应作为计量策略待定项处理。[S2,S3]

### 4.6 一个已复现的低优先级问题

`store.py:366–369` 对所有requested members预留decode空间，包括cache hits。cache=1,664B、output=1,664B、总预算3,584B的全缓存请求实际只需3,328B，却因剩余256B不够再次decode一个完整member而抛MemoryError。[S3]

建议只对miss做decode准入。另 `decoded_cache_peak_bytes` 在全hit单次read时为0，若字段表示实际驻留峰值，应从当前cache occupancy初始化。这两项是保守准入与指标细节，没有发现新的数据错误或无界驻留P1。

## 5 当前如何处理 Image Calculator 的最佳表示

**当前没有进行最佳表示选择。** `native.py` 实现了一组原生机制，实际matched build仍从预先指定的JPEG B8 profile出发。[S2]

| 能力 | 当前实现 |
| - | - |
| 表示配置 `NativeSpec(B,k,sampling,q)` | 已有，native.py13–30 |
| 6048个候选的枚举 | 已有，native.py33–35 |
| RGB→原生量化DCT→三平面mosaic | 已有参考codec，native.py45–80；色彩常数、padding、level shift和舍入等选择写入contract |
| NPZ存取、训练集归一化统计 | 已有基础函数，native.py83–130 |
| dataset/model/hardware/budget搜索任务 | 未实现 |
| 候选模型训练、准确率评估 | 未实现 |
| 候选storage/端到端latency profiling | 未实现 |
| bucket/transfer-learning准确率模型 | 未实现 |
| 预算筛选、最终训练、部署包 | 未实现 |

README第306–308行明确披露 `candidates()` 不训练且不实现self-design optimizer。该缺失是当前范围的限制，不能因已披露就把它视为完整IC复现。[S2]

当前 `resolve_profile` 固定JPEG B8、source512、三个分量all64源频率；`build`使用该profile的源依赖。spectral resize混合源频率，因此当前这些profile的FixedK也保留all64。[S2：materialize.py49–54；**main**.py177–183；README66–68]

- **FixedK** 处理既定变换中哪些源系数确实不需要。

- **safe dtype** 处理同一数值的无损整数宽度。

- **IC self-design** 改变表示的信息量、形状和模型计算量，并用真实准确率及资源预算决定选择。

因此前两项都不能替代self-design。

## 6 IC 原论文的最佳表示与搜索过程

### 6.1 最佳表示依赖任务和预算

IC的Build阶段以dataset、model、hardware建立accuracy/inference-time/storage模型；Exploration接收推理时间与容量预算；Deploy对选择的表示作最终训练并部署。作者后续教程也明确使用dataset、model、machine、budget四元组。[S6 §3；S7]

其目标可以形式化为：

`r* ≈ argmax estimated_accuracy(r), subject to latency_H(r) <= L_max and storage_D(r) <= S_max`。

这是一种对论文流程的表达，不是原文公开的伪代码。通常在部署前搜索并固定表示和相应模型；不需要每张图在线重新搜索。搜索是近似和预算受限的，报告应说“给定域和预算下找到的最佳可行配置”，避免无条件的全局最优声明。

### 6.2 搜索会改变模型真正看到的张量

原生表示包含四类primitive：采样、DCT分块、系数裁剪、标量量化。附件2024论文域：[S6 §4.1–4.2]

- sampling：无采样、行stride2、列stride2、行列均stride2。

- `B ∈ {8,16,32,64,128,256}`。

- `1 <= k <= B`；每块保留左上三角 `u+v<k`。

- 标量量化 `q ∈ {20,50,100}`；不是JPEG quality，也不是JPEG的Q矩阵。

候选数 `4 × (8+16+32+64+128+256) × 3 = 6048`。

量化系数组织为三张mosaic平面。若无padding且采样后大小为H'×W'，输出shape为 `3 × ((H'/B)k) × ((W'/B)k)`。例如256×256、B32、k3得到3×24×24；每块有6个保留系数及3个结构零。推理中解压后转float供模型使用，不把标量q乘回去、不做IDCT。搜索因此会同时改变存储、传输和模型计算量。[S6 §4.1,§4.4,§5.1]

原生训练在RGB上作RandomResizedCrop/Flip，再计算DCT。当前matched DCT24模型的输入通道语义和冻结checkpoint不能直接拿来评判所有native候选；新的表示需要自己的训练/微调与归一化统计。[S6 §5.1]

### 6.3 不需要从头训练6048个模型

原论文主要实验固定no sampling和q50，从504个B/k组合中采样96个配置。精确96列表没有披露；文中“20%”不能机械恢复出原列表。准确率模型将按压缩程度排序的候选分16个等大小bucket，采样6个关键bucket，以bucket内最高准确率插值未测bucket。[S6 §4.3.1,§5.1]

**96候选时，6个bucket对应36个候选模型，而不是6个模型。** 从压缩最少的表示开始，第一个模型从头训练，之后向更高压缩的表示顺次transfer learning。论文采用210epoch从头训练、20/40epoch迁移；时延和容量在目标机器实际profiling。若最终落在未测bucket，还要补训练该bucket里的候选进行选择，部署报告使用实测准确率。[S6 §4.3–4.5,§5.1]

bucket的预测最大准确率是候选组的上包络估计。不能把它赋给该bucket中任意一个最快的candidate并宣称该candidate满足准确率约束。候选级实测是最终验收条件。

需要公开工程选择的细节包括：96列表、排序和tie-break、通用关键bucket规则、跨候选checkpoint/optimizer/RNG迁移规则，以及未公开的色彩/舍入细节。作者2025教程写4104，而本次2024附件写6048；本次版本锚定2024论文，不拼接成一个假定一致的规格。[S6,S7]

## 7 建议补充的实验结构

### 7.1 表A 严格 matched 的存储与输入系统

保留现有路线：相同JPEG q/Q、相同请求与最终tensor、相同模型和checkpoint、相同训练recipe与硬件预算。该实验回答在同一工作负载下存储与输入系统的差异。

建议图例：

- `IC-style image-major NPZ (JPEG-DCT matched)`

- `IC-batched (JPEG-DCT matched)`

- `FS-style diagonal store (JPEG-DCT matched, B8)`

正文说明机制来源及适配。IC-b是显式批量变体；FS的B8是对JPEG-DCT的适配。严格matched要求最终tensor一致，通常已经排除了IC改变采样、B、k、q等有损自由度。该表可以有效支撑存储系统贡献，但不足以支撑“优于完整IC自设计部署方案”的结论。

### 7.2 表B 原生 IC 在准确率预算下的部署表现

另做一条IC-native search，先选一个共同模型家族，如ResNet50，跑通真实数据。不要首先把native搜索限制成B8来适应FastLens；B8锚点只能作为范围受限的补充。

建议以FastLens研究问题设定：

`min measured_latency_H(r)`，满足 `A_search(r) >= A_ref_search - delta` 和可选 `storage(r) <= S_max`。

例如预注册delta为0、0.5、1.0个百分点；这些是可选的新实验预算，不是IC原文参数。也可以保留原论文的latency/storage预算最大化accuracy，并给出accuracy–latency–storage Pareto。吞吐和单请求latency要选清楚，batch-size调优也需服从相同目标。

对齐任务、数据切分、评价目标和硬件资源；允许两系统的表示、实际输入尺寸及相应checkpoint不同，明确列出这些差异。此时端到端结果包含表示减少模型计算的收益。

### 7.3 可选表C 在 IC 选出的表示上交叉配对

先由IC搜索得到r\*，再使IC-NPZ和FastLens存储完全相同的原生量化系数，输出相同mosaic，运行同一个r\* checkpoint。该表回答“IC选择表示后，FastLens能否提供更好的存储与输入实现”。

需要新增native profile，保持原生量化语义，不能套用现有JPEG Q dequantize + DCT24组织路径。如果r\*的B不等于8，而FastLens只支持B8，须扩展格式/kernel或如实标为未支持。即使B8也要验证采样、三平面布局和mosaic语义；不能假设已有JPEG adapter直接可用。

表C价值高但开发成本更大；表A加表B应是当前优先的补实验组合。

## 8 可直接交给 agent 的开发任务

以下为建议新增任务，未在本轮修改用户实现。所有native搜索功能放在独立模块/命令中，保持已有matched契约含义。

### M1 固定正式 matched 入口

目标：专用launcher和强制pair contract检查。

产物：命令模板、common source/trace/profile/model/state/recipe/precision identities、准确的cache标签；不匹配时拒绝产出matched speedup。验证A0/B6不一致必定被拦截，同条件配对可通过。

### M2 完成生产 GPU gate 与计量

目标：真实共同源上 CPU oracle/FastLens/IC/IC-b/FS 的数值闭环；实际accumulation和tail；正式多worker路径的全局资源与全部I/O。

产物：冻结生产gate trace、校准/验证证据、worker lifetime counters、pipeline资源记录、原始日志和revision/binding hash。通过后再跑选定代表模型的完整验证集和training epoch。扩展其他模型、sharing和第二GPU的顺序由论文claim决定。

### N1 冻结 native 搜索任务

建议命令 `native-search prepare`。

输入：dataset/source/split、model family、RGB preprocessing、hardware、目标指标、latency/storage/accuracy budgets、search GPU budget、batch/worker/cache候选。

划分 `train_fit`、`search_val`、最终保留评估集。权重与归一化统计只用train_fit；候选选择与性能调优用search_val；最终表示和checkpoint冻结后才打开最终评估。ImageNet可从训练集划search_val，保留正式50K validation作为最终评估。这是为本次实验建议的严格隔离协议，非照抄IC原论文80/20划分。

产物：`search_contract.json`、`candidate_manifest.json`、`splits.json`及hash。

初期建议域 `ic_native_subset96_v1`：no sampling、q50，跨全部6个B选96个B/k组合。为精确可复现，可采用如下公开工程选择：先包含每个B的k=1和k=B共12点；将剩余492个(B,k)按字典序固定，用明确版本的NumPy PCG64和seed20261009无放回选择84点；最终保存全部96个candidate specs并以manifest而非RNG重新生成结果。该列表是本项目确定性样本，不宣称恢复原论文精确96点。后续可扩展到全部6048域。

### N2 原生候选训练与profiling

建议命令 `native-search profile` 和 `native-search fit`。

训练流程：RGB crop/flip→共同声明的色彩与采样→B-DCT→三角裁剪/标量量化→native mosaic→训练集统计归一化→对应模型。

要求：模型实际看到候选自己的mosaic尺寸；不统一resize回大图掩盖计算量收益；不把标量q乘回；不把现有DCT24 checkpoint直接用于新的native输入。若模型不支持某尺寸，依照预先冻结的stem/window/position规则适配或显式记录unsupported，避免偷偷排除候选。

现有NumPy FP64 codec用于正确性reference；真实native训练的GPU DCT实现需要另行接入并校准。在目标机器测实际编码容量、读/解压、H2D、模型推理及端到端目标指标，记录batch与cache条件。GPU DCT训练成本与部署离线编码成本分别记录。[S2,S6]

### N3 bucket 估计与预算选择

建议命令 `native-search select`。

公开排序指标及tie-break。例如用固定profile样本的实际bytes/图排序，candidate ID打破相同值；这属于本项目工程选择。96点分16bucket；关键6bucket索引必须在查看accuracy之前固定，并覆盖域两端。训练选中bucket内全部候选；首个从头训练，其余按固定顺序迁移。20/40epoch是可选budget，迁移权重与是否重置optimizer/BN/RNG均记录。

结果表严格区分 `measured_accuracy` 和 `predicted_bucket_accuracy`。估计可用于优先级和缩小搜索，不能直接满足candidate-level质量约束。对最终可能获选而未实测的bucket补测，并比较全部可行的已实测候选。没有可行候选时返回 `infeasible`，预算耗尽时保留 `budget_exhausted`。

### N4 最终训练、评估和部署包

建议命令 `native-search finalize`、`native-search evaluate`。

最终候选完成预注册的训练/微调。对准确率阈值附近的候选使用预定的重复seed或确认训练，避免只由短TL噪声决定结果。最终评估集不能继续参与选择；未满足预注册阈值时报告失效，不据最终标签重调k。

导出至少：

```text
selected_representation.json
normalization.json
checkpoint
storage_manifest.json
candidate_results.jsonl
search_summary.json
final_evaluation.json
```

每个candidate记录：spec和domain hash，source/split hashes，model/input shape/parameter count或FLOPs，checkpoint与stats身份，实际与预计accuracy，容量、E2E目标指标及分项，epochs、seed、迁移父候选、失败原因、硬件/runtime/batch/cache/worker预算、search/conversion成本。

最优措辞限制为“在声明的候选域与搜索预算下找到的最佳可行表示”。如果只跑subset96，就必须保留subset标签；没有证明全局最优。

### N5 可选 IC selected native paired replay

依赖N4，另加FastLens native表示支持和同r\*/checkpoint replay。不能因FastLens当前仅支持B8而回头删掉已经允许的IC候选。

### 总成本记录

对长期部署可报告 `T_total(N) = T_search + T_build + N × T_online`。IC的离线搜索、模型训练和编码成本单列，FastLens也记录转换、索引和调优成本。若比较稳态推理，明确搜索成本不计入单次latency；若比较time-to-deploy，所有相关成本进入同一口径。不同输入大小的一个epoch耗时不同，不能只把epoch数当GPU小时。

## 9 最终验收条件

本版可接受项：存储内容身份、有界decode/scatter、同microbatch合并、诊断报告identity、标签/trace基础契约、已披露范围的小型CPU/H100数值证据。[S1–S5]

正式matched论文结果仍需：M1、M2完成，实际共同源和生产配置通过数值与计量gate，充分调优IC/IC-b/FS并保留原始结果。完整IC竞争系统结论另需N1–N4。仅枚举6048候选或手选一个k并不满足native search验收。

## 10 来源与复核文件

源代码行号以patch重建后的new files为准；既有仓库文件使用integration.patch原始行号，并在正文说明应用后位置。

- **S1 提交说明和证据入口**：`submission/README.md`、`dependencies.json`、`host-unittest-result.txt`、`reproduce.py`、`run_cuda_gates.sh`；原始ZIP与revision见第1节。

- **S2 当前代码**：`submission/integration.patch`；`reconstructed/galp/benchmarks/dct_baselines/` 下的store、contracts、training、materialize、runner、unified、native、README和schema。

- **S3 独立存储探针**：`probes_storage/probe_storage.py`、`results.json`、`STORAGE_REVIEW.md`。只补齐父包缺失的标准SHA-256 helper，没有替换被审store/contracts逻辑。

- **S4 独立证据与schedule复核**：`probes_integrity/audit_evidence.py`、`results.json`、`schedule_results.json`；`probes_integration/trace_window_audit.json`。脚本明确保持上传JSON字节不变，只用已知前缀定位对应文件。

- **S5 提交的原始运行记录**：`submission/evidence-v2/` 和 `submission/evidence-h100/` 的calibration、verified JSON与日志、source/stores/contracts。它们是提交者运行记录，审查只在本环境核对其内容一致性。

- **S6 原始论文**：Sirin and Idreos, *The Image Calculator: 10x Faster Image-AI Inference by Replacing JPEG with Self-designing Storage Format*, PACMMOD/SIGMOD 2024, DOI [10.1145/3639307](https://doi.org/10.1145/3639307)。以用户提供的31页PDF为版本锚点，重点§3、§4.1–4.5、§5.1；具体实现无法从未公开源码核对。

- **S7 作者公开教程**：[Data Storage and Management for Image AI](https://www.utkusirin.com/sigmod25-image-ai-tutorial)，本轮重新打开。用于确认task/model/machine/budget与self-designed格式的定义；详细6048、96与训练设定均以S6为准。

审查材料包中的上述相对路径均位于 `review_round2_20261009/`。材料包不替代完整父仓库与大模型checkpoint；不包含伪造或估计的性能测量。

## 下载审查材料

[DCT_Baselines_Round2_Review_and_IC_Search.md](library-file\:fde1_bGliZmlsZV9odmNXNVIyMVhPXzdIZTJGWTlnZEtB_FileDrive_a7518b9df9748191912aceafac209a45)

[DCT_Baselines_Round2_Review_Pack.zip](library-file\:fde1_bGliZmlsZV80V0c0UFNJd1lTT2F2eTdhWGF3akx3_FileDrive_a7518b9df9748191912aceafac209a45)

材料包含完整审查、建议的 agent 任务、未改动的核心提交证据、独立复核脚本及结果。
