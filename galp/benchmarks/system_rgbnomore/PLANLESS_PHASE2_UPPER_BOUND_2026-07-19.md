# Planless Direct-DCT Phase 2：固定合同吞吐上限分析

日期：2026-07-19

## 1. 结论

在本轮固定合同（ImageNet-val 50K、batch 50、FP32、warmup 0、eager、预先固定的 domain checkpoints、cache 0）下，
`1.10 × DALI hot median` 门槛不可由当前固定模型执行达到。

最新完整轮 DALI hot median 为 `4839.499 img/s`，所以硬门槛为：

```text
4839.499 × 1.10 = 5323.449 img/s
50 / 5323.449 = 9.392407 ms/batch
```

隔离 GALP 的读取、解码、变换和异步预取后，只对一批已经生成的 FP32 DCT 输入重复执行固定 DCT 模型，
300 个 measured step、20 个 warmup step 的结果为：

```text
model-only = 5188.818 img/s
model-only = 9.636107 ms/batch
```

模型是每批必须执行的串行阶段，因此：

```text
T_end_to_end <= batch_size / t_model
T_end_to_end <= 5188.818 img/s
```

即使把存储读取、规划、workset build/upload、解码、变换、FP32 输入生成和提交全部假设为零成本，
仍有：

```text
5188.818 / 4839.499 = 1.072181 < 1.10
```

距离最新同轮门槛还差 `134.631 img/s`，即目标值的 `2.529%`；对应每批只差 `0.243699 ms`，
但这个差值已经出现在任何 GALP 工作发生之前的固定模型上。

这是“当前合同、当前 eager 模型实现、当前硬件状态”的操作上界，不是 RTX 4090 芯片的绝对物理上界。
改变模型算子实现、编译/执行后端或数值精度会改变这个实测界限，因此必须建立新合同并同时重测 GALP 与
DALI。单独替换形状和稠密算子完全相同的 FP32 checkpoint 通常不会实质改变计算量或吞吐上限，只会改变
accuracy/semantic 基线；只有 checkpoint 同时引入剪枝、结构化稀疏、量化或网络结构变化时，才会通过改变
执行工作量影响上限。第 13 节给出严格边界。

## 2. 证据范围

最新优化实现的完整结果目录：

```text
/tmp/galp-planless-phase2-upper-final-3932b5d
```

该轮 contract 在 tracked-clean commit `a5b932ae18791abf84fbc063fa441630bad23499` 创建，使用的 native
binary SHA-256 为 `0d4dd21af3f25020cfcf58af1b68f1f390562b24c7720d96c6d973f2f9f432fd`。运行期间 HEAD
变为 `22816846ed415294c70964d8c11536b179262559`，两 commit 之间只有本报告 Markdown 变化，benchmark
runtime files 和 native binary hashes 均未改变。因此最新性能、结构和语义结果反映同一实现，但严格 validator
仍正确保留 `git commit changed during the benchmark` provenance failure；第 10 节详述。

历史无争用基线和 model-only/storage/I/O 补充证据目录为：

```text
/tmp/galp-planless-phase2-gpu-d7b3e11
```

该基线 FastLanes benchmark source 为 tracked-clean，运行 commit 为
`f0102221ea860cf87a38c622b8055245afe65600`，native binary SHA-256 为
`4ed7e83999c031081d6fd421e4996e1d2d19eed96d04b8b538ce3102e0414917`。

最新目录关键文件：

```text
pipeline_galp.json
pipeline_galp_legacy.json
pipeline_rgbnomore.json
pipeline_dali.json
results.json
validation.json
commands.json
```

历史补充证据：

```text
storage_io_actual.json
forward_only.json
```

### 2.1 具体修改内容报告

#### 2.1.1 提交和代码范围

| Commit | 修改内容 | 规模 |
|---|---|---:|
| `d7b3e11` | Planless Direct-DCT 主实现、审计工具、validator 和测试矩阵 | 19 files，+4964/-1143 |
| `f010222` | 修正 `galp_legacy` 在 contract、runner、pipeline 间的名称/配置一致性 | 3 files，+6/-3 |
| `a3080b2` | 让实际 I/O 审计读取 pipeline 五轮 repeat schema 中的 native counter | 1 file，+22 |
| `3932b5d` | 64-thread planless kernel、FP32 原地范围映射和上限分析 | 4 files，+353/-9 |
| `a5b932a` | 加入首版运行审计、上限验收结论和证据哈希 | 1 file，+108 |
| `2281684` | 补充文件/函数级修改报告并澄清条件上界的合同边界 | 1 file，+199/-4 |

#### 2.1.2 执行架构的前后变化

修改前：

```text
每个 batch 的 image/crop requests
  -> CPU 为每个 source block 创建 FixedTransformItem
  -> 生成约 235,200 个 64-byte item
  -> remap selected vectors
  -> stable_sort + permutation + group offsets
  -> 再生成并上传 device batch items
  -> grouped transform kernel
```

修改后：

```text
每个 batch 的 image/crop requests
  -> CPU 生成每图一个 compact descriptor
  -> request-order flat rowgroup workset
  -> 一次 descriptor/binding upload
  -> GPU 每个 thread block 独占一个 output DCT block
  -> GPU 公式推导 source coordinates/physical row/output address
  -> dequantize + DCT transform + output
```

旧路径仅 `FixedTransformItem` vector 就是：

```text
235,200 × 64 B = 15,052,800 B/batch
```

新路径的核心 batch schedule 是：

```text
50 images × 172 B = 8,600 B/batch
```

即核心变换 schedule 从约 `15.05 MB` 降为 `8.6 KB`，约缩小 `1750×`；同时删除 source-list、全局排序、
permutation、group offsets 和第二份 device-batch-item 数组。Host planning 从
`O(transform_items log transform_items)` 变为 `O(batch_images + selected_rowgroups)`。

#### 2.1.3 Reader 和 compact representation

在 `galp/src/jpeg/jpeg_dct.cpp`、`galp/include/galp/jpeg_dct.hpp` 中完成：

- reader open 时把既有 image-major v2 metadata 编译为 `16 B/image` locator；
- 对验证过的 uniform shard ranges 使用公式计算 shard，不保存全量 shard index；不规则 ranges 才使用
  compact `uint16_t/image` fallback；
- 每 shard 使用 32-byte descriptor，相同 image layout 只保存一份 64-byte interned layout；
- quantization table 按值去重并常驻 reader；
- `plan_device_batch` 的 canonical path 每图只创建一个 `JpegDctDevicePlanlessImageDescriptor`；
- transformed production path 直接绕过 exact-batch plan cache，并停用 decoded-rowgroup cache；legacy 路径仅作为
  显式 A/B diagnostic 保留；
- selected image rowgroup 作为当前 FLS 的最小物理读取/解码原子，避免为了 crop 再创建无法减少物理读取量的
  per-vector source list。

canonical 50K reader-resident compact structures 的实测总量为 `800,544 B`：locator `800,000 B`、
derived shard index `0 B`、7 个 shard descriptors `224 B`、layout `64 B`、quant dictionary `256 B`。
磁盘格式、coefficient payload 和持久化文件均未改变。

#### 2.1.4 GPU mapping 与变换

在 `galp/src/jpeg/jpeg_dct_device.cu/.cuh` 中新增
`transformed_dct_grid_planless_kernel` 和 `project_planless_transformed_dct_grid_batch`：

- 一个 CUDA block 独占一个 output DCT block，不需要 atomic、全局 sort 或 group schedule；
- 从 output `(component, x, y)` 和 descriptor 中直接推导 bounded source stencil；
- 支持 raster、tiled-raster-32、Morton/Z-order 和 tiled-Z-32 的 physical-row 公式；
- 在同一 kernel 内完成 source mapping、coefficient load、dequantize、identity/down2 或 rational transform；
- identity/down2 保留 RGBNoMore 原有 FP32 operation graph；一般约分关系支持 `up/down <= 64`，每个轴只存
  `up + down - 1` 个共享 8x8 phase matrices，不按绝对 output position 展开矩阵；
- grayscale、4:4:4、4:2:0、variable shape、cross-shard 和 shuffled request order 使用同一 compact path；
- canonical batch 保持一个 logical workset、一个 decode launch 和一个 internal synchronization；
- 后续把 block size 从 256 改为 64 threads，每个 lane 循环装载最多四个 canonical down2 source
  coefficients；最新完整轮 fixed-transform p50 中位值从基线 `1.601024 ms` 降到 `0.878592 ms`。

Mapping 已融合进 fixed-transform kernel，因此单独的 `device_mapping_ms=0`，并由
`device_mapping_fused=true` 证明；mapping instructions 的时间包含在 fixed-transform 事件中，而不是被漏记。

#### 2.1.5 C++/Torch API、生命周期和内存计数

在 `galp/include/galp/direct_dct.hpp`、`galp/src/api/direct_dct.cpp`、
`galp/torch/direct_dct_torch.cpp` 和 `galp/src/cuda/memory/device_pool.cuh` 中完成：

- 复用已有 `y_tensor_async()`、`cbcr_tensor_async()` 和 `record_stream()` 生命周期接口，让 planless kernel 直接
  写入并返回 native output-owned Y/CbCr CUDA grid，不增加 host 中间 transform tensor；
- 为 `plan_batch/read_batch/prefetch_batch/read_batch_async` 增加 `enable_planless_execution` diagnostic A/B 开关，
  production 默认 `true`，`false` 只用于 legacy 对照；
- 新增 `RowgroupStorageBytes()` C++/Python diagnostic API，以真实 FLS record bytes 验证实际读取量；
- 扩展 Python plan preview 和 execution stats，暴露 compact representation、结构计数和各阶段 timing；
- native device pool 增加 in-use、peak、cached、allocation requests、实际 `cudaMalloc` 次数/字节计数，避免只看
  Torch allocator 而漏掉 GALP 自有 CUDA 内存。

新增的关键证明计数包括：

```text
host_expanded_transform_items_created
host_output_block_source_lists_created
host_global_transform_sort_items
planless_image_descriptor_count
planless_transform_output_block_count
planless_axis_program_count / phase_matrix_count / bytes
rowgroup_storage_bytes_read
device_mapping_fused / device_mapping_ms
exact_batch_plan_cache_enabled_batches
decoded_rowgroup_cache_enabled_batches
galp_native_device_* allocation counters
```

#### 2.1.6 Benchmark、审计和 validator

在 `galp/benchmarks/system_rgbnomore/` 中完成：

- 新增 `diagnostics/benchmark_planless_planning.py`：分别测 1K/50K、sequential/shuffled、5 repeats，并硬门控
  median、p95、顺序差异和规模差异；
- 新增 `diagnostics/audit_planless_storage_io.py`：统计 raw/compressed/index/persistent bytes、execution metadata、
  sequential/shuffled 实际 read amplification，并读取每个 repeat 的 native I/O counter；
- `run.py` 增加固定 50K 合同、tracked-clean runtime source hash、native binary fingerprint、四 pipeline 同轮运行
  和每 pipeline 前后 CPU/GPU/memory/block-I/O snapshot；
- `pipeline.py` 增加 `galp_legacy` controlled A/B、native per-batch stage distributions、主进程 VmRSS/VmHWM、完整
  50K compact prediction trace；
- `validate.py` 强制 planless/legacy 结构计数、cache-off、one-workset/decode/sync、planning/mapping gates、完整
  prediction agreement、sampled input/logit 数值一致性、CV/hot-min/hot-median gates；
- `direct_dct.py` 将 `(x + 1024) / 2040 * 2 - 1` 化简为 fresh FP32 tensor 上的
  `add_(4).mul_(1/1020)`，每批最低 tensor traffic 从 `136.377 MiB` 降到 `78.955 MiB`；
- `validate_pushdown.py` 同步检查 compact structural counters 和 native allocation/read counters。

#### 2.1.7 测试修改

`galp/tests/jpeg_dct_test.cpp` 新增三类核心测试：

- `CanonicalImageMajorFixedGridUsesCompactPlanlessDescriptors`：跨 shard canonical batch、零 expanded objects、
  cache bypass；
- `PlanlessRationalProgramsCoverSamplingShapesShardsAndSpatialOrders`：variable shapes、四种 spatial orders、
  grayscale/4:4:4/4:2:0、7/5 与 3/2 rational relations、shuffled requests；
- `PlanlessDeviceMatchesLegacyAcrossGeneralityMatrix`：目标 GPU 上逐元素比较 planless 与 legacy int16 Y/CbCr
  输出，并检查 one-workset/decode/sync。

`galp/tests/test_system_benchmark.py` 增加 contract/validator、repeat-schema I/O counter parser 和 FP32 原地范围映射
等回归测试。最终本地验证包括 23 个 Python tests、相关 CTest、CPU structural gtests 和目标 GPU 通用性矩阵。

硬件由 canonical pipeline 记录为 NVIDIA GeForce RTX 4090，compute capability 8.9，显存
`25,252,724,736` bytes。NVIDIA 公布的 RTX 4090 nominal shader FP32 峰值约为 `83 TFLOP/s`；
本报告只用它说明工作负载远未达到芯片算术峰值，不用 nominal peak 推导验收结论。

## 3. 稳定端到端结果

| 指标 | GALP planless | DALI |
|---|---:|---:|
| hot median throughput | 4478.479 img/s | 4839.499 img/s |
| hot min throughput | 4465.184 img/s | 4795.794 img/s |
| hot CV | 0.234% | 0.653% |
| 等价平均 batch 时间 | 11.164525 ms | 10.331704 ms |
| hot model-forward p50 中位值 | 10.402 ms | 约 9.527--9.600 ms |
| loader submit p50 中位值 | 0.492 ms | 约 0.391--0.441 ms |

GALP 当前达到 model-only ceiling 的：

```text
4478.479 / 5188.818 = 86.310%
```

对应剩余系统差距为：

```text
11.164525 - 9.636107 = 1.528418 ms/batch
```

这个差距不是阶段时间的简单求和，因为读取、CPU workset 构建和下一批 GPU 变换与当前模型前向重叠。
实测当前模型前向在重叠时从隔离值约 `9.64 ms` 上升到约 `10.40 ms`，说明下一批变换与当前模型
争用 GPU 是主要剩余损耗之一。

相对历史无争用基线，最新实现的 GALP hot median 提高 `6.549%`，平均 batch 时间降低 `6.146%`。

## 4. 存储与输入数据量

### 4.1 持久化数据

`storage_io_actual.json` 给出的完整 50K 数据集数据量为：

| 项目 | 完整数据集 | 等价每 batch（1000 batches） |
|---|---:|---:|
| raw DCT bytes | 39,321,600,000 B | 37.500 MiB |
| compressed coefficient rowgroups | 3,589,345,422 B | 3.423 MiB |
| total persistent bytes | 5,162,282,157 B | 4.923 MiB |

raw-to-total compression ratio 为 `7.6171×`。Planless candidate 相对 baseline 的持久化大小比例为 `1.0`，
sequential 和 shuffled 的 measured read amplification 也都为 `1.0`。

50K 图片的 compact reader 原生执行索引只有 `800,544` B，即 `16.011 B/image`；其中 canonical
uniform shard index 由公式推导，额外 shard-index bytes 为 0。持久化 execution metadata 为 0 B。

### 4.2 模型输入

每张图片的 DCT 模型输入元素数为：

```text
Y    = 1 × 28 × 28 × 8 × 8 = 50,176
CbCr = 2 × 14 × 14 × 8 × 8 = 25,088
合计 = 75,264 elements/image
```

batch 50 的张量数据量：

| 表示 | bytes/batch | MiB/batch |
|---|---:|---:|
| GALP native int16 grid | 7,526,400 | 7.178 |
| DCT model FP32 input | 15,052,800 | 14.355 |
| RGB model FP32 `50×3×224×224` input | 30,105,600 | 28.711 |

因此 Direct-DCT 把进入模型的 FP32 输入字节数减半。但这只影响输入阶段；patch embedding 之后，
两种模型都是 196 个、宽度 192 的 token，并进入相同深度的 Transformer。

### 4.3 FP32 adapter 流量

基线 adapter 先把 int16 grid 转成 FP32，再用四个 eager 逐元素算子执行：

```text
(x + 1024) / 2040 × 2 - 1
```

只按全局张量的必要读写计算，不含 allocator 和 kernel-launch 开销：

```text
int16 -> FP32 cast:                 22,579,200 B
4 × FP32 read + FP32 write:       120,422,400 B
合计:                             143,001,600 B = 136.377 MiB/batch
```

等价公式为 `(x + 4) / 1020`。使用新生成 FP32 张量上的两个 in-place 算子后，最低流量变为：

```text
22,579,200 + 2 × 30,105,600
= 82,790,400 B
= 78.955 MiB/batch
```

减少 `60,211,200 B/batch` 和每个 Y/CbCr pair 的四次额外 kernel launch，不改变模型、checkpoint、
精度或输入语义。

## 5. 计算量

使用 PyTorch `FlopCounterMode` 对固定模型做 batch-1 FP32 计数；FMA 按 2 FLOPs 计：

| 模型 | 参数量 | FP32 权重 | FLOPs/image | FLOPs/batch 50 |
|---|---:|---:|---:|---:|
| DCT ViT-Ti | 5,642,728 | 21.525 MiB | 2.467511 GFLOPs | 123.375565 GFLOPs |
| RGB ViT-Ti | 5,716,456 | 21.807 MiB | 2.493201 GFLOPs | 124.660070 GFLOPs |

DCT 模型只比 RGB 模型少：

```text
(2.493201 - 2.467511) / 2.493201 = 1.0304%
```

原因是两者主要计算都在相同的 12 层 Transformer 中。DCT 只在 patch embedding 节省计算：

```text
DCT: 2×2 Y block basis combine + Linear(384 -> 192)
RGB: Conv2d(3 -> 192, kernel=16, stride=16)
```

两者都产生 `14×14 = 196` 个 192-wide token；之后的 attention、MLP 和 classification head 基本相同。
所以输入字节减半不会自动转化成 10% 的模型吞吐差。

model-only 的实测有效算力为：

```text
123.375565 GFLOPs / 9.636107 ms = 12.803 TFLOP/s
```

原门槛对应：

```text
123.375565 GFLOPs / 9.392407 ms = 13.136 TFLOP/s
```

这两个数都显著低于 nominal shader FP32 peak，因为 ViT-Ti 包含大量小矩阵、attention、einsum、
逐元素算子和 kernel launch；nominal peak 不能作为这类 eager workload 的可达吞吐。

## 6. Direct-DCT 变换工作量

每个 batch 的实测结构计数为：

```text
planless image descriptors = 50
source DCT blocks          = 235,200
output DCT blocks          = 58,800
worksets                   = 1
decode launches            = 1
internal syncs             = 1
host expanded items        = 0
host global sort items     = 0
```

canonical down2 对每个输出 DCT block 组合 2×2 个源 block。其主要矩阵计算为：

```text
vertical:   8 × 16 outputs × 16 FMAs = 2,048 FMAs
horizontal: 8 ×  8 outputs × 16 FMAs = 1,024 FMAs
total:                                  3,072 FMAs/output block
```

全 batch：

```text
58,800 × 3,072 = 180,633,600 FMAs
                  361,267,200 FLOPs
```

这只有模型 `123.376 GFLOPs/batch` 的 `0.293%`。基线 kernel 实测约 `1.60 ms/batch`，不是算术峰值受限，
而是 58,800 个小 thread block 的调度、共享内存同步、地址推导和不规则列读取受限。

只计 decoded source、float accumulation 和 round-trip output，忽略量化表、binding 和 descriptor 读取，
设备工作集流量约为 `50.24--64.60 MiB/batch`（取决于 decoded coefficient 是 int8 还是 int16）。
因此减少空闲 warp 和 launch/synchronization 开销比增加算术吞吐更重要。

基线 kernel 每个输出 block 启动 256 threads，但 canonical down2 只有输入装载阶段需要 256 个 source
coefficient；后续最多使用 64 threads。新实现使用 64-thread block，每线程循环装载最多四个 source
coefficient，使通用 rational 路径与 canonical 路径都保持两 warp 的执行宽度。

最新完整轮由 native CUDA event 将 fixed-transform kernel 与 host 阶段分开记录。hot repeats 的
fixed-transform p50 中位值从历史基线 `1.601024 ms` 降到 `0.878592 ms`，降低 `45.12%`；这与 64-thread
block 减少空闲 warp 的预期一致。

## 7. 阶段时间与剩余余量

稳定轮典型 native 时间：

| 阶段 | 典型时间/batch |
|---|---:|
| planning | 0.03794 ms p50；0.08047 ms p95 |
| sync rowgroup read | 3.371 ms p50 |
| workset build | 0.876 ms p50 |
| workset upload | 1.144 ms p50 |
| decode | 0.107 ms p50 |
| mapping + fixed transform | 0.879 ms p50 |
| round | 0.073 ms p50 |

读取、CPU build/upload 和下一批 GPU 工作与当前模型重叠，因此不能把这些数直接相加得到端到端时间。
当前主要瓶颈已从 Phase 1 的 host planning 转移到 GPU 资源竞争和固定模型本身：

```text
legacy hot median             = 635.288 img/s
planless hot median           = 4478.479 img/s
planless / legacy             = 7.050×
planless model-only ceiling   = 5188.818 img/s
current / model-only ceiling  = 86.310%
```

planless 已经消除随 output blocks 增长的 host items、global sort 和 plan cache 依赖；剩余约 13.7% 的系统余量
来自不可完全隐藏的读取/变换、FP32 输入生成以及变换与模型在同一 GPU 上的竞争，而不是 planning 规模退化。

## 8. Gate 判定

本轮已经通过：

```text
GALP 与 RGBNoMore DCT 输入/输出语义
GALP 与 legacy exact input/logit 等价
完整 50K Top-1/Top-5 计数
planless/legacy 结构 A/B
persistent bytes 与 execution metadata
sequential/shuffled measured read amplification
planning median/p95、规模与顺序稳定性
GALP hot CV 和 DALI hot CV
```

最新 validator 的算法性能失败仍为原始两项相对吞吐门槛：

```text
GALP hot median / DALI hot median = 0.925401 < 1.10
GALP hot min                      = 4465.184 < DALI median 4839.499
```

此外存在一项非算法 provenance failure：运行期间仅报告 commit 发生变化；benchmark runtime files 和 native
binary 未变化，详见第 10 节。固定模型的隔离上界只有最新 DALI 的 `1.072181×`，因此 `1.10×` 门槛在本轮
合同下仍是过约束。
合理的上限验收应改为同时报告：

```text
1. correctness/storage/I/O/planning/structure gates 全部通过；
2. model-only ceiling；
3. end-to-end / model-only ceiling efficiency；
4. 相同硬件状态下的 DALI ratio；
5. 未通过原 1.10× gate 的数学原因，而不是把 gate 静默放宽。
```

## 9. 复现 model-only ceiling

```bash
PYTHONPATH=build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
galp/benchmarks/system_rgbnomore/diagnostics/direct_dct.py \
/tmp/galp-imagenet-val512-20260717-122634/manifest.bin \
  --phase forward \
  --preprocess rgbnomore-val-pushdown \
  --batch-size 50 \
  --steps 300 \
  --warmup 20 \
  --cache-capacity-mib 0 \
  --plan-cache-capacity 0 \
  --decode-batch-rowgroups 64 \
  --rowgroup-prefetch-depth 16 \
  --rowgroup-prefetch-workers 4 \
  --rowgroup-prefetch-min-decode-batches 1 \
  --output-json /tmp/galp-planless-phase2-gpu-d7b3e11/forward_only.json
```

## 10. 最新优化完整轮与 provenance 审计

目录 `/tmp/galp-planless-phase2-upper-final-3932b5d` 已在 16:45--16:57 被最新完整轮覆盖。该轮产生所有四条
pipeline、5 repeats、50K prediction trace 和 validator 文件。两个严格语义比较均通过：

```text
GALP vs RGBNoMore: 50,000/50,000 prediction trace，Top-1 agreement = 1.0
GALP vs legacy:    inputs/logits max_abs = 0，Top-1 agreement = 1.0
```

最新性能结果：

| 指标 | 历史无争用基线 | 最新优化完整轮 | 变化 |
|---|---:|---:|---:|
| GALP hot median | 4203.211 img/s | 4478.479 img/s | +6.549% |
| GALP mean batch time | 11.895668 ms | 11.164525 ms | -6.146% |
| fixed-transform p50 中位值 | 1.601024 ms | 0.878592 ms | -45.123% |
| planless/legacy hot median | 6.405× | 7.050× | +10.06% |
| GALP hot CV | 0.112% | 0.234% | 两者均通过 5% gate |

最新轮的 GALP 与 DALI host load 都较高，但比被覆盖的旧污染轮更对称：GALP 前后 loadavg 为
`27.76 -> 34.81`、全机 CPU utilization `30.11%`；DALI 为 `35.08 -> 33.53`、CPU utilization `35.55%`。
GALP 运行期间 GPU1 保持空闲；DALI 结束时 GPU2 出现外部负载，因此系统状态仍不是理想独占环境。尽管如此，
GALP/DALI hot CV 分别只有 `0.234%/0.653%`，四个 hot repeats 内部稳定。

validator 的第三项失败不是代码或 binary 变化，而是基准运行期间发生了一次报告提交：

```text
contract/start commit = a5b932ae18791abf84fbc063fa441630bad23499
end commit            = 22816846ed415294c70964d8c11536b179262559
changed tracked file  = PLANLESS_PHASE2_UPPER_BOUND_2026-07-19.md only
native binary SHA-256 = 0d4dd21af3f25020cfcf58af1b68f1f390562b24c7720d96c6d973f2f9f432fd
```

contract 中列出的所有 benchmark runtime file hashes 与执行时一致，native `.so` 也与当前文件一致；否则
validator 会同时报告 runtime source 或 binary mismatch。严格复现规则仍把 commit 变化判为失败，因此本报告不把
`validation.ok` 改写为 true。该轮可作为同一实现的最新性能/语义观察值，但若需要一份 validator 完全 clean 的
正式归档，仍需在报告提交停止后原样重跑。

## 11. 最终上限验收判定

原始 validator 和门槛均未修改。严格按原始 Definition of Done，性能项仍是：

```text
FAIL: 4478.479 / 4839.499 = 0.925401 < 1.10
FAIL: GALP hot min 4465.184 < DALI hot median 4839.499
```

最新 artifact 另有一项报告 commit 在运行中变化的 provenance failure；它不改变上述算法性能结论，但意味着
该 artifact 的 `validation.ok=false` 不能只归因于两项吞吐 gate。

按用户授权的“达到固定合同上限即可，但必须给出数据量和计算量理论分析”收口，本阶段接受结论为：

```text
ACCEPT AT OPERATIONAL UPPER BOUND
```

理由不是端到端实现已经等于 model-only ceiling，而是原 `1.10×` 要求本身高于固定模型的实测操作上界：

```text
required                         = 5323.449 img/s = 9.392407 ms/batch
fixed model-only                 = 5188.818 img/s = 9.636107 ms/batch
fixed model-only / DALI          = 1.072181
required - fixed model-only      = 134.631 img/s = 2.529% of target
```

任何合法 Direct-DCT 端到端路径还必须执行正成本的读取、workset、解码、变换和输入生成，因此不能超过该
model-only bound。优化模型 operator graph/执行后端、降低 precision，或改变 batch/model architecture 可以建立
一个不同的新上限，但这些会改变固定合同；同形状稠密 checkpoint 的单独替换一般不会改变吞吐上限。无论采用
哪种新合同，都必须让 GALP 与 DALI 使用等价设置并重新测量，不能只优化分子。

除被数学上界否定的两项相对吞吐 gate 外，correctness、通用性、结构、存储、metadata、实际 I/O amplification、
planning、mapping/fixed-transform 和 CV gate 均已有文件化证据通过。原失败保持可见，没有被静默放宽。

## 12. 证据哈希归档

历史无争用补充证据：

| 文件 | SHA-256 |
|---|---|
| `contract.json` | `93688a5f21a06b6889b9d61ce7adf1b8e7b08f3d8f5f6893178fff4cf03cf204` |
| `results.json` | `4ca5d47da0c2ed1793013754eaaedf11ef321b227beb83c1d334425a42034a1f` |
| `validation.json` | `05aafcd42fddc8637d3afe8eb1d3173287e36b9c6751e2fc7a5cb7e2449b7850` |
| `storage_io_actual.json` | `7feee9a55b5f163da0222433098e7fa3c647fb0f5a828646513d8996483e3ecf` |
| `forward_only.json` | `95614686382da81fa165bd60780010d47dd6b0a3c356ef3b1f530d5b5fdd199b` |
| CPU `planning.json` | `21bf797377787ff69fb9a4aea5d39bb6dea3b901c75f866c0ced362c30c3a58e` |

最新优化完整轮证据：

| 文件 | SHA-256 |
|---|---|
| `contract.json` | `44928ecb039ff7bf666558725db0812d6184e9707e44fe62471a1531267eb78c` |
| `results.json` | `9f15f61d0f252f4cdb1463ad1f532d670c3ef0ba495768f8455ccfaed2f48a4e` |
| `validation.json` | `b1a65d59ea0a2ee95c6a23cb47f6193b912e129da91d06d0407355687f7a36ce` |
| `commands.json` | `e54c2206c975bb0b3a90f277794c0826710aa26da070bd1f3e8211dc3aee3a64` |
| `pipeline_galp.json` | `71c606adb904b79d4107753a9534b527525e68d5a4c82a8dee1ce044733ae00e` |
| `pipeline_galp_legacy.json` | `419f22e5c82db6d7af783acb45e5ced67e19879286fcd47e1342bfbbac135fdc` |
| `pipeline_rgbnomore.json` | `f1a90199a76f03887874a7bfb71dca130d7136bee113a855c9874189acc9a893` |
| `pipeline_dali.json` | `3d3052b7337cd1a6b99acfc5a6b89695f3cfb3bcb40777e656a5af99c9e1355e` |

两个 FastLanes contract 在创建时都记录 `benchmark_source_clean=true`、`git_tracked_dirty=false`。最新轮结束前
发生的 tracked change 仅为本报告提交；仓库中用户原有的 `.cache/` 和
`galp/examples/image_order_benchmark/res` 两个未跟踪目录未进入 benchmark runtime source，也没有被本阶段修改
或删除。

## 13. “改变模型实现、编译模式、精度或 checkpoint”的确切含义

`5188.818 img/s` 是条件上界：它测量的是当前 DCT ViT-Ti checkpoint、FP32、PyTorch eager operator graph、
batch 50 和当前 CUDA/PyTorch 栈。下面这些改变会有不同影响：

| 改变 | 例子 | 为什么可能改变上限 | 是否仍是当前合同 |
|---|---|---|---|
| 模型算子实现 | 融合 QKV/MLP、fused LayerNorm、FlashAttention、手写 Triton/CUDA kernel | 减少 kernel launch、global-memory round trip 或提高 GEMM 利用率 | 否，需建立新实现合同 |
| 编译/执行模式 | `torch.compile`/Inductor、TensorRT、CUDA Graph | 融合 eager operators、消除 Python/dispatcher 开销、固定 graph launch | 否，必须两侧同等启用并重测 |
| 数值精度 | FP32 改为 TF32、FP16、BF16、INT8 | 减少数据字节并使用 Tensor Core，但数值误差和可用 kernel 改变 | 否，需重新定义 correctness tolerance |
| 模型结构 | depth/width/token 数、attention/MLP 结构、patch embedding | 直接改变参数量、FLOPs 和 activation traffic | 否，已经是另一个模型 |
| checkpoint，仅权重值变化 | 同一稠密 ViT-Ti 的另一组 FP32 weights | shape、operator graph、FLOPs 和内存量不变，通常吞吐几乎不变 | 性能近似不变，但 accuracy/语义合同失效 |
| checkpoint 携带执行结构变化 | pruning、2:4 sparsity、量化权重、蒸馏后小模型 | 只有执行后端真正利用稀疏/量化/小结构时，计算和流量才减少 | 否，本质上同时改了结构或精度 |

因此原句不是说“随便换一个 checkpoint 就能通过”。更准确的判定是：

```text
只换同结构 dense FP32 weights：t_model 基本不变，5188.818 img/s 上界基本不变；
换执行图/后端/precision/architecture：t_model 必须重新测量，旧上界作废；
只给 GALP 启用优化而 DALI 不启用：不是公平的 Direct-DCT vs DALI 证据。
```

当前比较本来就使用 input-domain-specific checkpoints：GALP/RGBNoMore 使用 DCT checkpoint，DALI 使用 RGB
checkpoint，两者不是同一组权重。所谓公平重测不是强行使用同一 checkpoint bytes，而是在实验前固定同一模型
规模/recipe family、precision 和执行后端，以及各自预先声明的 domain checkpoint，不能看完结果后只替换其中
一侧。对于相同 shape 的 dense checkpoint，性能边界基本不变，变化的主要是 accuracy 和语义基线。

要让原 `1.10×` gate 从数学上“可能”，新模型前向首先必须满足：

```text
t_model < 50 / (1.10 × 4839.499) = 9.392407 ms/batch
```

这只是必要条件，不是充分条件，因为端到端还存在正成本的非重叠读取、变换和提交。当前最新端到端为
`11.164525 ms/batch`，距离目标 `9.392407 ms/batch` 需要再减少 `1.772118 ms/batch`（`15.87%`）。如果新的
模型/编译方案同时改变 DALI 吞吐，右侧目标也必须使用新的同轮 DALI median 重新计算，不能继续使用
`4839.499 img/s` 这个分母。

## 14. 高优先级模型与受限 Direct-DCT 重叠实现

### 14.1 为什么继续处理 0.766317 ms

第 1 节的 `5188.818 img/s` 证明原 `1.10× DALI` gate 在固定模型下不可达，但当前
end-to-end 仍未到这个模型上界。最新完整轮的 GALP model-forward p50 中位值为
`10.402424 ms`，forward-only 为 `9.636107 ms`：

```text
T_model_with_transform - T_model_only
= 10.402424 - 9.636107
= 0.766317 ms/batch
```

同期 planless transform p50 中位值为 `0.878592 ms`，模型额外延迟相当于 transform
时间的 87.22%。这表明 next-batch transform 与当前模型发生了明显 GPU 资源争用；即使
不改变固定模型上界，也仍应消除这部分实现损失，使 end-to-end 接近 DALI 并尽量接近
`5188.818 img/s`。

### 14.2 具体代码修改

本轮未改变模型、checkpoint、FP32 精度、输入形状或 Direct-DCT 数值语义，修改内容如下：

| 文件 | 具体修改 |
|---|---|
| `galp/src/cuda/memory/cuda_raii.cuh` | `CudaStream` 增加 `cudaStreamCreateWithPriority` RAII 接口 |
| `galp/src/engine/workset/model.cuh`、`streams.cu` | persistent H2D/compute/D2H workset stream 支持设备定义的 priority |
| `galp/include/galp/jpeg_dct.hpp` | 新增三种 scheduling policy、transform chunk 配置与 stream/event/launch 统计 |
| `galp/src/jpeg/jpeg_dct.cpp`、`jpeg_dct_device.cuh` | scheduling options 进入 plan 和 plan-cache key，并传到 device executor |
| `galp/src/jpeg/jpeg_dct_device.cu` | 独立低优先级 transform stream、decode→transform event handoff、offset-aware chunked planless kernel、低优先级 round/cache stream |
| `galp/torch/direct_dct_torch.cpp` | Python API 暴露 scheduling policy、chunk blocks、低优先级开关和新增 counters |
| `pipeline.py` | 模型、adapter tensor ops 和计时 event 使用显式 priority `-1` stream；serial 延迟下一批 prefetch |
| `run.py` | 正式 contract 记录 model/Direct-DCT priority、策略和 chunk size，默认 limited-overlap/64 blocks |
| `scheduler_matrix.py` | 同 contract 自动运行 fully-overlapped、limited-overlap、serial 并计算核心 delta |
| `jpeg_dct_test.cpp` | 7-block chunk 的尾块/多 launch 输出必须与 single-grid 和 legacy bit-exact |
| `test_system_benchmark.py` | 证明 serial 在下一次 `load()` 前不会提交 next-batch prefetch |

Native event graph 由原来的：

```text
H2D event -> decode compute stream -> transform（同一 stream）
host synchronize -> round stream -> batch completion
```

变为：

```text
低优先级 H2D stream
  -> H2D-ready event
低优先级 decode stream
  -> decode-done event
低优先级 transform stream（可分块）
  -> 低优先级 round stream
  -> batch completion event
高优先级 model stream 只等待当前 batch completion
```

同一 rowgroup 的 transform 依赖 decode 输出，因此该依赖不能删除；“解耦”的含义是
把 copy、decode、transform 从一个隐式 compute-stream 尾链拆成独立 stream 和可审计 event
handoff，使不同 batch 的 Direct-DCT 工作可以被高优先级模型调度打断，而不是取消真实数据依赖。

### 14.3 三种策略的严格定义

| 策略 | next-batch overlap | transform grid | 用途 |
|---|---|---|---|
| `fully-overlapped` | 是 | 每 workset 一个完整 grid | 低优先级 stream 本身的隔离效果 |
| `limited-overlap` | 是 | 默认最多 64 blocks/launch | 限制 active SM 并增加模型插入边界 |
| `serial` | 否 | 完整 grid | 同 contract 的 `T_model_only` 参考和吞吐下界 |

serial 不是另一个模型-only microbenchmark：它仍执行同样的 Direct-DCT、adapter 和模型，只是
下一批预取必须等上一批 model stream 完成。这使三种策略的样本、输入、模型、checkpoint、精度
和统计边界保持相同，可直接用 serial 的 model-forward 时间计算
`T_model_with_transform - T_model_only`。

### 14.4 数据量、计算量和 chunk 上限

batch 50 的输出和执行规模为：

```text
每图 output blocks = 28×28 + 2×14×14 = 1,176
每 batch output blocks = 50×1,176 = 58,800
每 block CUDA threads = 64
每 batch transform threads = 3,763,200
source blocks = 235,200 = 4×output blocks
```

canonical factor-2 路径每个 output block 的主要矩阵工作约为 3,072 FMA，因此一个
batch 约为 180.63M FMA（约 361.27M FLOPs），另有 235,200 source blocks 的系数
读取、反量化和 clamp。transform 实测只需约 0.879 ms，而模型为 9.636 ms，因此从
工作量看 transform 可以被模型窗口隐藏；限制来自共享 SM 和 memory subsystem，不是必须串行
相加的理论依赖。

RTX 4090 有 128 SM。64-block grid 在任一 launch 内最多把 blocks 分配给 64 个 SM，
active-SM 几何上界约为 50%，但每 batch 需要：

```text
ceil(58,800 / 64) = 919 launches
```

若每次 launch/调度固定成本约 3–5 µs，919 次的固定开销约为 2.76–4.60 ms，连同
原 kernel 工作约为 3.64–5.48 ms。最新 planning、read、workset build/upload、decode、
transform、round 热态合计约 6.49 ms，原本低于 9.636 ms 模型窗口；64-block 策略在
低 launch-overhead 端仍可能完全隐藏，在高端则可能损失 next-batch readiness。因此 64 是
“优先保护模型”的实测起点，不是预先宣称的最优值；若吞吐下降，应继续比较 128/256，取得
模型额外延迟与 launch 开销的 Pareto 点。

### 14.5 验证状态和复现命令

已通过：

- 全量 `cmake --build build -j2`；
- `_galp_direct_dct` 和 `galp_tests` 增量构建；
- Direct-DCT Torch import CTest；
- `galp.tests.test_system_benchmark` 20/20；
- Python `py_compile`；
- `git diff --check`。

当前受控执行环境可通过 `nvidia-smi` 枚举 RTX 4090，但测试进程中的
`cudaGetDeviceCount` 返回无可用设备，所以 GPU correctness test 被明确 skip，未伪造三策略
结果。请在有 CUDA runtime 权限的目标终端运行：

```bash
cd /home/tangyuxin/gfastlanes/FastLanes

GALP_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0 \
  ./build/galp/tests/galp_tests \
  --gtest_filter=JpegDct.PlanlessDeviceMatchesLegacyAcrossGeneralityMatrix

CUDA_VISIBLE_DEVICES=0 \
  /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_rgbnomore/scheduler_matrix.py \
  --contract /tmp/galp-planless-phase2-upper-final-3932b5d/contract.json \
  --output-dir /tmp/galp-planless-scheduler-matrix \
  --binding-dir build/galp/torch \
  --transform-blocks 64
```

矩阵汇总写入 `/tmp/galp-planless-scheduler-matrix/scheduler_matrix.json`。最终验收必须同时
检查 bit-exact correctness、三策略 Top-1/Top-5、sample trace、event/launch counters、
`model_extra_p50_ms_vs_serial` 和总吞吐，不能只挑一个最好 latency 数字。
