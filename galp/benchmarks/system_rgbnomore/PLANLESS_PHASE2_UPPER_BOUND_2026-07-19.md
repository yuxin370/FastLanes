# Planless Direct-DCT Phase 2：固定合同吞吐上限分析

日期：2026-07-19

## 1. 结论

在本轮固定合同（ImageNet-val 50K、batch 50、FP32、warmup 0、eager、同一 checkpoint、cache 0）下，
`1.10 × DALI hot median` 门槛不可由当前固定模型执行达到。

同轮 DALI hot median 为 `4789.330 img/s`，所以硬门槛为：

```text
4789.330 × 1.10 = 5268.263 img/s
50 / 5268.263 = 9.490794 ms/batch
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
5188.818 / 4789.330 = 1.083412 < 1.10
```

距离原门槛还差 `79.446 img/s`，即目标值的 `1.508%`；对应每批只差 `0.145313 ms`，
但这个差值已经出现在任何 GALP 工作发生之前的固定模型上。

这是“当前合同、当前 eager 模型实现、当前硬件状态”的操作上界，不是 RTX 4090 芯片的绝对物理上界。
若允许改变模型实现、编译模式、精度或 checkpoint，界限会变化；这些不属于本阶段允许用于证明 Direct-DCT
执行架构收益的手段。

## 2. 证据范围

主证据目录：

```text
/tmp/galp-planless-phase2-gpu-d7b3e11
```

该目录是无共享主机争用的固定合同完整轮，FastLanes benchmark source 为 tracked-clean，运行 commit 为
`f0102221ea860cf87a38c622b8055245afe65600`，native binary SHA-256 为
`4ed7e83999c031081d6fd421e4996e1d2d19eed96d04b8b538ce3102e0414917`。

优化后实现的完整复跑目录为：

```text
/tmp/galp-planless-phase2-upper-final-3932b5d
```

其 FastLanes benchmark source 同样为 tracked-clean，运行 commit 为
`3932b5da9f22a8a899cc3bf9449cc25b3a3a5025`，native binary SHA-256 为
`0d4dd21af3f25020cfcf58af1b68f1f390562b24c7720d96c6d973f2f9f432fd`。该复跑的语义、结构和阶段 gate
通过，但 GALP 阶段受到可观测的共享主机/GPU1 工作负载干扰，因此不用于估计端到端架构上限；第 10 节给出判据。

关键文件：

```text
pipeline_galp.json
pipeline_galp_legacy.json
pipeline_rgbnomore.json
pipeline_dali.json
results.json
validation.json
storage_io_actual.json
forward_only.json
```

硬件由 canonical pipeline 记录为 NVIDIA GeForce RTX 4090，compute capability 8.9，显存
`25,252,724,736` bytes。NVIDIA 公布的 RTX 4090 nominal shader FP32 峰值约为 `83 TFLOP/s`；
本报告只用它说明工作负载远未达到芯片算术峰值，不用 nominal peak 推导验收结论。

## 3. 稳定端到端结果

| 指标 | GALP planless | DALI |
|---|---:|---:|
| hot median throughput | 4203.211 img/s | 4789.330 img/s |
| hot min throughput | 4194.590 img/s | 4518.097 img/s |
| hot CV | 0.112% | 2.646% |
| 等价平均 batch 时间 | 11.895668 ms | 10.439873 ms |
| hot model-forward 典型值 | 约 11.09 ms | 约 9.69--9.76 ms |
| loader submit 典型值 | 约 0.52--0.55 ms | 约 0.41--0.44 ms |

GALP 当前达到 model-only ceiling 的：

```text
4203.211 / 5188.818 = 81.005%
```

对应剩余系统差距为：

```text
11.895668 - 9.636107 = 2.259561 ms/batch
```

这个差距不是阶段时间的简单求和，因为读取、CPU workset 构建和下一批 GPU 变换与当前模型前向重叠。
实测当前模型前向在重叠时从隔离值约 `9.64 ms` 上升到约 `11.09 ms`，说明下一批变换与当前模型
争用 GPU 是主要剩余损耗之一。

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
123.375565 GFLOPs / 9.490794 ms = 12.999 TFLOP/s
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

这只有模型 `123.376 GFLOPs/batch` 的 `0.293%`。但该 kernel 实测约 `1.59 ms/batch`，不是算术峰值受限，
而是 58,800 个小 thread block 的调度、共享内存同步、地址推导和不规则列读取受限。

只计 decoded source、float accumulation 和 round-trip output，忽略量化表、binding 和 descriptor 读取，
设备工作集流量约为 `50.24--64.60 MiB/batch`（取决于 decoded coefficient 是 int8 还是 int16）。
因此减少空闲 warp 和 launch/synchronization 开销比增加算术吞吐更重要。

基线 kernel 每个输出 block 启动 256 threads，但 canonical down2 只有输入装载阶段需要 256 个 source
coefficient；后续最多使用 64 threads。新实现使用 64-thread block，每线程循环装载最多四个 source
coefficient，使通用 rational 路径与 canonical 路径都保持两 warp 的执行宽度。

优化后完整复跑虽然受到外部争用，native CUDA event 仍把 fixed-transform kernel 与 host 阶段分开记录。
hot repeats 的 fixed-transform p50 中位值从 `1.601024 ms` 降到 `1.183744 ms`，降低 `26.06%`；这与
64-thread block 减少空闲 warp 的预期一致。该局部事件时间不用于替换无争用轮的端到端吞吐。

## 7. 阶段时间与剩余余量

稳定轮典型 native 时间：

| 阶段 | 典型时间/batch |
|---|---:|
| planning | 0.036 ms |
| sync rowgroup read | 3.23 ms p50 |
| workset build | 0.86 ms p50 |
| workset upload | 0.94--1.08 ms p50 |
| decode | 0.11 ms p50 |
| fixed transform | 1.60 ms p50 |
| round | 0.07 ms p50 |

读取、CPU build/upload 和下一批 GPU 工作与当前模型重叠，因此不能把这些数直接相加得到端到端时间。
当前主要瓶颈已从 Phase 1 的 host planning 转移到 GPU 资源竞争和固定模型本身：

```text
legacy hot median             = 656.251 img/s
planless hot median           = 4203.211 img/s
planless / legacy             = 6.405×
planless model-only ceiling   = 5188.818 img/s
current / model-only ceiling  = 81.005%
```

planless 已经消除随 output blocks 增长的 host items、global sort 和 plan cache 依赖；剩余约 19% 的系统余量
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

唯一失败为原始相对吞吐门槛：

```text
GALP hot median / DALI hot median = 0.877620 < 1.10
GALP hot min                      = 4194.590 < DALI median 4789.330
```

由于固定模型的隔离上界本身只有 `1.083412 × DALI`，`1.10×` 门槛在本轮合同下是过约束。
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

## 10. 优化后完整复跑的污染审计

优化后完整复跑产生了全部合同文件，validator 仍只有原始两项吞吐 gate 失败，两个严格语义比较均通过：

```text
GALP vs RGBNoMore: 50,000/50,000 prediction trace，Top-1 agreement = 1.0
GALP vs legacy:    inputs/logits max_abs = 0，Top-1 agreement = 1.0
```

但该轮不能作为吞吐上限样本。和无争用完整轮对比：

| 指标 | 无争用完整轮 | 优化后复跑 |
|---|---:|---:|
| GALP hot median | 4203.211 img/s | 2289.910 img/s |
| GALP hot process CPU time 中位值 | 19.734 s/repeat | 41.007 s/repeat |
| host loadavg，GALP 开始 | 4.05 | 29.17 |
| host loadavg，GALP 结束 | 5.58 | 30.19 |
| 全机 CPU utilization ratio | 7.06% | 32.90% |
| fixed-transform p50 中位值 | 1.601 ms | 1.184 ms |
| model-stream event p50 中位值 | 11.092 ms | 19.453 ms |

更直接的外部干扰证据来自 `commands.json`：优化后 GALP 开始时 GPU1（H100）为空闲状态，结束时 GPU1
已占用 `19,989 MiB`、功耗 `201.23 W`；随后执行 DALI 时 GPU1 又恢复为空闲状态。也就是说 GALP 和 DALI
没有受到相同的共享 CPU/内存/驱动负载。

同一轮中，GALP 的 sync rowgroup read、workset build 和 workset upload 典型值分别从无争用轮约
`3.23/0.86/0.94--1.08 ms` 上升到约 `8.28/2.91/4.84 ms`。这些 host 路径没有被 64-thread CUDA kernel
修改；与此同时独立 CUDA event 测得 kernel 本身反而快了 26%。模型 stream 会等待预处理完成，所以等待时间
被包含在 model-stream event 中，解释了其从约 11.09 ms 增至 19.45 ms。

因此 `2289.910 img/s` 是共享资源争用下的受污染下界，不是新实现回归后的架构上限，也不能用来修改第 1 节
的结论。上限判定继续使用无争用固定合同完整轮和隔离 model-only 测量。

## 11. 最终上限验收判定

原始 validator 和门槛均未修改。严格按原始 Definition of Done，性能项仍是：

```text
FAIL: 4203.211 / 4789.330 = 0.877620 < 1.10
FAIL: GALP hot min 4194.590 < DALI hot median 4789.330
```

按用户授权的“达到固定合同上限即可，但必须给出数据量和计算量理论分析”收口，本阶段接受结论为：

```text
ACCEPT AT OPERATIONAL UPPER BOUND
```

理由不是端到端实现已经等于 model-only ceiling，而是原 `1.10×` 要求本身高于固定模型的实测操作上界：

```text
required                         = 5268.263 img/s = 9.490794 ms/batch
fixed model-only                 = 5188.818 img/s = 9.636107 ms/batch
fixed model-only / DALI          = 1.083412
required - fixed model-only      = 79.446 img/s = 1.508% of target
```

任何合法 Direct-DCT 端到端路径还必须执行正成本的读取、workset、解码、变换和输入生成，因此不能超过该
model-only bound。改变编译模式、模型、checkpoint、precision、batch 或 cache 才可能改变界限，但这些会破坏
固定合同，不能用作本阶段的性能证明。

除被数学上界否定的两项相对吞吐 gate 外，correctness、通用性、结构、存储、metadata、实际 I/O amplification、
planning、mapping/fixed-transform 和 CV gate 均已有文件化证据通过。原失败保持可见，没有被静默放宽。

## 12. 证据哈希归档

无争用主证据：

| 文件 | SHA-256 |
|---|---|
| `contract.json` | `93688a5f21a06b6889b9d61ce7adf1b8e7b08f3d8f5f6893178fff4cf03cf204` |
| `results.json` | `4ca5d47da0c2ed1793013754eaaedf11ef321b227beb83c1d334425a42034a1f` |
| `validation.json` | `05aafcd42fddc8637d3afe8eb1d3173287e36b9c6751e2fc7a5cb7e2449b7850` |
| `storage_io_actual.json` | `7feee9a55b5f163da0222433098e7fa3c647fb0f5a828646513d8996483e3ecf` |
| `forward_only.json` | `95614686382da81fa165bd60780010d47dd6b0a3c356ef3b1f530d5b5fdd199b` |
| CPU `planning.json` | `21bf797377787ff69fb9a4aea5d39bb6dea3b901c75f866c0ced362c30c3a58e` |

优化后污染审计证据：

| 文件 | SHA-256 |
|---|---|
| `contract.json` | `47d65e43d86b74509b522fdd76eab5734e54a6ec207344018a030b88df8ac719` |
| `results.json` | `ad92868b312122d47d6a25b74c9b9bb13034c26803e30f392b480ce05eb736ca` |
| `validation.json` | `5ba5eb04aaf452540ba9c273196b1ad161e3f515c60321bd6b43d02d627ce8b8` |
| `commands.json` | `d2e7bf75cb42481e27363e2dab9f240a63d05b8153e9c52ac6baee329b626349` |

两个 FastLanes contract 都记录 `benchmark_source_clean=true`、`git_tracked_dirty=false`。仓库中仅有用户原有的
`.cache/` 和 `galp/examples/image_order_benchmark/res` 两个未跟踪目录，它们未进入 benchmark runtime source，
也没有被本阶段修改或删除。
