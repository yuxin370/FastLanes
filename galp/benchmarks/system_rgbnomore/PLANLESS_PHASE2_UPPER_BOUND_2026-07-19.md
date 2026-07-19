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

