# 最新端到端比较运行说明

本文说明如何运行 GALP 与 RGB-no-more 的正式 DCT 端到端比较，以及每个参数为什么这样设置。

## 1. 比较范围

正式计时边界为：

```text
读取 → DCT 解压/预处理 → GPU tensor → 模型 forward → 准确率统计
```

不计入正式时间：

- 构建；
- manifest 创建；
- 模型加载；
- warmup；
- payload fingerprint 计算。

GALP 与 RGB-no-more 都使用 DCT 输入域、相同样本顺序、标签、DCT checkpoint、模型架构和 FP32 精度，因此可以做严格语义与性能比较。

## 2. 构建最新 Torch 扩展

在仓库根目录运行：

```bash
cd /home/tangyuxin/gfastlanes/FastLanes

cmake -E env CCACHE_DIR=/tmp/galp-ccache \
  cmake --build build --target _galp_direct_dct -j2
```

参数说明：

| 字段 | 含义 | 设置原因 |
|---|---|---|
| `cmake -E env` | 为构建命令设置临时环境 | 不长期修改当前 shell |
| `CCACHE_DIR=/tmp/galp-ccache` | 指定编译缓存目录 | 避免默认 ccache 目录权限问题 |
| `cmake --build build` | 使用已有 CMake build tree | 保持当前 Release/CUDA 架构配置 |
| `--target _galp_direct_dct` | 只构建 Torch Direct-DCT 扩展 | Python 端到端管线实际加载该模块 |
| `-j2` | 两个并行构建任务 | 控制 NVCC 内存和系统负载；不影响最终运行性能 |

关键产物：

```text
build/galp/torch/_galp_direct_dct.cpython-311-x86_64-linux-gnu.so
```

## 3. 正式运行命令

以下命令把所有结果写到本目录的 `res/e2e_compare_latest`：

```bash
cd /home/tangyuxin/gfastlanes/FastLanes

OUT="$PWD/galp/benchmarks/system_rgbnomore/res/e2e_compare_latest"

numactl --cpunodebind=0 --membind=0 env \
  PYTHONPATH=build/galp/torch \
  /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset e2e \
  --output-dir "$OUT" \
  --benchmark-id galp-e2e-compare-latest \
  --pipelines galp_planless rgbnomore \
  --data-root /tmp/galp-image-major-1024-dataset \
  --split val \
  --index-csv /tmp/galp-image-major-1024-fixed/index.csv \
  --galp-manifest /tmp/galp-wizard-1024-tiled-z32/tiled_z_32/manifest.bin \
  --galp-label-map-json /tmp/galp-image-major-1024-fixed/labels.json \
  --torch-binding-dir build/galp/torch \
  --galp-preprocess rgbnomore-val-pushdown \
  --galp-cache-capacity-mib 0 \
  --galp-decode-batch-rowgroups 64 \
  --galp-batch-prefetch-depth 2 \
  --galp-rowgroup-prefetch-depth 64 \
  --galp-rowgroup-prefetch-workers 16 \
  --galp-rowgroup-prefetch-min-decode-batches 1 \
  --batch-size 64 \
  --warmup-batches 5 \
  --measurement-batches 10 \
  --repeats 5 \
  --workers 8 \
  --semantic-samples 640 \
  --seed 11997733 \
  --device cuda:0 \
  --precision fp32 \
  --refresh-galp-payload-fingerprints
```

重复运行时，建议使用新的输出目录，例如：

```bash
OUT="$PWD/galp/benchmarks/system_rgbnomore/res/e2e_compare_20260717_r2"
```

这样不会把不同运行的合同和结果混在一起。

## 4. Shell、GPU 与 NUMA 字段

| 字段 | 含义 | 设置原因 |
|---|---|---|
| `OUT=...` | 本次结果目录 | 统一保存合同、日志和报告 |
| `numactl --cpunodebind=0` | 将进程调度到 NUMA node 0 的 CPU | 当前机器 GPU0 的 CPU affinity 属于 NUMA0，减少跨 NUMA 调度 |
| `--membind=0` | 从 NUMA0 分配主机内存 | 减少 rowgroup buffer、pinned memory 和 H2D 前的跨 UPI 访问 |
| `env` | 给后续命令设置环境变量 | 与 `numactl` 组合使用 |
| `PYTHONPATH=build/galp/torch` | 指定本地 Torch 扩展搜索路径 | 防止加载旧安装或其他 build tree 的 `.so` |
| `fastlanes-cuda/bin/python` | 使用固定 CUDA Python 环境 | 保持 PyTorch、CUDA、NumPy 和模型依赖版本一致 |
| `--device cuda:0` | 使用可见设备中的 GPU0 | 与 NUMA0 绑定匹配；当前为 RTX 4090 |

换机器或换 GPU 前先检查：

```bash
nvidia-smi topo -m
```

然后让 `cpunodebind`、`membind` 与目标 GPU 的 `NUMA Affinity` 一致。若设置了 `CUDA_VISIBLE_DEVICES`，`cuda:0` 表示“可见 GPU 中的第一个”，未必是物理 GPU0。

## 5. Benchmark 身份与输出字段

| 参数 | 含义 | 设置原因 |
|---|---|---|
| `--preset e2e` | 正式端到端 benchmark preset | 启用正式聚合、validation 和性能门槛，而不是只检查可运行性的 smoke |
| `--output-dir "$OUT"` | 结果输出目录 | 保存完整可追溯证据 |
| `--benchmark-id` | 实验标识 | 写入合同和结果，便于区分多次运行；不参与计时 |

命令中显式提供的 batch、warmup、measurement 和 repeat 会覆盖 preset 的对应默认值；`e2e` 仍提供正式验证和聚合策略。

## 6. Pipeline 字段

```text
--pipelines galp_planless rgbnomore
```

`galp_planless` 路径：

```text
FLS shard
→ selective DCT decode
→ resize/crop/transformed-grid pushdown
→ GPU DCT tensor
→ DCT ViT
```

`rgbnomore` 参考路径：

```text
JPEG
→ RGB-no-more 原生 DCT loader/transform
→ GPU DCT tensor
→ 同一个 DCT ViT
```

只选择这两条路径，是因为它们属于相同 DCT 输入域，可以严格比较 input tensor、logits、Top-1 和 Top-5。

如需额外运行 RGB 域诊断，可改成：

```text
--pipelines galp_planless rgbnomore dali pytorch
```

DALI/PyTorch 使用 RGB 模型，与 DCT 模型只能做 system-level reference，不能跨域逐元素比较。

## 7. 数据集身份字段

| 参数 | 含义 | 设置原因 |
|---|---|---|
| `--data-root` | 原始 ImageNet JPEG 根目录 | RGB-no-more 参考管线需要原始 JPEG，也用于核对样本身份 |
| `--split val` | 使用验证集 | 与 ImageNet evaluation recipe 和 checkpoint 对齐 |
| `--index-csv` | 规定样本路径、ordinal、顺序和 label | 避免不同管线自行扫描目录产生顺序或标签偏差 |
| `--galp-manifest` | GALP shard manifest | 指向 `.fls`、metadata 和 image/shard 映射 |
| `--galp-label-map-json` | GALP image ID 到类别的映射 | 保证 FLS 内部 image index 与参考标签一致 |
| `--torch-binding-dir` | Torch native binding 目录 | 记录到合同并传递给 pipeline 子进程 |

`PYTHONPATH` 决定当前 Python 实际从哪里 import；`--torch-binding-dir` 则把该位置写入合同并传给子进程，两者用途不同。

## 8. GALP 预处理与 decode 字段

### `--galp-preprocess rgbnomore-val-pushdown`

使用正式 production 路径，在 DCT 域下推 resize/crop/transformed-grid：

```text
DCT block 选择
→ 只读需要的 rowgroup/vector
→ 只解压需要的数据
→ 直接产生模型输入 tensor
```

相比诊断用的 `rgbnomore-val`，它减少完整中间 DCT 图、无关 vector 解码和额外数据搬运。

### `--galp-cache-capacity-mib 0`

关闭 decoded-rowgroup cache，原因是：

- 测量真实 streaming read/decode；
- 避免不同 repeat 的 cache 热度不一致；
- 防止 cache hit 掩盖 I/O 和解压成本；
- 与当前 image-major production baseline 一致。

### `--galp-decode-batch-rowgroups 64`

batch size 为 64，image-major 布局通常每张图对应一个 rowgroup。该值允许将一个 batch 的 rowgroup 组织成一个统一 workset 和一次 decode dispatch，同时作为 legacy layout 的兼容上限。

## 9. Prefetch 字段

| 参数 | 含义 | 设置原因 |
|---|---|---|
| `--galp-batch-prefetch-depth 2` | 有界预取后续两个 batch | 将下一批 native read/decode 与当前模型 forward 重叠，同时限制内存占用 |
| `--galp-rowgroup-prefetch-depth 64` | 最多预取 64 个 rowgroup | 覆盖一个 64-image batch |
| `--galp-rowgroup-prefetch-workers 16` | 16 个 host 读取 worker | 在并行 I/O 与线程/文件系统争用之间折中 |
| `--galp-rowgroup-prefetch-min-decode-batches 1` | 一个逻辑 decode workset 也启用并行读取 | image-major 路径本来就会把整个 batch 合并为一个 workset，不能因 workset 数为 1 而关闭预取 |

不使用 64 个读取线程，是因为线程数继续增加通常会带来文件系统队列、page cache 和调度争用，不等于线性加速。

## 10. Batch 与统计字段

| 参数 | 含义 | 设置原因 |
|---|---|---|
| `--batch-size 64` | 每个 batch 64 张图 | 与 DCT ViT、GPU 内存和 image-major rowgroup 合并基线一致 |
| `--warmup-batches 5` | 每个 repeat 先运行 5 个不计时 batch | 完成 CUDA context、kernel/module、allocator、cache 初始化和 GPU 升频 |
| `--measurement-batches 10` | 每个 repeat 测量 10 个 batch | 每轮测量 640 张图，足以形成稳定吞吐 |
| `--repeats 5` | 独立重复五轮 | `e2e` 聚合排除 repeat 0，用后四轮中位数降低冷 cache 和系统噪声 |
| `--workers 8` | 外层 loader/prepare worker 数 | 与既有 baseline 一致；各 pipeline 的 worker 实现语义会在报告中记录 |

GALP 内部 rowgroup I/O 并行度由 `--galp-rowgroup-prefetch-workers 16` 单独控制，不等同于外层 `--workers 8`。

## 11. 语义、随机性与精度字段

| 参数 | 含义 | 设置原因 |
|---|---|---|
| `--semantic-samples 640` | 保存并验证 640 个样本 | 正好覆盖一轮正式 measured trace，即 `10 × 64` |
| `--seed 11997733` | 固定样本和验证顺序 | 确保不同实现和日期使用相同 trace，可通过 SHA256 核对 |
| `--precision fp32` | 模型使用 FP32 | 与历史 baseline/checkpoint 一致，避免 AMP kernel 和数值误差成为混杂因素 |

semantic validation 会核对：

- sample ID；
- ordinal；
- label；
- measured trace SHA256；
- input tensor 数值；
- logits；
- Top-1 prediction agreement。

## 12. Payload fingerprint

```text
--refresh-galp-payload-fingerprints
```

显式计算或刷新 manifest、`.fls` 和 metadata 的文件身份与 SHA256，保证结果对应确切 payload，而不是只有路径相同。

fingerprint 计算在正式计时之外。文件没有变化、fingerprint cache 已存在时，日常重复运行可以省略；最终归档结果建议保留。

## 13. 输出文件

运行完成后查看：

```bash
sed -n '1,200p' \
  /home/tangyuxin/gfastlanes/FastLanes/galp/benchmarks/system_rgbnomore/res/e2e_compare_latest/report.md
```

结果目录中主要文件：

| 文件 | 内容 |
|---|---|
| `report.md` | 人类可读汇总 |
| `results.json` | 聚合指标、性能门槛和合同快照 |
| `results.csv` | 表格形式的结果 |
| `validation.json` | 性能与语义验证结论 |
| `pipeline_galp_planless.json` | GALP planless 每轮、每 batch 和 native counters |
| `pipeline_rgbnomore.json` | RGB-no-more 每轮详细结果 |
| `contract.json` | 完整、不可歧义的实验配置 |
| `commands.json` | 实际执行的 pipeline 子进程命令 |

## 14. 成功标准

`report.md` 应显示：

```text
Validation: PASS
galp median_throughput_images_per_s >= 3000
galp vs rgbnomore: PASS
logits top-1 agreement = 1.0000
```

性能聚合排除 repeat 0，并使用后四轮结果的中位数。

最近一次已经通过的 GALP-only 正式结果为：

```text
吞吐：3250.396 img/s
平均延迟中位数：19.692 ms/batch
Top-1：0.6984
Top-5：0.9141
```

相对历史 GALP baseline：

```text
吞吐：3129.080 → 3250.396 img/s，+3.877%
延迟：20.468 → 19.692 ms/batch，-3.791%
```

## 15. 运行前建议

正式运行前确认 GPU0 没有其他任务：

```bash
nvidia-smi
```

如果模型 forward 明显高于历史约 12 ms/batch，或者同一个 repeat 内吞吐持续爬升，通常说明 GPU 时钟、系统负载或 page cache 尚未稳定。不要用这种结果判断代码回退，应待 GPU 空闲后重新使用新输出目录运行。
