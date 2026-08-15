# GALP Python 公共 API 重构：实施状态与后续计划

- 状态：公共 Reader/Profile、native-owned Pipeline/Iterator 与稳定 metrics schema 已实现；
  DCT-major native stitching 尚待后续阶段
- 更新日期：2026-08-14
- 范围：Direct-DCT 配置所有权、C++ profile、PyTorch API、canonical benchmark adapter
- 设计参考：DALI 的“公共 pipeline 接口 + 私有执行引擎”边界，不照搬其 DSL

## 1. 审核结论

原接口把三类性质不同的配置混在 `JpegDctDeviceBatchOptions` 和 Python 调用参数中：

1. **应用语义**：输出 tensor layout、DCT 网格、crop reference、clamp、dtype、归一化、
   chroma sampling 兼容范围；
2. **通用运行策略**：prefetch、workset、plan cache、planless、调度、stream priority、
   kernel launch、double buffer、bounded read；
3. **存储事实**：manifest 版本、image-major/block-major 布局、shard/rowgroup 边界和 sidecar。

其中只有第一类应该由模型 profile 定义；第二类应由 native runtime 拥有并版本化；第三类
应从 manifest/companion metadata 读取。把三者合成
`prefetch_rgbnomore_val_block_major_batch()` 会让应用名、存储布局和执行机制进入同一个
公共方法名，既不通用，也无法独立演进。

当前已完成两阶段拆分：Python 正常调用只提交逻辑 batch、transform descriptor 和语义
profile；有界预取、future、submission gate、CUDA completion 与 reclaim 由 native pipeline
拥有。正常调用不再提交或观察 allocator、I/O planner、stream、kernel launch、rowgroup 或
buffer keepalive 机制。

## 2. 已实现的分层

### 2.1 通用 native 运行策略

文件：`galp/include/galp/profiles/direct_dct.hpp`

- `DirectDctRuntimePolicy`：只包含 cache、prefetch、workset、planless、调度、launch、stream、
  double-buffer 和 bounded-read 策略；
- `DirectDctOutputProfile`：只包含模型输出 layout、grid transform 和 coefficient selection；
- `RegisteredDirectDctProfile`：在 native registry 中把语义输出与 runtime policy 组合；
- `materialize_direct_dct_options()`：组合点集中在 C++，Python 不再构造
  `JpegDctDeviceBatchOptions`。

当前保留两个经生产验收的通用 runtime policy：

| runtime policy ID | 用途 | Python 可调 |
| --- | --- | --- |
| `compact-v3-planless-limited-o512-c512-v1` | Compact-v3 image-major 生产策略 | 否 |
| `block-major-p4-scheduled-bounded-110-v1` | block-major scheduled bounded-I/O 生产策略 | 否 |

这些 ID 是 provenance/兼容性身份，不是供调用方选择的 knob。若将来替换实现，应发布新的
policy ID 并重新做性能、内存和语义验收。

`galp/include/galp/profiles/registry.hpp` 是唯一应用 profile 注册点；通用 PyTorch binding
只依赖 registry，不直接拼装 RGB-no-more 配置。

### 2.2 RGB-no-more 专有语义

文件：`galp/include/galp/profiles/rgbnomore.hpp`、`galp/profiles/rgbnomore.py`

RGB-no-more 文件现在只拥有以下模型语义：

- Y 输出 `28×28` blocks、Cb/Cr 输出 `14×14` blocks；
- crop reference、偶数 block 对齐和 chroma crop 比例；
- dequantize、clamp `[-1024, 1016]`；
- FP32 affine `(x + 4) / 1020`；
- grayscale 与允许的 JPEG chroma sampling ratios。

它不再定义 rowgroup batch、prefetch worker、plan cache、CTA、stream、double buffer 或
bounded-read 参数。通用 planner/kernel 中的 `kRgbNoMoreDown2Conversion` 也已更名为
`kReferenceDown2Conversion`；2:1 DCT 变换实现不再以某个 benchmark 命名。

### 2.3 私有 binding 与公共 Python API

私有 `_galp_direct_dct` binding 内部仍提供 profile 驱动 reader/future 入口，并新增：

- `DirectDctReader.pipeline(profile_id)`；
- `DirectDctPipeline.reset(image_id_batches, transforms_by_batch=None)`；
- `DirectDctPipeline.__next__()`；
- `available_direct_dct_profiles()`；
- `direct_dct_profile_info(profile_id)`。

reader/future 的宽入口只属于下划线开头的私有 extension 和 diagnostics；不是
`galp.torch` 公共合同。

以下 benchmark 专有方法已删除：

- `plan_rgbnomore_val_batch`；
- `prefetch_rgbnomore_val_batch`；
- `prefetch_rgbnomore_val_block_major_batch`。

正常用户只使用公开包：

```python
from galp.profiles.rgbnomore import VALIDATION
from galp.torch import DirectDctReader

reader = DirectDctReader(
    "/data/imagenet/manifest.bin",
    module_path="build/galp/torch",
)

pipeline = reader.pipeline(VALIDATION).start(
    [[0, 1, 2, 3], [4, 5, 6, 7]],
)
for batch in pipeline:
    logits = model(batch.y, batch.cbcr)
```

公共类型为：

- `galp.profiles.DirectDctProfile`：只含语义 ID 的不可变 descriptor，不含 runtime 字段；
- `galp.torch.DirectDctReader`：稳定 reader；
- `galp.torch.DirectDctPipeline`：native-owned 有界 iterator，不暴露 future 或 queue depth；
- `galp.torch.DirectDctBatch`：模型 tensor 与样本身份；
- `galp.torch.DirectDctMetrics`：版本化的聚合观测，不含内部 counter 名称。

native extension 采用延迟导入。公共 reader 会校验 profile schema 与 profile ID；
`profile_info()` 返回 native 实际 runtime policy ID，benchmark contract 可将其用于
provenance 校验，但语义 profile 对象本身不持有该字段。

## 3. 当前语义 profile 表

| Python 常量 | 公共 profile ID | crop reference | 输出 | native runtime policy |
| --- | --- | ---: | --- | --- |
| `rgbnomore.VALIDATION` | `rgbnomore-validation-v1` | `32×32` blocks | Y `28×28`、CbCr `14×14`、FP32 | `compact-v3-planless-limited-o512-c512-v1` |
| `rgbnomore.VALIDATION_CENTER_CROP_512` | `rgbnomore-validation-center-crop-512-v1` | `64×64` blocks | Y `28×28`、CbCr `14×14`、FP32 | `block-major-p4-scheduled-bounded-110-v1` |

`64×64` reference 表示“512×512 输入上的固定中心 crop”语义，不表示 block-major
存储本身。当前 DCT-major 数据与该语义一起验收，所以 registry 将它组合到 block-major
生产策略；两个定义仍位于不同类型中，不能再从名称或 Python 参数互相推导。

## 4. 公共参数表

### 4.1 `DirectDctReader`

| 参数/属性 | 类型 | 公共 | 含义 |
| --- | --- | --- | --- |
| `manifest_path` | `str | Path` | 是 | 数据集 manifest |
| `module_path` | `str | Path | None` | 是 | 可选 native binding 搜索目录，部署辅助参数 |
| `image_count` | `int` | 是 | manifest 中可读取图片数 |
| `profile_info(profile)` | `dict` | 是，观测 | profile schema、输出 layout、runtime policy identity |
| `pipeline(profile)` | `DirectDctPipeline` | 是 | 创建固定语义 profile 的 native iterator |
| `read(image_ids, profile, transforms=None)` | `DirectDctBatch` | 是 | 无预取需求时的同步便捷入口 |

`initialization_stats`、binding import latency、planner preview、JPEG metadata 与 rowgroup
storage 查询已移至 `galp.diagnostics.direct_dct`，不属于模型调用 API。

### 4.2 `DirectDctPipeline.start`

| 参数 | 类型 | 必需 | 含义 |
| --- | --- | --- | --- |
| `image_id_batches` | `Sequence[Sequence[int]]` | 是 | 按消费顺序排列的逻辑 batch |
| `transforms_by_batch` | `Sequence[Sequence[Mapping] | None] | None` | 否 | 与逻辑 batch 对齐的逐样本语义变换 |

公共调用中没有 cache capacity、rowgroup prefetch、planless、workset、scheduling、CTA、
stream priority、double buffer、bounded read、prefetch depth、future、submission gate、manual
reclaim 或 buffer keepalive 参数。

### 4.3 `DirectDctBatch`

| 字段/方法 | 含义 |
| --- | --- |
| `y`, `cbcr`, `coefficients` | GPU-resident 输出；profile 决定哪些字段有效 |
| `tensors` | `(y, cbcr)` 便捷视图 |
| `global_image_ids` | 实际输出样本身份 |
| `transform_descriptors` | native 接受的逐样本变换描述 |
| `layout`, `profile_id` | 输出合同身份 |
| `metrics` | `DirectDctMetrics` 稳定聚合观测 |

底层 batch 生命周期由 PyTorch tensor 的 owner/deleter 自动管理，公共 API 不提供
`record_stream()` 或 keepalive。binding 在输出 tensor 被取得时自动登记当前 CUDA consumer
stream，并在 tensor storage 最终释放时把 native owner 延迟到所有已登记 stream 的 event
完成之后；默认流和非默认流使用同一套生命周期规则。完整 native counters 不属于
`DirectDctBatch`；benchmark
审计/调试工具必须显式从
`galp.diagnostics.direct_dct` 调用 `execution_stats()`、`execution_stats_snapshot()` 或
`cache_stats()`。该模块不承诺字段级稳定性。

### 4.4 `DirectDctMetrics`

schema `galp-direct-dct-metrics-v2` 固定提供 completion 状态、consumer wait、submit-to-ready、
producer、planning、I/O、decode、transform 时间，以及 logical/physical bytes 和 peak
transient memory。`submit_to_ready_ms` 在 producer 发布 future 时定格，不包含 consumer
空闲时间；`transform_ms` 是互斥 transform 阶段之和，不重复累计 planless alias counter。
指标读取永不隐式同步：单 batch 和 pipeline snapshot 在 GPU event 尚未完成时返回
`complete=False`；native pipeline 只暂存尚未完成的 owner，并在正常 CUDA 同步边界后给出
完整累计值。它不公开 rowgroup、workset、plan cache、allocator、kernel launch 或 stream
priority counter。

## 5. canonical 迁移状态

| 调用方 | 当前入口 | 状态 |
| --- | --- | --- |
| RGB-no-more inference | `DirectDctPipeline` + `rgbnomore.VALIDATION` | 已迁移；无 Python FIFO/gate |
| RGB-no-more training facade | 一个 epoch schedule 对应一个 `DirectDctPipeline` | 已迁移；无 Python Future/FIFO/K sweep |
| DCT-major production | `DirectDctPipeline` + `VALIDATION_CENTER_CROP_512` | 预取/gate 已迁移；stitching 待下沉 |
| DCT-major inspect/metadata tool | public reader + diagnostics opt-in | 已迁移 |
| model-facing Direct-DCT example | `DirectDctPipeline` + stable metrics | 已迁移；无 raw runtime 参数 |
| A/B diagnostics 与历史复现实验 | 私有 low-level binding | 有意保留为 experimental |

原 `galp/torch/rgbnomore_dct_profile.py` 已移至
`galp/benchmarks/system_rgbnomore/diagnostics/`；通用 PyTorch 包不再携带应用专有 raw
配置字典。

production contract 仍记录 runtime policy ID 用于可复现性和证据校验，但不再把 policy 的
内部字段复制为 Python options。

native iterator 的有界 in-flight 窗口属于 runtime 实现，并采用当前验收过的唯一配置；训练
CLI 不再接受 `--prefetch-depth`，Gate 3 也不再对同一个 native 策略贴上 2/4/8 等不同标签。
sample-order 证据从 native 已实际接收的 batch 数生成，不再把尚未进入有界 pipeline 的完整
逻辑 schedule 误报为 prefetched。

## 6. 尚未完全封装的机制

公共“配置面”和普通 batch pipeline 已收敛，但仍有以下后续工作：

- DCT-major adapter 仍感知 physical shard segment，并在 Python 中进行跨 segment tensor 拼接；
- benchmark audit 模式仍读取完整 native counter 字典形成历史验收证据；
- 私有 binding 为 diagnostics 保留宽参数的 `plan_batch/read_batch/prefetch_batch`。

这些属于下一阶段的执行编排封装，不应重新变成 profile 字段或 CLI 参数。

## 7. 后续修改计划

### P0：native-owned Pipeline/Iterator

1. ~~实现 `galp.torch.DirectDctPipeline` 与有界 iterator；~~ 已完成；
2. ~~native 统一拥有 plan → I/O → CUDA submission → completion → reclaim 状态机；~~ 已完成；
3. ~~将 FIFO、cancel/reset/close、异常传播和 CUDA event handoff 移出 adapter；~~ 已完成；
4. 在 native 中完成 DCT-major segment stitching，Python 只接收普通 model batch。

完成标准：canonical Python 不出现 submission gate、physical segment、manual keepalive 或
reclaim 编排。

### P0：稳定 metrics schema

1. ~~从完整内部 counters 中提炼版本化指标：consumer wait、I/O、decode、transform、
   logical/physical bytes、peak transient memory；~~ 已完成 `galp-direct-dct-metrics-v2`；
2. 完整 planner/allocator/kernel counters 已从稳定 PyTorch API 移入顶层
   `galp.diagnostics.direct_dct`；
3. inference/training 热路径已使用 native 累计 metrics，并只在已有同步边界要求
   `complete=True`；DCT-major 历史 acceptance 聚合仍待切换。

### P1：experimental 隔离

1. 将宽参数 raw binding 明确标记为 private/experimental；
2. 盘点 diagnostics 真实使用的开关，删除无调用和已被最优 profile 淘汰的历史分支；
3. 将 diagnostics 中暂存的 RGB-no-more raw 字典改为 native profile 的显式 experimental
   序列化接口，消除 C++/Python 语义值重复；
4. 历史报告继续保留原始命令，但必须保持历史快照标记。

## 8. 验收要求

- public profile/API 单元测试；
- 非默认 CUDA consumer stream 的 tensor/native-owner 生命周期；
- producer-ready 时间戳、planless transform 不重复计时、同步边界 metrics 完整性；
- bounded pipeline 实际接收的 prefetch/sample-order 证据；
- inference、training、DCT-major CPU/static tests；
- native binding 与 JPEG-DCT targets 构建；
- crop 对齐及非 16 倍数边界；
- async completion transient-memory accounting；
- io_uring short-read retry；
- scheduled sidecar 冻结与 cache-eviction contract；
- GPU 语义与性能 benchmark 不低于两个冻结 runtime policy 的证据基线。

GPU 驱动不可用时，只能记录 CPU/static/build 结果，不能把 GPU 验收标记为通过。
