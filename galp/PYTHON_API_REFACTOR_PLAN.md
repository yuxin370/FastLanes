# GALP Python 公共 API 重构：实施状态与后续计划

- 状态：公共 Reader/Profile API 已实现；native-owned Pipeline/Iterator 尚待后续阶段
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

本轮已完成第一阶段拆分：Python 正常调用只提交样本、transform descriptor 和语义
profile，不再提交 allocator、I/O、planner、stream 或 kernel 参数。

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

私有 `_galp_direct_dct` binding 提供通用 profile 驱动入口：

- `DirectDctReader.plan(image_ids, profile_id, transforms=None)`；
- `DirectDctReader.prefetch(image_ids, profile_id, transforms=None)`；
- `DirectDctReader.read(image_ids, profile_id, transforms=None)`；
- `available_direct_dct_profiles()`；
- `direct_dct_profile_info(profile_id)`。

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

preview = reader.plan([0, 1, 2, 3], VALIDATION)
future = reader.prefetch([0, 1, 2, 3], VALIDATION)
batch = future.read()
logits = model(batch.y, batch.cbcr)
batch.record_stream()
```

公共类型为：

- `galp.profiles.DirectDctProfile`：只含语义 ID 的不可变 descriptor，不含 runtime 字段；
- `galp.torch.DirectDctReader`：稳定 reader；
- `galp.torch.DirectDctFuture`：单消费者异步结果；
- `galp.torch.DirectDctBatch`：模型 tensor、样本身份、稳定生命周期操作。

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
| `initialization_stats` | `dict` | 是，观测 | reader 初始化摘要 |
| `profile_info(profile)` | `dict` | 是，观测 | profile schema、输出 layout、runtime policy identity |

### 4.2 `plan` / `prefetch` / `read`

| 参数 | 类型 | 必需 | 含义 |
| --- | --- | --- | --- |
| `image_ids` | `Sequence[int]` | 是 | 请求的全局图片 ID，保持输入顺序 |
| `profile` | `DirectDctProfile | str` | 是 | 模型语义 profile；推荐传 profile 对象 |
| `transforms` | `Sequence[Mapping] | None` | 否 | 每样本 crop/flip/sample-id/augmentation-key 语义 |

公共调用中没有 cache capacity、rowgroup prefetch、planless、workset、scheduling、CTA、
stream priority、double buffer、bounded read、manual reclaim 或 buffer keepalive 参数。

### 4.3 `DirectDctBatch`

| 字段/方法 | 含义 |
| --- | --- |
| `y`, `cbcr`, `coefficients` | GPU-resident 输出；profile 决定哪些字段有效 |
| `tensors` | `(y, cbcr)` 便捷视图 |
| `global_image_ids` | 实际输出样本身份 |
| `transform_descriptors` | native 接受的逐样本变换描述 |
| `layout`, `profile_id` | 输出合同身份 |
| `record_stream()` | 在当前 PyTorch stream 上延长底层 buffer 生命周期 |

完整 native counters 不属于 `DirectDctBatch`。benchmark/调试工具必须显式从
`galp.torch.diagnostics` 调用 `execution_stats()`、`execution_stats_snapshot()` 或
`cache_stats()`；prefetch 内部计时同样通过 `prefetch_stats()` 读取。该模块不承诺字段级
稳定性。

### 4.4 `DirectDctFuture`

稳定接口只包含 `ready/started/active/finished` 状态、`read()` 和 `cancel()`；producer、
planning、I/O staging 与 ordered-submission 计时不作为 Future 属性，通过 diagnostics
读取。submission release 仍是 canonical inference overlap 协调器的私有过渡钩子。

## 5. canonical 迁移状态

| 调用方 | 当前入口 | 状态 |
| --- | --- | --- |
| RGB-no-more inference | `DirectDctReader` + `rgbnomore.VALIDATION` | 已迁移 |
| RGB-no-more training facade | `DirectDctReader` + `rgbnomore.VALIDATION` | 已迁移 |
| DCT-major production | `DirectDctReader` + `VALIDATION_CENTER_CROP_512` | 已迁移 |
| DCT-major inspect/metadata tool | 公共 reader/profile | 已迁移 |
| A/B diagnostics 与历史复现实验 | 私有 low-level binding | 有意保留为 experimental |

原 `galp/torch/rgbnomore_dct_profile.py` 已移至
`galp/benchmarks/system_rgbnomore/diagnostics/`；通用 PyTorch 包不再携带应用专有 raw
配置字典。

production contract 仍记录 runtime policy ID 用于可复现性和证据校验，但不再把 policy 的
内部字段复制为 Python options。

## 6. 尚未完全封装的机制

公共“配置面”已经收敛，但 canonical benchmark adapter 仍负责编排部分执行生命周期：

- inference adapter 维护两批 future FIFO，并通过一个私有兼容钩子协调跨 batch submission；
- DCT-major adapter 感知 physical shard segment、跨 segment 拼接和 batch keepalive；
- benchmark 仍读取完整 native counter 字典形成验收证据；
- 私有 binding 为 diagnostics 保留宽参数的 `plan_batch/read_batch/prefetch_batch`。

这些属于下一阶段的执行编排封装，不应重新变成 profile 字段或 CLI 参数。

## 7. 后续修改计划

### P0：native-owned Pipeline/Iterator

1. 实现 `galp.torch.DirectDctPipeline` 与有界 iterator；
2. native 统一拥有 plan → I/O → CUDA submission → completion → reclaim 状态机；
3. 将 FIFO、cancel/reset/close、异常传播和 CUDA event handoff 移出 adapter；
4. 在 native 中完成 DCT-major segment stitching，Python 只接收普通 model batch。

完成标准：canonical Python 不出现 submission gate、physical segment、manual keepalive 或
reclaim 编排。

### P0：稳定 metrics schema

1. 从完整内部 counters 中提炼版本化指标：consumer wait、I/O、decode、transform、
   logical/physical bytes、peak transient memory；
2. 完整 planner/allocator/kernel counters 移入 `galp.diagnostics`；
3. production adapter 不再逐字段解释 native 实现计数器。

### P1：experimental 隔离

1. 将宽参数 raw binding 明确标记为 private/experimental；
2. 盘点 diagnostics 真实使用的开关，删除无调用和已被最优 profile 淘汰的历史分支；
3. 将 diagnostics 中暂存的 RGB-no-more raw 字典改为 native profile 的显式 experimental
   序列化接口，消除 C++/Python 语义值重复；
4. 历史报告继续保留原始命令，但必须保持历史快照标记。

## 8. 验收要求

- public profile/API 单元测试；
- inference、training、DCT-major CPU/static tests；
- native binding 与 JPEG-DCT targets 构建；
- crop 对齐及非 16 倍数边界；
- async completion transient-memory accounting；
- io_uring short-read retry；
- scheduled sidecar 冻结与 cache-eviction contract；
- GPU 语义与性能 benchmark 不低于两个冻结 runtime policy 的证据基线。

GPU 驱动不可用时，只能记录 CPU/static/build 结果，不能把 GPU 验收标记为通过。
