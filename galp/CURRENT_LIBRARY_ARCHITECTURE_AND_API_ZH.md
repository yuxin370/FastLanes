# GALP 当前架构、模块职责与使用指南

本文描述当前仓库中 GALP 的实际结构和公开接口，面向两类读者：

- 希望直接使用 Python API 读取 Direct-DCT 数据、训练或推理的用户；
- 需要维护 C++、CUDA、PyTorch 扩展或数据流水线的开发者。

本文尽量使用通俗语言解释各层职责，但会保留必要的专业术语和所有权约束。源码始终是最终事实来源；本文所列路径均相对于仓库根目录。

## 1. GALP 是什么

GALP 是 FastLanes 仓库中的 GPU 数据读取与解压子系统。它目前有两条主要能力线：

1. 通用 FLS Table 解压：读取 FastLanes/FLS 文件，将列数据解压成表格视图。
2. Direct-DCT 图像流水线：直接读取 JPEG DCT 域数据，在 GPU 上完成解码、选择系数、变换和物化，为 PyTorch 模型提供 Tensor。

另外还有一条面向训练研究的 PLS 流水线。PLS 复用 Direct-DCT runtime，但它仍属于 Advanced / Experimental API，不属于稳定入口。

从使用者角度，可以把 GALP 理解为：

```text
数据文件 / manifest
    -> Native 规划与异步 I/O
    -> CUDA 解码、变换和物化
    -> PyTorch Tensor
    -> 模型消费
```

Python 只负责表达请求、包装 Tensor、转发 `record_stream` 和翻译异常。真正的调度、物理分段、资源回收和指标聚合都由 Native 层负责。

## 2. 当前核心架构

Direct-DCT 默认调用链如下：

```text
Python Stable API
    DirectDctReader / DirectDctPipeline / DirectDctBatch
        |
        v
PyTorch C++ 扩展（薄适配层）
        |
        v
NativeLogicalBatchPipeline
    逻辑 batch 调度、队列和 backpressure
        |
        v
SegmentPlan + canonical shard execution
    跨 shard 分段、物理执行计划和 logical output 映射
        |
        v
DirectDctRuntime
    planner / I/O staging / workset / CUDA submission
        |
        v
现有 JPEG-DCT CUDA kernels
    gather / decode / transform / materialization
        |
        v
NativeBatchLease / NativeBatchCompletion
    producer/consumer completion、Tensor 生命周期和回收
        |
        v
PyTorch Tensor + DirectDctMetricsAggregator
```

当前生产所有权为：

| 概念 | 唯一 owner | 主要代码位置 |
| --- | --- | --- |
| 逻辑 batch 调度 | `NativeLogicalBatchPipeline` | `galp/src/direct_dct/native_logical_batch_pipeline.*` |
| 物理分段与 logical assembly | Native physical orchestration | `galp/src/direct_dct/physical_layout_planner.*` 及 JPEG canonical execution |
| producer/consumer 生命周期 | `NativeBatchLease` / `NativeBatchCompletion` | `galp/src/direct_dct/native_batch_lifetime.*` |
| metrics 聚合 | `DirectDctMetricsAggregator` | `galp/src/direct_dct/direct_dct_metrics.*` |
| coefficient selection | profile/policy resolution 后的统一 selection | `galp/src/direct_dct/profile_registry.*`、`resolved_execution_policy.*` 及 JPEG planner/workset |
| PLS pool 前瞻 | PLS pipeline 内的单步、有界 lookahead | `galp/src/api/direct_dct_pls.cpp` |
| CUDA 执行 | 现有 planner、workset 和 kernel | `galp/src/jpeg/`、`galp/src/cuda/`、`galp/src/engine/` |

这里的“唯一 owner”很重要：Python benchmark 可能组织实验流程，但不能再次实现生产调度器、资源回收器或 metrics reducer。

## 3. 目录与模块职责

### 3.1 C++ 公开头文件：`galp/include/galp/`

这是 C++ 用户应首先查看的目录。

| 路径 | 职责 |
| --- | --- |
| `galp/include/galp/stable.hpp` | Stable C++ 总入口；普通 Table 用户优先包含它 |
| `galp/include/galp/reader.hpp` | FLS 文件读取入口 `galp::Reader` |
| `galp/include/galp/table.hpp` | `Table`、`RowgroupView`、`ColumnView` 等物化结果视图 |
| `galp/include/galp/options.hpp` | Table 解压选项和支持的稳定数据类型 |
| `galp/include/galp/advanced/direct_dct.hpp` | Direct-DCT 的 Advanced C++ runtime/descriptor API |
| `galp/include/galp/advanced/direct_dct_pls.hpp` | PLS Advanced C++ API |
| `galp/include/galp/diagnostics/direct_dct.hpp` | 诊断信息、计划预览和内部观测入口 |
| `galp/include/galp/profiles/` | 语义 profile 的 C++ 声明 |

建议 Stable C++ 用户包含：

```cpp
#include <galp/stable.hpp>
```

不要把 `galp/galp.hpp` 当成新的稳定入口。它包含兼容性内容，主要用于旧调用者迁移。

### 3.2 公共 API 实现：`galp/src/api/`

这一层把公开合同连接到内部 runtime。

| 文件 | 职责 |
| --- | --- |
| `table.cu` | 通用 Table 解压 API 实现 |
| `direct_dct.cpp` | Advanced Direct-DCT C++ API 实现 |
| `direct_dct_pls.cpp` | PLS pool、microbatch、单步前瞻和执行状态 |
| `direct_dct_pls_postprocess.cu` | PLS 专用后处理 primitive，例如转换、统计、增强和 mixup |

`direct_dct_pls.cpp` 中的 pool context 是有界的：一个 active pool 加一个 preparing/ready pool。它复用现有 runtime 和生命周期机制，不建立第二套 scheduler。

### 3.3 数据模型与表达式：`galp/src/core/`

这一层保存基础类型、列/表描述、表达式和请求数据结构。它回答“数据是什么”，不负责 CUDA 调度。

维护时应避免把 I/O、CUDA stream 或 Python 对象放入这一层。

### 3.4 FLS 格式与读取：`galp/src/format/`

职责包括：

- 读取和验证 FLS 文件描述符、schema 和元数据；
- 构造 read plan；
- 将文件中的 rowgroup/column 信息交给执行层。

它主要服务通用 Table 路径。

### 3.5 通用执行引擎：`galp/src/engine/`

这一层负责把计划变成实际工作：

- operator 和 table runner；
- pipeline/prefetch；
- workset 构造；
- 输出 materialization；
- 通用执行期资源协调。

它不应该知道 Python benchmark 的目录结构或实验报告格式。

### 3.6 CUDA 基础设施：`galp/src/cuda/`

这一层提供 CUDA 资源和执行基础设施，例如：

- stream/event 的 RAII 管理；
- device/pinned arena 与 pool；
- launch 辅助；
- 内存分配和错误处理。

它负责“如何安全地使用 CUDA 资源”，但不决定某个语义 profile 需要哪些 DCT 系数。

### 3.7 编码和解码 primitive：`galp/src/codecs/`

这里是压缩编码及对应 device decode primitive。通用解压路径和部分 JPEG-DCT 执行会复用这些能力。

### 3.8 JPEG-DCT 执行：`galp/src/jpeg/`

这是 Direct-DCT 的底层执行主体，职责包括：

- JPEG-DCT metadata、manifest、shard 和 storage 描述；
- canonical plan 与物理读取范围；
- staged I/O；
- workset 和 device descriptor；
- gather、decode、transform、materialization kernel；
- 输出目标 offset 和跨 shard 映射。

重点文件包括：

| 文件 | 职责 |
| --- | --- |
| `jpeg_dct_planner.*` | 将 profile/request 转换成物理工作计划 |
| `jpeg_dct_device.cu` | 上传 descriptor/workset 并提交 device 执行 |
| `jpeg_dct_gather_kernels.cu` | DCT 数据 gather/decode 相关 kernel |
| `jpeg_dct_transform_kernels.cu` | transformed-grid 变换与物化 kernel |

完整 64 个系数和任意合法子集共享同一个 coefficient-selection 合同；完整集合只是 selection size 为 64 的普通输入。

### 3.9 Native Direct-DCT orchestration：`galp/src/direct_dct/`

这是重构后最关键的控制层。

| 文件 | 职责 |
| --- | --- |
| `native_logical_batch_pipeline.*` | Native scheduler、逻辑请求队列和 batch 交付 |
| `physical_layout_planner.*` | `SegmentPlan`，处理 logical batch 跨物理 shard 的分段 |
| `native_batch_lifetime.*` | producer/consumer completion、lease、Storage 生命周期和 reclaim |
| `direct_dct_metrics.*` | 唯一 metrics 聚合与 finalized 语义 |
| `profile_registry.*` | profile ID 到语义配置的注册和解析 |
| `resolved_execution_policy.*` | 将 profile 和请求选项归一化为下游 typed policy |

这一层的关键原则是：

- scheduler 只由 `NativeLogicalBatchPipeline` 拥有；
- lifetime 只由 `NativeBatchLease` / `NativeBatchCompletion` 拥有；
- Python 不维护第二套 pending queue 或 physical stitching；
- metrics 查询不会为了获得 GPU timing 主动增加 CUDA synchronize。

### 3.10 Python 与 PyTorch：`galp/torch/`

| 文件 | 职责 |
| --- | --- |
| `galp/torch/__init__.py` | Stable Python facade，只导出四个核心类型 |
| `galp/torch/direct_dct.py` | Stable Reader/Pipeline/Batch/Metrics 的 Python 适配 |
| `galp/torch/experimental.py` | PLS Experimental Python API |
| `galp/torch/direct_dct_torch.cpp` | pybind/Torch C++ 扩展，连接 Python 与 Native runtime |

这里是薄适配层。若在此处看到生产物理调度、`torch.cat` 拼接、主动 synchronize 或独立 backpressure 队列，通常意味着职责放错了层。

### 3.11 Profiles：`galp/profiles/`

Profile 是一个稳定的语义 ID，而不是一大包可以随意组合的底层开关。

Python 基本类型为：

```python
from galp import DirectDctProfile

profile = DirectDctProfile(id="rgbnomore-validation-v1")
```

常用预定义 profile 位于 `galp/profiles/rgbnomore.py`，例如：

- `VALIDATION`
- `VALIDATION_CENTER_CROP_512`
- `TRAINING_PLS`

调用者应优先使用预定义 profile，而不是依赖内部 planner 参数。

### 3.12 Diagnostics：`galp/diagnostics/`

Diagnostics API 用于排查问题和采集证据，不保证与 Stable API 相同的兼容级别。

主要入口是：

```python
import galp.diagnostics.direct_dct as direct_dct_diag
```

可访问的能力包括：

- execution stats 和快照；
- metric descriptors 和聚合检查；
- cache/pipeline/initialization stats；
- plan preview；
- image metadata；
- rowgroup storage bytes。

诊断数据不应成为普通模型代码的运行依赖。

### 3.13 Tests、benchmarks、experiments 和 tools

| 路径 | 正确职责 |
| --- | --- |
| `galp/tests/` | C++/CUDA contract 和集成测试 |
| `galp/torch/tests/`、`galp/torch/*_test.py` | Python/Torch API contract 与设备集成检查 |
| `galp/benchmarks/*/tests/` | 对应 benchmark 的合同、adapter 和报告测试 |
| `galp/benchmarks/` | 性能协议、训练 runner、对照系统、研究实验和证据报告 |
| `galp/examples/` | 可运行示例与验证入口 |
| `galp/tools/` | 数据转换、检查和辅助命令 |
| `galp/scripts/codegen/` | generated binding 的生成与一致性检查 |
| `galp/data/` | 小型 canonical 数据、profile 配置或测试资源；大结果不应放入源码 |

`GALP_BUILD_BENCHMARKS` 默认关闭，benchmark targets 只有显式启用时才参与构建。核心 runtime 不链接 benchmark 或 experiment。

当前仍可能有测试/实验脚本复用 benchmark 的 workload helper。这种依赖只应存在于证据层，不代表 benchmark 是 production runtime 的 owner。

## 4. API 分层

### Stable

面向普通用户，尽量只暴露语义概念：

- `DirectDctProfile`
- `DirectDctReader`
- `DirectDctPipeline`
- `DirectDctBatch`
- `DirectDctMetrics`
- `record_stream` 合同
- C++ `Reader` / `Table`

### Advanced

面向需要直接控制 Native runtime 或集成 C++ 系统的用户：

- `<galp/advanced/direct_dct.hpp>`
- `<galp/advanced/direct_dct_pls.hpp>`

这些接口可能暴露 CUDA stream、device descriptor 或执行选项，需要调用者理解生命周期和设备语义。

### Experimental

当前主要是 Python PLS：

```python
from galp.torch.experimental import DirectDctPlsPipeline
```

它可以使用，但兼容性承诺低于 Stable API，且不会从 `galp.torch` 的 stable `__all__` 自动导出。

### Diagnostics

只用于观测、调试和验证，不应作为业务控制流的 source of truth。

### Internal

以下概念属于内部实现，不应从 Stable API 暴露：

- raw CUDA event；
- device batch 和 workset detail；
- rowgroup/shard/segment implementation；
- submission gate；
- allocator/cache internals；
- rollback selector。

## 5. Python Stable API

`galp.torch` 当前稳定导出项只有：

```python
from galp.torch import (
    DirectDctBatch,
    DirectDctMetrics,
    DirectDctPipeline,
    DirectDctReader,
)
```

### 5.1 `DirectDctReader`

构造：

```python
reader = DirectDctReader(
    manifest_path,
    module_path=None,
    native_module=None,
)
```

参数含义：

- `manifest_path`：Direct-DCT manifest 文件；
- `module_path`：开发模式下扩展模块所在目录。正式安装 wheel 后通常不需要传；
- `native_module`：测试或嵌入场景注入 native binding，一般用户不使用。

主要接口：

| 接口 | 说明 |
| --- | --- |
| `image_count` | manifest 中的图片总数 |
| `profile_info(profile)` | 查询 profile 的已解析信息 |
| `read(image_ids, profile, coefficients=None, transforms=None)` | 一次性读取一个 logical batch；`None` 表示全部系数 |
| `iter_batches(logical_batches, profile=..., coefficients=None)` | 普通用户推荐的连续 batch 入口 |
| `pipeline(profile, coefficients=None)` | 需要显式 reset/close/累计 metrics 时创建 Native pipeline |

`dct_coeffs="all"`、`"first:N"` 和 `"list:..."` 继续作为兼容参数支持；
`module_path` 与 `native_module` 是开发/测试兼容入口，不是安装环境中的推荐参数。

### 5.2 `DirectDctPipeline`

主要接口：

| 接口 | 说明 |
| --- | --- |
| `start(image_id_batches, transforms_by_batch=None)` | 提交一个 batch 序列，并返回自身 |
| `__iter__()` / `__next__()` | 逐批取得 `DirectDctBatch` |
| `metrics` | 当前 pipeline 聚合指标 |
| `close()` | 停止 pipeline 并释放资源 |
| context manager | 推荐使用 `with` 确保异常路径也关闭 |

`start()` 接收的是 logical batch 列表。例如 `[[0, 1], [2, 3]]` 表示两个 batch，而不是一个包含四张图的 batch。

### 5.3 `DirectDctBatch`

主要属性：

| 属性/方法 | 说明 |
| --- | --- |
| `y` | Y 分量 Tensor；是否存在由 profile 输出合同决定 |
| `cbcr` | CbCr 分量 Tensor；是否存在由 profile 输出合同决定 |
| `coefficients` | compact coefficient 输出；是否存在由 profile 决定 |
| `tensors` | 固定返回 `(y, cbcr)`；compact coefficient 输出需读取 `coefficients` |
| `global_image_ids` | 输出对应的全局 sample ID，顺序与请求合同一致 |
| `sample_ids` | `global_image_ids` 的只读、无底层数据复制别名 |
| `transform_descriptors` | profile/transform 对应的描述信息 |
| `layout` | 当前输出布局标识 |
| `metrics` | 本 batch 指标快照 |
| `record_stream(stream=None)` | 注册非默认 consumer stream |

### 5.4 `DirectDctMetrics`

稳定字段包括：

- `complete`
- `consumer_wait_ms`
- `submit_to_ready_ms`
- `producer_ms`
- `planning_ms`
- `io_ms`
- `decode_ms`
- `transform_ms`
- `logical_bytes`
- `physical_bytes`
- `peak_transient_bytes`

读取 metrics 不会为了等待 GPU timing 而主动同步设备。若底层 event 尚未完成，快照可以是 unfinished；若第一次读取前 GPU 已完成，也可以直接 finalized。正确合同是“状态与真实 completion 一致”，而不是强制外部观察一次 `false -> true`。

## 6. Coefficient selection

Direct-DCT 使用统一 coefficient selection。通用合法合同为：

- 数量为 1 到 64；
- 每个 index 位于 `[0, 64)`；
- index 不重复；
- 显式空集合非法；
- 调用者给出的顺序会成为 selection 的语义顺序；
- 全部系数等价于 `[0, 1, ..., 63]`。

Python 推荐写法：

```python
coefficients=None
coefficients=range(32)
coefficients=[5, 0, 2]
```

调用者给出的显式顺序保持不变，不排序也不去重；重复、空集合和越界 index 会被拒绝。
旧 `dct_coeffs` 字符串仍会归一化到同一 Native selection，不会选择另一套 executor。

某些 semantic profile 可能由于自身变换语义要求完整输入；这种限制应在 profile/policy 边界明确报错，而不是让底层 executor 悄悄解码 64 个系数后丢弃。

## 7. Python 使用示例

### 7.1 安装后的最小读取

```python
from galp.profiles.rgbnomore import VALIDATION
from galp.torch import DirectDctReader

reader = DirectDctReader("/data/imagenet-dct/manifest.bin")

batch = reader.read(
    image_ids=[0, 1, 2, 3],
    profile=VALIDATION,
    coefficients=None,
)

y = batch.y
cbcr = batch.cbcr
print(batch.sample_ids)
print(y.shape, y.dtype, y.stride())
print(cbcr.shape, cbcr.dtype, cbcr.stride())
```

具体 shape 由 profile、图像几何和 transform 决定，不应在应用中假设所有 profile 都返回相同布局。

### 7.2 任意非连续系数

```python
from galp.profiles.rgbnomore import VALIDATION
from galp.torch import DirectDctReader

reader = DirectDctReader("/data/imagenet-dct/manifest.bin")
batch = reader.read(
    [999, 1000, 1001],
    VALIDATION,
    coefficients=[5, 0, 2],
)

assert batch.sample_ids == [999, 1000, 1001]
```

如果所选 profile 要求完整 transformed grid，profile resolver 会明确拒绝不兼容的 selection；不要在调用端用“先解全部再切片”绕过合同。

### 7.3 连续 batches：普通用户推荐入口

```python
from galp.profiles.rgbnomore import VALIDATION
from galp.torch import DirectDctReader

reader = DirectDctReader("/data/imagenet-dct/manifest.bin")
logical_batches = [
    list(range(0, 64)),
    list(range(64, 128)),
    list(range(128, 192)),
]

with reader.iter_batches(
    logical_batches,
    profile=VALIDATION,
    coefficients=range(32),
) as batches:
    for batch in batches:
        outputs = model(batch.y, batch.cbcr)
```

`iter_batches()` 只是现有 `pipeline()`、`start()` 和 iteration 的薄包装；不会创建
Python producer、queue、Future 或新的 Native submission protocol。

### 7.4 显式控制 Native pipeline

需要复用 pipeline、显式 reset/close 或读取累计 `pipeline.metrics` 时，可以继续使用高级
Stable 入口：

```python
with reader.pipeline(VALIDATION, coefficients=range(32)) as pipeline:
    pipeline.start(logical_batches)

    for batch in pipeline:
        outputs = model(batch.y, batch.cbcr)

    # 如应用本来就需要在此等待，可以在自然边界同步后读取最终指标。
    # 不要为了每批 metrics 在循环中 synchronize。
    final_metrics = pipeline.metrics
    print(final_metrics)
```

调度、prefetch、跨 shard 分段和 output assembly 都在 Native pipeline 内完成。应用不需要自己建立 Future 队列或拼接跨 shard Tensor。

### 7.5 在额外 CUDA stream 上消费

如果模型工作提交到取得 batch 时的当前 stream，通常不需要额外操作。若 Tensor 会在另一个 CUDA stream 上使用，必须显式注册：

```python
import torch

side_stream = torch.cuda.Stream()

batch = reader.read([0, 1, 2, 3], VALIDATION)
batch.record_stream(side_stream)

with torch.cuda.stream(side_stream):
    outputs = model(batch.y, batch.cbcr)
```

`record_stream` 应在向 consumer stream 提交工作前调用。Native lifetime 会等待 producer 完成、所有已注册 consumer 完成以及 Storage owner 释放，然后才允许回收 backing storage。

该合同不会自动保护“释放所有 Tensor 引用之后才新提交”的未来工作。调用者必须在对象仍有效时正确登记实际 consumer stream。

### 7.6 PLS Experimental API

PLS 不从 stable `galp.torch` 自动导出，必须明确从 experimental 模块导入：

```python
from galp.torch.experimental import DirectDctPlsPipeline

with DirectDctPlsPipeline(
    manifest_path="/data/train/manifest.bin",
    premixed_mapping_csv="/data/train/premixed.csv",
    training_seed=11997733,
    expected_mapping_sha256="<expected-sha256>",
    segments_per_pool=4,
    microbatch_images=64,
) as pipeline:
    pipeline.start_epoch(0)

    while pipeline.has_next_pool:
        pool = pipeline.next_pool()
        try:
            for microbatch in pool:
                logits = model(microbatch.y, microbatch.cbcr)
                loss = criterion(logits, microbatch.targets)
                loss.backward()
        finally:
            pool.retire()
```

常用 PLS 类型：

- `DirectDctPlsPipeline`：epoch 和 pool 级控制；
- `DirectDctPlsPool`：一个有界 pool 的 microbatch 视图；
- `DirectDctPlsMicrobatch`：Y/CbCr/targets/sample IDs 和 `record_stream`。

PLS 内部采用一个 active pool 加一个 preparing/ready pool 的单步前瞻。Python 不负责创建 worker、切换 postprocess stream 或决定 Native backing 的回收时机。

## 8. Python 包安装与开发模式

### 8.1 构建 wheel

仓库根目录的 `pyproject.toml` 使用 scikit-build-core。一个典型构建命令是：

```bash
python -m pip wheel . \
  --no-build-isolation \
  --wheel-dir dist \
  --config-settings=cmake.define.FLS_BUILD_GALP=ON \
  --config-settings=cmake.define.GALP_BUILD_TORCH=ON \
  --config-settings=cmake.define.GALP_INSTALL_TORCH_PACKAGE=ON \
  --config-settings=cmake.define.FLS_ENABLE_INSTALL=OFF \
  --config-settings=cmake.define.GALP_ENABLE_INSTALL=OFF
```

安装：

```bash
python -m pip install dist/pyfastlanes-*.whl
```

安装产物包含：

- `galp` Python package；
- `galp.torch` Stable facade；
- `galp.torch.experimental`；
- profiles 和 diagnostics 模块；
- package-local `_galp_direct_dct` 扩展；
- 扩展运行所需的共享库解析设置。

benchmarks、tests、runs 和历史诊断脚本不是 stable wheel 的默认内容。

### 8.2 验证正式安装

应在不依赖源码目录或 build-tree `PYTHONPATH` 的新环境中检查：

```bash
python -c "import galp; import galp.torch; import galp.torch.experimental; import galp.diagnostics.direct_dct"
```

### 8.3 开发模式构建

开发扩展时可使用独立 build 目录：

```bash
cmake -S . -B build-galp-torch -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
  -DGALP_BUILD_TORCH=ON \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

cmake --build build-galp-torch --target _galp_direct_dct -j
```

然后可以显式传入：

```python
reader = DirectDctReader(
    "/data/manifest.bin",
    module_path="build-galp-torch/galp/torch",
)
```

`module_path` 是开发和测试便利入口，不应成为部署环境的长期依赖。

## 9. C++ Stable Table API

通用 Table 路径的最小示例：

```cpp
#include <cstddef>
#include <cstdint>
#include <galp/stable.hpp>

int main() {
  galp::Reader reader("/data/table.fls");

  galp::DecompressOptions options;
  options.write_output = true;

  galp::Table table = reader.decompress(options);

  for (std::size_t rowgroup_index = 0;
       rowgroup_index < table.rowgroup_count();
       ++rowgroup_index) {
    const auto rowgroup = table.rowgroup(rowgroup_index);
    for (std::size_t column_index = 0;
         column_index < rowgroup.column_count();
         ++column_index) {
      const auto column = rowgroup.column(column_index);
      if (column.type() == galp::DataType::I16) {
        const auto values = column.values<int16_t>();
        // 使用 values；该 view 的生命周期不能超过 table。
      }
    }
  }
}
```

当前 Stable materialized Table 类型面与 storage 能力保持一致，主要是 `int8_t` 和 `int16_t`。不受支持的类型应在 API 边界明确拒绝，而不是让调用者在深层遇到 `bad_variant_access`。

当只需要元数据或执行计划、不需要写出列数据时，可以使用：

```cpp
galp::DecompressOptions options;
options.write_output = false;
```

这表示执行路径应避免物化输出，不等同于“解压后再丢弃结果”。

## 10. 生命周期和 stream 语义

这是使用 Direct-DCT 时最容易误用、也最重要的合同。

### Same-stream

模型在取得 Tensor 的当前 CUDA stream 上继续执行时，正常 Tensor/Storage 生命周期即可保护 backing storage，一般无需额外调用。

### Cross-stream

模型或后处理在另一个 stream 上执行时，调用：

```python
batch.record_stream(actual_consumer_stream)
```

PLS microbatch 也提供同样的方法。

### Reclaim 条件

同一份 backing storage 只有在以下条件全部满足后才能复用：

```text
producer 已完成
AND
所有已注册 consumer 已完成
AND
Batch/Tensor Storage ownership 已释放
```

producer event、consumer event 和 reclaim 权限由 Native lifetime 对象统一管理。Rollback lifetime 只在构造时选择，不应与 Native lifetime 在同一 production batch 上同时执行。

## 11. Metrics 的正确使用

Metrics 用于观测，不应改变执行时序。

推荐方式：

```python
snapshot = pipeline.metrics

if snapshot.complete:
    print(snapshot.decode_ms, snapshot.transform_ms)
else:
    # GPU timing 还没自然完成；稍后在已有同步边界之后再读。
    pass
```

不要为了让 `complete` 立即为真而在每个 batch 后执行 `torch.cuda.synchronize()`。如果应用在 epoch、checkpoint 或最终结果读取处本来就需要同步，可以在那个自然边界之后读取最终 metrics。

## 12. 什么代码不属于 production core

### Benchmarks

`galp/benchmarks/` 可以：

- 组织训练或推理 workload；
- 定义公平的 A/B contract；
- 记录 commit、GPU、profile、manifest、seed 和指标；
- 生成报告。

它不应：

- 实现生产 scheduler 或 lifetime；
- 成为 Stable API 的 import 依赖；
- 把一次实验结论硬编码成 runtime policy。

### Experiments

`galp/benchmarks/` 可以验证新想法，但不能反向污染 production API。实验稳定并产品化后，应把真正通用的合同和实现迁入相应的 `include/src/torch` 层，而不是让正式调用者永久 import 实验目录。

### Generated artifacts

runs、checkpoint、JSONL、NPY/NPZ、Nsight 报告和长日志是运行产物，不属于库源码。它们应位于被精确忽略的输出目录或仓库外部存储。

## 13. 常见误用

### 误用 1：直接 import private extension

不推荐：

```python
import _galp_direct_dct
```

推荐：

```python
from galp.torch import DirectDctReader
```

Stable facade 会检查 binding、profile 和 metrics schema，并隔离内部类型变化。

### 误用 2：在 Python 中重新做跨 shard 拼接

不要在 production path 中自己 `cat`、`clone` 或 synchronize 多个 shard 输出。Native physical orchestration 已负责 logical output 的分段和 assembly。

### 误用 3：跨 stream 使用但不登记

如果实际 consumer stream 与当前 stream 不同，必须调用 `record_stream`，否则 allocator 无法知道该 stream 上仍有未完成工作。

### 误用 4：把 PLS 当成 Stable API

PLS 应从 `galp.torch.experimental` 或 Advanced C++ header 使用。它目前可以用于受控训练系统，但升级或兼容策略与 Stable facade 不同。

### 误用 5：依赖 build-tree `PYTHONPATH` 部署

`PYTHONPATH=build/galp/torch` 适合开发 smoke，不是正式安装方式。发布和部署应使用 wheel/install 产物。

## 14. 如何选择入口

| 需求 | 推荐入口 |
| --- | --- |
| 解压普通 FLS 表 | C++ `<galp/stable.hpp>` 中的 `Reader` / `Table` |
| Python 模型读取 Direct-DCT | `galp.torch.DirectDctReader.read(...)` |
| 连续 logical batches 和 Native prefetch | 普通调用用 `iter_batches(...)`；显式生命周期/累计 metrics 用 `pipeline(...)` |
| 跨 CUDA stream 消费 | `DirectDctBatch.record_stream(...)` |
| PLS 训练研究 | `galp.torch.experimental.DirectDctPlsPipeline` |
| C++ 深度集成 Direct-DCT | `<galp/advanced/direct_dct.hpp>` |
| 调试 planner、cache 或 metrics | `galp.diagnostics.direct_dct` |
| 做性能对比或科学证据 | `galp/benchmarks/` 中对应 contract/runner |
| 验证新研究想法 | `galp/benchmarks/` 中独立 package |

## 15. 维护者快速检查清单

修改 GALP 时，建议先确认：

```text
是否仍只有一个 scheduler owner？
是否仍只有一个 lifetime/reclaim owner？
是否仍只有一个 metrics reducer？
是否仍只有一个 production physical owner？
Python 是否仍是薄适配层？
是否复用了已有 planner/workset/kernel？
是否引入了新的 synchronize、D2H 或无界队列？
coefficient selection 是否仍使用统一合同？
PLS 是否仍位于 Advanced/Experimental 边界？
benchmark/experiment 是否反向成为 runtime 依赖？
```

如果答案出现异常，应先修正所有权和依赖方向，再考虑局部性能优化。

## 16. Python 可导入符号总表

这一节只列当前库面向调用者的导入项。benchmark 和 experiment 中虽然还有很多 Python 函数，但它们属于具体 workload 的实现，不是 GALP 库 API。

### 16.1 Stable 顶层：`galp`

```python
from galp import DirectDctProfile
```

| 符号 | 构造/调用 | 说明 |
| --- | --- | --- |
| `DirectDctProfile` | `DirectDctProfile(id: str)` | 不可变的语义 profile ID；不包含 cache、stream 或 launch 调优参数 |

`DirectDctProfile` 唯一公开字段是 `id`。空 ID 会在构造时拒绝。

### 16.2 Stable Torch：`galp.torch`

```python
from galp.torch import (
    DirectDctBatch,
    DirectDctMetrics,
    DirectDctPipeline,
    DirectDctReader,
)
```

这是 `galp.torch.__all__` 的完整内容，没有隐藏的第五个 stable 类型。

#### `DirectDctReader`

```python
DirectDctReader(
    manifest_path: str | pathlib.Path,
    *,
    module_path: str | pathlib.Path | None = None,
    native_module: object | None = None,
)
```

| 成员 | 签名/类型 | 说明 |
| --- | --- | --- |
| `image_count` | property → `int` | manifest 中的图片数量 |
| `profile_info` | `(profile: DirectDctProfile | str) -> dict` | 解析并校验 profile 元数据 |
| `pipeline` | `(profile, *, coefficients=None, dct_coeffs=<compat>) -> DirectDctPipeline` | 创建可复用的 Native pipeline |
| `read` | `(image_ids, profile, *, coefficients=None, dct_coeffs=<compat>, transforms=None) -> DirectDctBatch` | 一次性读取一个 logical batch |
| `iter_batches` | `(logical_batches, *, profile, coefficients=None, dct_coeffs=<compat>, transforms_by_batch=None)` | 复用现有 Pipeline 的便捷迭代入口 |

`native_module` 是测试注入点，`module_path` 是 build-tree 开发入口。正式安装环境通常只传 `manifest_path`。

#### `DirectDctPipeline`

| 成员 | 签名/类型 | 说明 |
| --- | --- | --- |
| `start` | `(image_id_batches, *, transforms_by_batch=None) -> DirectDctPipeline` | 用一组 logical batches 重置并启动 pipeline |
| `__iter__` | `() -> DirectDctPipeline` | 返回迭代器自身 |
| `__next__` | `() -> DirectDctBatch` | 取得下一批；调度、等待和交付由 Native 层完成 |
| `metrics` | property → `DirectDctMetrics` | 非阻塞累计快照 |
| `close` | `() -> None` | 关闭 pipeline，异常路径也应调用 |
| `__enter__` / `__exit__` | context manager | 推荐使用 `with` |

#### `DirectDctBatch`

| 成员 | 类型/签名 | 说明 |
| --- | --- | --- |
| `profile_id` | `str` | 实际使用的 profile ID |
| `y` | Tensor | Y 输出 |
| `cbcr` | Tensor | CbCr 输出 |
| `coefficients` | Tensor/Native output | compact coefficient 输出；由 profile 决定是否有意义 |
| `tensors` | property → `(y, cbcr)` | 便于模型调用的固定二元组 |
| `global_image_ids` | `list[int]` | 输出 sample 顺序 |
| `sample_ids` | `list[int]` | `global_image_ids` 的只读别名 |
| `transform_descriptors` | `list[dict]` | 每个 sample 的变换描述 |
| `layout` | `str` | Native 输出布局名称 |
| `metrics` | `DirectDctMetrics` | 当前 batch 非阻塞指标快照 |
| `record_stream` | `(stream: torch.cuda.Stream | None = None) -> None` | 登记真实 consumer stream；不做 host synchronize |

无参数的 `record_stream()` 登记当前 PyTorch CUDA stream；显式参数用于调用代码当前不在目标 stream 上下文内的情况。

#### `DirectDctMetrics`

`DirectDctMetrics` 是不可变 dataclass，没有额外动作函数。完整字段如下：

```text
complete
consumer_wait_ms
submit_to_ready_ms
producer_ms
planning_ms
io_ms
decode_ms
transform_ms
logical_bytes
physical_bytes
peak_transient_bytes
```

### 16.3 Profile 常量：`galp.profiles.rgbnomore`

```python
from galp.profiles.rgbnomore import (
    TRAINING_PLS,
    VALIDATION,
    VALIDATION_CENTER_CROP_512,
)
```

| 常量 | Profile ID | 用途 |
| --- | --- | --- |
| `VALIDATION` | `rgbnomore-validation-v1` | 常规 RGB-no-more DCT 验证/推理 |
| `VALIDATION_CENTER_CROP_512` | `rgbnomore-validation-center-crop-512-v1` | 512 源图中心裁剪验证 |
| `TRAINING_PLS` | `rgbnomore-training-pls-v1` | PLS 训练；API 级别仍是 Experimental/Advanced |

`galp.profiles` 自身只导出 `DirectDctProfile`，因此预定义常量要从 `galp.profiles.rgbnomore` 明确导入。

### 16.4 Experimental PLS：`galp.torch.experimental`

```python
from galp.torch.experimental import (
    DirectDctPlsMicrobatch,
    DirectDctPlsPipeline,
    DirectDctPlsPool,
)
```

#### `DirectDctPlsPipeline`

```python
DirectDctPlsPipeline(
    manifest_path,
    premixed_mapping_csv,
    *,
    training_seed,
    expected_mapping_sha256,
    crop_policy="per-pls",
    order_policy="closed-pool",
    segments_per_pool=4,
    microbatch_images=64,
    segment_images=1024,
    model_classes=1000,
    profile="rgbnomore-training-pls-v1",
    module_path=None,
    native_module=None,
)
```

| 成员 | 签名/类型 | 说明 |
| --- | --- | --- |
| `start_epoch` | `(epoch: int) -> DirectDctPlsPipeline` | 建立该 epoch 的 Native pool schedule |
| `next_pool` | `() -> DirectDctPlsPool` | 取得下一个已准备或正在完成的 pool |
| `__iter__` | `() -> DirectDctPlsPipeline` | 返回 pipeline 自身 |
| `__next__` | `() -> DirectDctPlsMicrobatch` | 兼容性的扁平 microbatch 迭代 |
| `close` | `() -> None` | 关闭 worker、stream 和 Native runtime |
| `reclaim_finished_pools` | `() -> int` | 兼容性回收触发器；Native lease 仍是 authority |
| `sample_count` | property → `int` | 当前 epoch 的 sample 数 |
| `has_next_pool` | property → `bool` | 是否还有 pool |
| `pls_count` | property → `int` | virtual PLS 数量 |
| `segment_images` | property → `int` | 每 physical segment 的图片数 |
| `prefetch_stats` | property → `dict` | one-pool lookahead 统计 |

#### `DirectDctPlsPool`

| 成员 | 签名/类型 | 说明 |
| --- | --- | --- |
| `microbatch` | `(index: int) -> DirectDctPlsMicrobatch` | 按 pool 内索引取 microbatch |
| `__iter__` / `__next__` | iterator | 顺序消费 pool 内 microbatches |
| `retire` | `() -> None` | 退休调度上下文；Tensor backing 仍由 Native lifetime 独立保护 |
| `epoch` | `int` | epoch 编号 |
| `pool_index` | `int` | pool 编号 |
| `image_count` | `int` | pool 中图片数 |
| `microbatch_count` | `int` | pool 中 microbatch 数 |
| `virtual_pls_ids` | `list[int]` | pool 覆盖的 virtual PLS |
| `execution_stats` | `dict` | pool 执行统计 |

#### `DirectDctPlsMicrobatch`

| 成员 | 类型/签名 | 说明 |
| --- | --- | --- |
| `y` / `cbcr` / `targets` | Tensor | 模型输入和训练目标 |
| `tensors` | `(y, cbcr, targets)` | 三个 Tensor 的固定元组 |
| `record_stream` | `(stream=None) -> None` | 与 Stable batch 相同的 consumer-stream 合同 |
| `epoch` | `int` | epoch 编号 |
| `pool_index` | `int` | pool 编号 |
| `microbatch_index_in_pool` | `int` | pool 内 microbatch 索引 |
| `pool_offset` | `int` | microbatch 在 pool 中的图片偏移 |
| `image_count` | `int` | 本 microbatch 图片数 |
| `is_pool_end` | `bool` | 是否为 pool 最后一批 |
| `global_image_ids` | `list[int]` | 全局 sample ID |
| `labels` | `list[int]` | CPU 可见标签列表 |

### 16.5 Diagnostics：`galp.diagnostics.direct_dct`

`galp.diagnostics` 的顶层 `__init__` 故意不导出任何名字。必须显式导入：

```python
from galp.diagnostics.direct_dct import (
    aggregate_metric_snapshots,
    binding_import_ms,
    cache_stats,
    execution_stats,
    execution_stats_observation,
    execution_stats_snapshot,
    image_metadata,
    initialization_stats,
    metric_descriptors,
    pipeline_stats,
    plan_preview,
    rowgroup_storage_bytes,
)
```

| 函数 | 签名 | 是否可能等待 |
| --- | --- | --- |
| `execution_stats` | `(batch) -> dict` | 是；需要时等待 batch completion |
| `execution_stats_snapshot` | `(batch) -> dict` | 否；只返回当前已有计数 |
| `execution_stats_observation` | `(batch) -> dict` | 否；同时给出 host/GPU completion 状态 |
| `metric_descriptors` | `(reader) -> list[dict]` | 否 |
| `aggregate_metric_snapshots` | `(reader, snapshots) -> dict` | 否；调用 Native canonical reducer |
| `cache_stats` | `(batch) -> dict` | 读取实现级 cache 统计 |
| `pipeline_stats` | `(pipeline) -> dict` | 读取 ready/started/prefetch 等不稳定状态 |
| `initialization_stats` | `(reader) -> dict` | reader 初始化诊断 |
| `binding_import_ms` | `(reader) -> float` | 扩展导入耗时 |
| `plan_preview` | `(reader, image_ids, profile, *, transforms=None) -> dict` | 只做诊断性 planning preview |
| `image_metadata` | `(reader, global_image_index) -> dict` | 读取私有 JPEG component metadata |
| `rowgroup_storage_bytes` | `(reader, shard_id, rowgroup_indices) -> int` | 计算物理 rowgroup bytes |

不要在训练热路径中调用会等待 completion 的 `execution_stats`。普通程序应优先使用 Stable `batch.metrics` / `pipeline.metrics`。

### 16.6 模块级支持常量

下面两个常量存在于 `galp.torch.direct_dct` 的模块级 `__all__`，主要用于 binding/schema 检查，不从 `galp.torch` stable umbrella 导出：

```python
from galp.torch.direct_dct import METRICS_SCHEMA, PROFILE_SCHEMA
```

应用通常不需要导入它们。`BINDING_SCHEMA` 和以下划线开头的函数属于内部实现。

## 17. 当前性能证据应该怎样理解

“性能最佳”必须先说明比较范围：

1. 最快的 GALP 训练配置；
2. 最快的 GALP 推理配置；
3. 所有 backend 中的绝对最快对照。

它们不是同一个问题，也不能跨 workload 比较绝对 img/s。

当前仓库 HEAD 为 `31c9fd0a12971bab219e90d3fa8e12ba17004f68`。训练性能提交已经进入该 HEAD；最佳推理证据来自其祖先 `cd56bb360bf795dbb877fa687534743b7520ea94`，后续提交没有改动 Stable inference hot path。

### 17.1 推荐的 GALP 训练配置

推荐基线是 B6 Native Physical PLS：

| 配置项 | 当前推荐值 | 原因 |
| --- | --- | --- |
| condition | `B6` | per-PLS crop + closed-pool order 的目标策略 |
| execution backend | `native-physical-pls` | 使用 Native physical execution 和 PLS CUDA 后处理 |
| semantic profile | `rgbnomore-training-pls-v1` | 当前注册的训练语义 |
| segment size | 1024 images | 冻结 physical layout 合同 |
| segments per pool | 4 | B6 的闭合 pool，约 4096 images/pool |
| physical microbatch | 64 | recipe 固定值 |
| gradient accumulation | 16 | effective batch = 1024 |
| precision | FP32 | published recipe 固定值 |
| model | RGB-no-more ViT-Ti DCT | 12 layers、192 hidden、3 heads |
| compile | `torch.compile`, Inductor, fixed shape | 当前执行 recipe |
| audit mode | `runtime-first-100` | 全局前 100 optimizer updates 严格审计，之后 device accumulation + epoch boundary readback |
| workers | 4 | 当前正式训练 runner 配置 |
| CLI prefetch depth | 2 | 保留的 runner 参数；不要把它误解为两个 next pools |
| Native pool lookahead | 1 active + 1 preparing/ready | 内部固定有界设计，最大 context 数为 2 |
| seed | `11997733` | 已验证基准 seed；科学矩阵另有四个配对 seeds |
| GPU | RTX 4090 | 当前性能证据平台 |

训练数学合同还包括：

```text
epochs                         300
AdamW learning rate            3e-3
betas                           (0.9, 0.999)
gradient clipping               1.0
warmup                          10,000 optimizer updates
schedule                        epoch-aware cosine
RGB-no-more independent WD      1e-4
DCT Mixup alpha                 0.2
RandAugment                     2 operations, magnitude 3
```

已提交持续性证据为 warm Epoch 2：

```text
throughput                      2246.38 img/s
pool prepare hidden ratio       99.6416%
exposed pool time               0.4636% of epoch
ready before activation         312 / 313 pools
only miss                       pool 0 startup
maximum live pool contexts      2
normal boundary device sync     0
duplicate reads / inversions    0 / 0
```

另一个 2026-09-01 scoped-dirty fairness run 在同一 HEAD 基线上测得 warm Epoch 2 `2290.31 img/s`，但其训练 source tree 含未提交 benchmark WIP。它可以作为最新观测，不能替代 `2246.38 img/s` 的已提交推荐基线。

### 17.2 推荐训练命令

先准备环境和路径：

```bash
cd /home/tangyuxin/gfastlanes/FastLanes

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
export PYTHONPATH="$PWD/galp/benchmarks/system_dct_major"

export TRAIN_JSON="$PWD/galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/train.json"
export VAL_JSON="$PWD/galp/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json"
export LAYOUT=/mnt/nvme2/home/tangyuxin/pls-experiments/pls-layout-20260811/physical_layout_plan.json
export PHYSICAL_MANIFEST=/home/tangyuxin/gfastlanes/FastLanes/galp/data/compressed/imagenet512_train_block_major_premixed/dct/manifest.bin
export MAPPING=/home/tangyuxin/gfastlanes/FastLanes/galp/data/compressed/imagenet512_train_block_major_premixed/ordered_mapping.csv
export MAPPING_SHA256=98f77515e5886c098e46c23cddb41f57098b406ab32790509254c56356dc24ff
export OUT=/mnt/nvme2/home/tangyuxin/pls-experiments/native-b6-current
```

运行 B6：

```bash
"$PY" -m training_pls.run_matrix \
  --output-dir "$OUT" \
  --train-manifest "$TRAIN_JSON" \
  --val-manifest "$VAL_JSON" \
  --layout-plan "$LAYOUT" \
  --conditions B6 \
  --seeds 11997733 \
  --epochs 300 \
  --device cuda:0 \
  --required-gpu-name-substring "RTX 4090" \
  --audit-mode runtime-first-100 \
  --workers 4 \
  --prefetch-depth 2 \
  --galp-torch-module-path "$PWD/build/galp/torch" \
  --execution-backend native-physical-pls \
  --physical-galp-manifest "$PHYSICAL_MANIFEST" \
  --premixed-mapping-csv "$MAPPING" \
  --expected-mapping-sha256 "$MAPPING_SHA256" \
  --execute
```

如只做两轮 operational/performance 验证，可额外加入：

```text
--stop-after-epoch 2
```

这不会把 300-epoch scientific recipe 改成 2 epochs，而是在完整 epoch 边界保存 checkpoint 后暂停。正式科学训练不要加入该参数。

### 17.3 最小 PLS 训练代码

benchmark runner 包含 published model、augmentation、optimizer、checkpoint 和 provenance。若只想集成数据 API，可以采用下列骨架：

```python
from galp.profiles.rgbnomore import TRAINING_PLS
from galp.torch.experimental import DirectDctPlsPipeline

pipeline = DirectDctPlsPipeline(
    manifest_path=physical_manifest,
    premixed_mapping_csv=mapping_csv,
    training_seed=11997733,
    expected_mapping_sha256=mapping_sha256,
    crop_policy="per-pls",
    order_policy="closed-pool",
    segments_per_pool=4,
    microbatch_images=64,
    segment_images=1024,
    profile=TRAINING_PLS,
)

with pipeline:
    pipeline.start_epoch(epoch)
    while pipeline.has_next_pool:
        pool = pipeline.next_pool()
        try:
            for microbatch in pool:
                logits = model(microbatch.y, microbatch.cbcr)
                loss = criterion(logits, microbatch.targets)
                (loss / 16).backward()
                # 每 16 个 microbatches 执行 clip/optimizer/scheduler。
        finally:
            pool.retire()
```

实际发布训练应继续使用 `training_pls.run_matrix`，因为它还会冻结 recipe、manifest hash、sample order、audit cursor 和 checkpoint compatibility。

## 18. 当前最佳推理配置与例子

### 18.1 GALP 最佳已测配置

当前最佳已测 GALP inference 配置来自 2026-08-30 rerun：

| 配置项 | 值 |
| --- | --- |
| preset | `e2e` |
| batch size | 50 |
| measured batches | 1000（50,000 images） |
| repeats | 5；聚合时排除 repeat 0 |
| workers | 8 |
| precision | FP32 |
| device | `cuda:0`, RTX 4090 |
| seed | 11997733 |
| model stream priority | greatest |
| semantic profile | `rgbnomore-validation-v1` |
| preprocess | `rgbnomore-val-pushdown` |
| storage | compact-v3 tiled-z32 RGB-no-more 512 manifest |
| runtime profile | `compact-v3-planless-limited-o512-c512-v1` |
| physical owner | Native |
| internal sync per batch | 0 |

同轮 hot median：

| Pipeline | Throughput | Mean latency | Top-1 | 解释 |
| --- | ---: | ---: | ---: | --- |
| GALP | 4690.37 img/s | 10.660 ms/batch | 75.14% | 最快 GALP 配置 |
| DALI | 4787.04 img/s | 10.445 ms/batch | 74.08% | 所有 backend 的绝对最快，对应 RGB 模型/预处理域 |
| RGB-no-more | 1821.10 img/s | 27.456 ms/batch | 75.14% | 与 GALP 同 DCT 语义 |
| PyTorch | 1622.12 img/s | 30.825 ms/batch | 74.10% | RGB reference |

GALP 是同轮 DALI 的 `97.98%`，且 GALP 与 RGB-no-more 的 50,000 张完整预测一致。DALI/PyTorch 属于 RGB 域，不能把其 Top-1 和 DCT checkpoint 做逐 tensor 等价比较。

这次 report 的总 `Validation` 仍为 FAIL，原因是旧 gate 要求 GALP 至少达到 DALI 的 `1.10x`，并且 cold repeat 0 的 planning p50/p95 略超旧阈值。它不否定 4690 img/s 的测量，也不代表该结果通过了正式性能 gate。因此文档称其为“最佳已测配置”，而不是“当前 HEAD release-certified inference”。

### 18.2 完整 E2E 推理命令

```bash
cd /home/tangyuxin/gfastlanes/FastLanes

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PY=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
export OUT=/mnt/nvme2/home/tangyuxin/pls-experiments/inference-e2e-current

PYTHONPATH="$PWD/build/galp/torch" "$PY" \
  galp/benchmarks/system_rgbnomore/inference/run.py \
  --preset e2e \
  --pipelines galp rgbnomore dali pytorch \
  --output-dir "$OUT" \
  --python "$PY" \
  --data-root "$PWD/galp/data/system_rgbnomore/e2e_v3/imagenet_512" \
  --index-csv "$PWD/galp/data/system_rgbnomore/e2e_v2/indexbase_val.csv" \
  --rgbnomore-root /home/tangyuxin/RGB-no-more \
  --rgb-checkpoint "$PWD/galp/data/system_rgbnomore/e2e_v2/checkpoints/imgnetRGBViTTi_ep300_74.1.pth" \
  --dct-checkpoint "$PWD/galp/data/system_rgbnomore/e2e_v2/checkpoints/imgnetDCTViTTi_ep300_75.1.pth" \
  --galp-manifest "$PWD/galp/data/compressed/imagenet512_val_compact_v3/manifest.bin" \
  --galp-label-map-json "$PWD/galp/data/compressed/imagenet512_val_compact_v3/labels.json" \
  --torch-binding-dir "$PWD/build/galp/torch" \
  --device cuda:0 \
  --precision fp32
```

`e2e` preset 强制同轮运行四条 pipeline，以保证 workload identity 和对照可信。若只是检查 API 是否工作，应使用 `--preset smoke`，不要用手工缩短 e2e 后仍把结果称为正式性能数据。

### 18.3 Stable API 推理骨架

```python
import torch

from galp.profiles.rgbnomore import VALIDATION
from galp.torch import DirectDctReader

reader = DirectDctReader("/data/compact-v3/manifest.bin")
batches = [
    list(range(begin, begin + 50))
    for begin in range(0, 1000, 50)
]

model.eval()
with torch.inference_mode(), reader.iter_batches(
    batches,
    profile=VALIDATION,
    coefficients=None,
) as direct_dct_batches:
    for batch in direct_dct_batches:
        logits = model(batch.y, batch.cbcr)
        predictions = logits.argmax(dim=-1)
```

需要累计 `pipeline.metrics` 时改用上一节的显式 `DirectDctPipeline`；两种写法进入完全相同
的 Native execution path。

若模型运行在单独 stream，先调用 `batch.record_stream(model_stream)`，再提交 forward。不要在每个 batch 后调用 `torch.cuda.synchronize()`；benchmark 只在其计时和结果合同要求的自然边界同步。

## 19. 当前推荐结论

如果目标是训练：

```text
B6
+ native-physical-pls
+ runtime-first-100 audit
+ microbatch 64 / accumulation 16
+ one-pool bounded lookahead
```

如果目标是 GALP 推理：

```text
VALIDATION profile
+ compact-v3 tiled-z32 manifest
+ Native DirectDctPipeline
+ batch 50
+ FP32
+ bounded lookahead / no per-batch synchronize
```

如果目标只是全系统最高推理吞吐且不要求 DCT-domain checkpoint/语义，当前同轮绝对最快是 DALI；如果要求 RGB-no-more DCT 语义和 75.14% Top-1，则当前最佳已测是 GALP。
