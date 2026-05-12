# G-ALP

G-ALP 是 FastLanes 的 GPU 解压路径。当前目录中的代码来自
`FastLanesGpu-Damon2025`，但已经被改造成 FastLanes 仓库内的库组件。

当前目标不是暴露完整 GPU 执行细节，而是提供一个很小的稳定公共 facade，
同时把 reader、execution、runtime、codec、memory、benchmark 等内部层次收敛到
`galp::*` 命名空间下。

## 当前状态

G-ALP 仍然是早期库组件。稳定范围很小：

- 支持的主路径：i8/i16 FastLanes rowgroup 的表级 GPU 解压。
- 公共 API：`galp/include/galp` 下的 `galp::Reader`、`galp::Table`、
  `galp::RowgroupView`、`galp::ColumnView`、`galp::DecompressOptions`。
- 内部实现：`engine/*`、`flsgpu/*`、generated bindings、benchmark headers、
  nvCOMP 相关代码都不是稳定 API。
- CUDA 是运行和 GPU 正确性测试的必需条件。没有 CUDA device 时，相关测试会 skip
  或 CLI 运行失败。

## 架构概览

旧版 G-ALP 更像一个内部 CUDA 解压代码集合：CLI、test、benchmark 直接 include
`reader`、`dispatch`、`flsgpu` 等内部头，`galp_core` 也把 `src/include` 作为
public include 暴露出去。

当前架构增加了公共 facade，并把内部模块按职责重新命名和收边界：

```text
external consumer
  -> galp::Reader / galp::decompress_table
  -> galp::execution::decompress_table
  -> galp::runtime::execute_table_pipeline
  -> galp::format::FlsReader
  -> galp::expression::assemble
  -> galp::memory::DeviceArena / pinned pools / H2D
  -> galp::kernels / galp::codec::device
  -> optional materialization
```

整体数据路径仍然是：

```text
FLS rowgroup
  -> zero-copy rowgroup view
  -> expression assembly
  -> execution workset
  -> DeviceArena/H2D upload
  -> CUDA kernel decode
  -> optional write-out and D2H materialization
```

## 模块划分

| 模块 | 路径 | 命名空间 | 职责 |
|---|---|---|---|
| Public facade | `include/galp`, `src/api` | `galp::*` | 稳定外部 API |
| Format/reader | `src/include/engine/reader.cuh`, `types.cuh`, `enums.cuh` | `galp::format` | FLS descriptor、rowgroup IO、zero-copy rowgroup |
| Expression | `src/include/engine/expression.cuh` | `galp::expression` | 把 rowgroup columns 组装成执行表达式 |
| Execution | `src/include/engine/execution/*` | `galp::execution` | 解压配置、数据模型、rowgroup/table decode 入口 |
| Runtime internals | `src/include/engine/execution/internal/*` | `galp::runtime` | streaming pipeline、prefetch、H2D、launch、materialize |
| GPU codec | `src/include/flsgpu/columns`, `src/include/flsgpu/fls` | `galp::codec::{host,device}` | host/device codec column 和 device 解压原语 |
| GPU memory | `src/include/flsgpu/memory` | `galp::memory` | CUDA RAII、DeviceArena、DevicePool、pinned host pool |
| Kernels | `src/include/engine/kernels.cuh`, dispatch headers | `galp::kernels` | CUDA kernel 和 host launch helper |
| CLI | `tools/galp_cli.cu` | internal | `read_table`、`benchmark`、launch measurement |
| Benchmarks | `benchmark` | `galp::bench` | generated bindings、microbenchmarks、nvCOMP 对比 |
| Codegen | `code-generators` | script | 生成 benchmark kernel bindings |
| Tests | `test` | mixed | public API smoke、reader/GPU/internal tests |

## 公共 API

外部实验和消费代码优先只 include facade：

```cpp
#include <galp/galp.hpp>

galp::Reader reader("/path/to/data.fls");

galp::DecompressOptions options;
options.write_output = true;
options.scope = galp::TableDecompressionScope::WholeTable;

galp::Table table = reader.decompress(options);
auto rowgroups = table.rowgroup_count();
auto total_columns = table.total_columns();
auto first_rowgroup_columns = rowgroups == 0 ? 0 : table.rowgroup_column_count(0);
if (rowgroups != 0 && first_rowgroup_columns != 0) {
	auto rowgroup = table.rowgroup(0);
	auto column = rowgroup.column(0);
	if (column.type() == galp::DataType::I8) {
		std::span<const int8_t> values = column.values<int8_t>();
	}
}
```

`galp::Table` 暴露表级 metadata 和只读数据视图：

- `rowgroup_count()`
- `total_columns()`
- `rowgroup_column_count()`
- `rowgroup_column_counts()`
- `rowgroup(size_t) -> galp::RowgroupView`
- `empty()`

`galp::RowgroupView` 暴露：

- `column_count()`
- `column(size_t) -> galp::ColumnView`

`galp::ColumnView` 暴露：

- `name()`
- `type()`
- `size()`
- `values<T>() -> std::span<const T>`

输出数据所有权仍由 `galp::Table` 持有；`RowgroupView` 和 `ColumnView` 是轻量只读视图，
生命周期不能超过其来源 `Table`。

`DecompressOptions::write_output=false` 可用于 write-back-free 路径。该路径不写出 decoded
global output，但仍返回表级 metadata；调用 `RowgroupView::column()` 或
`ColumnView::values<T>()` 需要 `write_output=true`。

当前稳定表级解压路径支持 i8/i16 的单 token rowgroup，包括 uncompressed i8、constant i8、
frequency i8/i16、FFOR i8/i16、FFOR+SLPATCH i8/i16、dictionary i8/i16、cross-RLE i8、
RLE i8/i16 以及 `EXP_EQUAL` alias。遇到其他 token 或 multi-op expression 时，reader 会抛出
`galp::UnsupportedFormatError`，其中包含 token、rowgroup、column 和 column name。

公共头不应 include `engine/`、`flsgpu/`、`benchmark/`、`alp/`、`nvcomp/` 或
`generator/` 内部路径。可用下面的 target 检查公共 API 边界：

```bash
cmake --build build --target galp_api_boundary_checks
```

## 构建

从仓库根目录配置：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
  -DGALP_BUILD_TESTS=ON \
  -DGALP_BUILD_TOOLS=ON \
  -DGALP_BUILD_BENCHMARKS=OFF \
  -DGALP_WITH_NVCOMP=OFF
```

常用开发目标：

```bash
cmake --build build --target galp_core galp_public_api_smoke galp_tests galp_cli -j
```

常用 CMake 选项：

```text
GALP_BUILD_TESTS       Build GALP tests
GALP_BUILD_TOOLS       Build galp_cli
GALP_BUILD_BENCHMARKS  Build generated bindings and micro-benchmarks
GALP_WITH_NVCOMP       Build nvCOMP compressor comparison targets
GALP_ENABLE_MULTI_COLUMN Reserved; keep OFF, ON fails configure by design
GALP_ENABLE_INSTALL    Generate install/export/package targets
```

默认建议保持 benchmarks 和 nvCOMP 关闭。只有需要 generated microbenchmarks 或 nvCOMP
对比时再打开：

```bash
cmake -S . -B build \
  -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
  -DGALP_BUILD_BENCHMARKS=ON \
  -DGALP_WITH_NVCOMP=ON
```

## Package 消费

G-ALP 导出 `Galp::core`：

```cmake
find_package(Galp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE Galp::core)
```

从 build tree 验证 package consumer：

```bash
ctest --test-dir build -R GalpPackageConsumerSmoke --output-on-failure
```

`galp_core` 是 CUDA static library。构建时会解析 device symbols，避免纯 C++ consumer
链接 `Galp::core` 时缺少 `__cudaRegisterLinkedBinary_*`。

最小示例：

- `galp/examples/public_api_reader.cpp`：只 include `<galp/galp.hpp>`，打印 rowgroup/column
  metadata 并读取第一列 span。
- `galp/examples/cmake-consumer/CMakeLists.txt`：演示 `find_package(Galp CONFIG REQUIRED)` 和
  `target_link_libraries(... Galp::core)`。

## Tests

跑全部测试：

```bash
ctest --test-dir build --output-on-failure
```

常用子集：

```bash
ctest --test-dir build -L public-api --output-on-failure
ctest --test-dir build -L gpu --output-on-failure
```

Reader tests 会优先使用 `FLS_READER_TEST_FILE`。未设置时使用仓库样本：

```text
data/fls/galp-test/data.fls
```

需要 CUDA 或样本文件的测试，在本地环境缺失时应 skip，而不是失败。

## CLI

`galp_cli` 提供两个主要入口：

```bash
./build/galp/tools/galp_cli read_table data/fls/galp-test/data.fls /tmp/galp_out.csv
./build/galp/tools/galp_cli benchmark data/fls/galp-test/data.fls --samples 5
```

读取单个 rowgroup：

```bash
./build/galp/tools/galp_cli read_table data/fls/galp-test/data.fls /tmp/galp_out.csv --rowgroup 0
```

benchmark 默认是 write-back-free / consume-only 路径：decoded registers 被消费，不写回
global output。需要测输出路径时加：

```bash
./build/galp/tools/galp_cli benchmark data/fls/galp-test/data.fls --samples 5 --include-materialize
```

常用 benchmark 参数：

```text
--samples N
--kernel-samples N
--rowgroup N
--per-rowgroup-workset
--no-mixed-dispatch
--no-rowgroup-prefetch
--prefetch-depth N
--prefetch-workers N
--max-prefetch-storage-bytes N
--stream-target-work-items N
--stream-max-rowgroups N
--include-materialize
--reuse-table-resources
--freq-patcher stateful|branchless|hybrid[:threshold]
```

## Benchmark 口径

当前主路径是 fused rowgroup prefetch：

```bash
PREFETCH_DEPTH=4 PREFETCH_WORKERS=0 STREAM_MAX_ROWGROUPS=1 \
  bash scripts/my_tool/run_bench.sh
```

`PREFETCH_WORKERS=0` 表示由 `galp_cli` 按 rowgroup 数自动选择 worker 数。
`--samples` 是独立 benchmark 采样次数，默认 5；`benchmark_wall_ms`、`query_wall_ms` 和
`kernel_event_ms` 输出中位样本，同时提供 `_min`、`_median`、`_mean` 字段。
`--kernel-samples` 控制每个样本内部的 kernel replay 次数，默认 1，用于保留旧的 kernel-only
重复测量口径。

本地脚本：

- `scripts/my_tool/run_bench.sh`：默认端到端 benchmark，输出 CSV 到 `scripts/my_tool/end2end_res/`。
- `scripts/my_tool/run_bench_code_cache.sh`：冷/暖 cache 对照，通过 `CACHE_DROP_MODE` 控制 cache drop。
- `scripts/my_tool/run_galp_io_sweep.sh`：IO/prefetch 参数矩阵。
- `scripts/my_tool/run_ncu.sh`：NCU profiler 入口。
- `scripts/my_tool/run_nsys.sh`：NSYS profiler 入口。
- `scripts/my_tool/diff.sh`：对比 `galp_cli` 和 FastLanes 解压结果。

关键指标解释：

- `benchmark_wall_ms`：benchmark 整体 wall clock。
- `benchmark_wall_ms_min/median/mean`、`query_wall_ms_min/median/mean`、
  `kernel_event_ms_min/median/mean`：跨独立样本的汇总。
- `resource_prepare_ms`：`--reuse-table-resources` 下 reader/pinned-pool 准备时间。
- `query_wall_ms`：reuse 模式下的查询主体时间。
- `pipeline_active_ms`：query wall 中除 pipeline setup 以外的活跃时间。
- `kernel_event_ms`：CUDA event 统计的 kernel 时间。
- `read_rowgroup_ms`、`pread_ms`、`zero_copy_view_setup_ms`：rowgroup 读取与 zero-copy 构造成本。
- `upload_workset_ms`、`upload_dma_gpu_ms`、`h2d_bytes`、`h2d_copies`：H2D upload 成本。
- `prefetch_wait_ms`、`prefetch_depth_block_ms`、`prefetch_byte_block_ms`：prefetch 等待与背压。
- `write_back`、`include_materialize`、`consume_only`、`write_back_free`：执行口径字段。

注意：streaming pipeline 中 host 读取、H2D 和 kernel 可以重叠，所以各阶段耗时求和可能大于
`benchmark_wall_ms`。

## 调试环境变量

```text
GALP_ROWGROUP_TIMELINE_CSV=/tmp/galp_timeline.csv
    输出每个 rowgroup 的 read/build/ready/upload 时间线。

GALP_MEASURE_H2D=1
    用 CUDA event 同步测量 upload_dma_gpu_ms。该模式可能降低 H2D/compute overlap。

GALP_DISABLE_ASYNC_H2D=1
    禁用 dedicated h2d_stream，H2D 回退到 default stream。

GALP_PINNED_ROWGROUP_PREWARM_BYTES=N
    限制 pinned rowgroup buffer 预热字节数。

GALP_PINNED_ROWGROUP_PREWARM_MODE=full|off|none|lazy
    控制 pinned rowgroup buffer 预热模式。

GALP_PINNED_ROWGROUP_PREWARM_SLOTS=N
    覆盖自适应 prewarm slot 数。

GALP_VALIDATE_SHARED_POSITION_OFFSETS=1
    调试 reader/expression offset 时启用共享 position offset 校验。
```

## Generated Bindings

`galp/benchmark/generated-bindings` 是生成产物。修改 generated kernel binding 时应先改
`galp/code-generators` 下的生成器，再提交对应 regenerated outputs。

`GALP_ENABLE_MULTI_COLUMN=ON` 当前会在 configure 阶段失败。multi-column/generated
`query_multi_column` translation units 编译时间过长，不纳入支持的 GALP benchmark 构建。
相关生成产物短期保留在源码树中，便于后续重新设计或删除。

## 开发约定

- public consumer 只依赖 `<galp/galp.hpp>` 和 `Galp::core`。
- `src/include` 是 private include 面，不对外承诺兼容。
- 不要从 public header 泄漏 `engine`、`flsgpu`、`benchmark`、`nvcomp` 等内部路径。
- 改 execution/runtime/memory 时，优先跑 `galp_core`、`galp_cli`、`public-api` 测试。
- 改 public API 或 package 相关代码时，必须跑 `GalpPublicApiSmoke`、
  `GalpPublicApiBoundaries`、`GalpPackageConsumerSmoke`。
- 改 generated binding 生成器时，把生成器和 regenerated outputs 放在同一个变更里。

## 架构演进总结

当前 G-ALP 的核心变化不是替换了解压算法主流程，而是把工程边界理顺：

- 从内部 CUDA 解压实现，演进为有稳定 facade 的库组件。
- 从外部可见 `reader`、`dispatch`、`flsgpu`，收敛为 public `galp::*` facade。
- 从强依赖 nvCOMP/benchmark，变为可裁剪构建。
- 从裸 CUDA resource 管理，逐步迁移到 RAII stream/event/pool/workset。
- 从默认 materialize benchmark，变为默认 write-back-free pipeline tuning 路径。

后续模块化重点仍然是继续拆小 `engine/reader.cuh` 和
`engine/execution/internal/table_pipeline.cuh`，把 reader、zero-copy plan、codec column
builder、resource preparation、prefetch integration 和 chunk runner 的边界进一步硬化。
