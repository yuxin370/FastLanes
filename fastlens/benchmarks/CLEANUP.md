# 已执行的目录清理

2026-10-21 清理了 42 个可重建目录，合计释放 **4.74 GiB**。
其中按原清单清理 41 个目录（1.95 GiB）；新构建验证通过后，移除了闲置的
旧名称下的组件构建目录（2.79 GiB），当前构建统一使用 `build/fastlens/`。
删除前检查了 Git 跟踪文件和活动进程引用；源码、数据集和现有回归测试保留。

| 已删除的生成目录 | 原用途 |
| --- | --- |
| `build-galp-torch-verify/` | 旧 Torch 验证构建；脚本默认路径已改为 `build/` |
| `build-tsan/` | 旧线程检查构建 |
| 旧名称下的组件构建目录 | 已由验证通过的 `build/fastlens/` 构建取代，旧兼容链接已删除 |
| 根目录 `_deps/` | 旧依赖缓存；当前构建仍使用 `build/_deps/` |
| `tmp/fls_quick_fuzz/` | 十组生成的随机测试数据 |
| `.cache/`、`.pytest_cache/`、源码目录 `__pycache__/` | 工具和字节码缓存，可再次生成 |

## 已归档的历史材料

32 组历史路径已移至仓库根目录
[`benchmark_results/archive/pre_fastlens_cleanup_20261021/`](../../benchmark_results/archive/pre_fastlens_cleanup_20261021/)，
归档目录及记录中的路径已统一使用 `fastlens`，测量值、格式标识、冻结源码和二进制内容保留：

- `fastlens/benchmarks/dct_retokenization/runs/`：约 1.7 GB，旧训练检查点和探索结果。
- `fastlens/benchmarks/coefficient_mask_evaluator/runs/`：约 171 MB，旧 mask 评估。
  retokenization 分析入口已指向归档位置。
- `fastlens/ImageCalulatorRe-implement/`：约 184 MB，方案和实现快照；文档引用已更新。
- 根目录 `0-*.log`、`pipeline_*.json/log`、`semantic_*.npz` 和 2026-09-12
  的 contract、commands、results、manifest、metadata、validation、report、index 输出。

这些材料不属于最终论文的运行源码。归档只整理位置，不释放其占用空间。

## 保留的内容

根目录 `experiments/` 已按用途拆分并移除：论文实现归入 `paper/` 的对应章节或
`paper/shared/`，执行工具归入 `execution/`，FFCV 准备代码归入 `baselines/ffcv/`；
开发诊断、测试、历史队列与报告生成分别归入 `development/` 的对应子目录。
两份旧报告和工作区笔记保存在
[`benchmark_results/archive/experiments/`](../../benchmark_results/archive/experiments/)。
没有保留根目录兼容链接；实验计算、默认参数和已发表结果表不变。
同步更新了5,715份历史文本中的脚本路径；原文本压缩备份保存在
`/home/tangyuxin/tmp/fastlens-experiments-reorg/records-before.tar.zst`。

- `build/`：当前构建及下载的依赖；FastLens 目标已重新构建并验证。
- [`paper/results/`](paper/results/)：最终 PDF 采用的完整表格和绘图输入。
- 根目录 `results/`：约 173 GB，原始测量、预测、检查点和 contract。
  新的部分端到端实验仍需要其中输入，不能整目录删除。
- `benchmark_results/`：历史性能和诊断依据，包含上述归档。
- `fastlens/data`：链接到已同步改名的数据集 `/var/dataset/tangyuxin/fastlanes/fastlens/data`。
  根目录 `data/` 是 FastLanes 的测试数据。
- `test/`、`fastlens/tests/` 和实现旁的 `tests/`：现有回归测试；旧日期或旧轮次名称
  不代表测试已失效。
- 根目录 `benchmark/`：FastLanes 自身的 CPU benchmark。
- `failed.jpeg`、`review_exports/`：用户的故障样本和导出材料。

`dct_retokenization/`、`coefficient_mask_evaluator/`、`metadata/`、
`model_only_training_calibration.py`、`run_block_major_graph.py` 属于开发或诊断工作流；
`baselines/pcr/` 和 `baselines/pcr_tensorflow/` 属于后续稿件。
这些源码保留，但不纳入 `fastlens-final.pdf` 的审稿复现入口。

## 路径统一

实验记录和数据清单中的路径已统一使用 `fastlens`；仓库根目录和 `build/`
下不再保留旧兼容链接。外部数据根目录同步改名，数据文件未搬盘或重新编码。
本轮更新了 12,952 份文本和 42 个目录，论文结果表中的测量值保持不变。
当前训练数据合同的 mapping 和输入列表摘要已同步刷新，历史运行记录中的指纹保留。
原文本压缩备份位于 `/home/tangyuxin/tmp/fastlens-path-unification/original-texts.tar.zst`。
