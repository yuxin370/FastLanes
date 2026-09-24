# CNN 全量在线 crop 推理实验

状态：2026-09-16 全部 50K 源数据已生成并通过逐系数校验；B6 M4 的 MobileNet32 GPU 输入检查通过。完整运行器先实测选择配置，再执行全量对照。

## 数据与计算边界

“完整数据”指现有 ImageNet-512 验证集全部 50,000 张，不是已经生成目标几何的 DCTNet 副本，
也不是 ImageNet 原始分辨率文件。五个模型共用一份全空间、每分量 64 频率的源 DCT。
离线只提取 JPEG 原始系数、无损列式压缩、建立访问索引；生成器分别记录这些耗时、校验耗时和大小。
不会离线 crop、resize、选模型通道或 normalize。

运行时在 512px 源图坐标执行 `[x=32,y=32,w=448,h=448]` 中心 crop。
eFUN 的输出为 28×28 个块，ResNet50 为 56×56，MobileNetV2 为 112×112；Cb/Cr 从源 4:2:0 几何在线放大。
这保留相同视野，但 DCT 域 resize 不等价于原论文的像素 resize/重新编码，准确率必须重新评估。

`require_all_coefficients=true` 保留 resize 的源频率依赖。输出 DC 也可能依赖源高频：
CPU 测试中，仅在一个 Cb 块的自然频率索引 7 放入非零值，resize 后的 DC 就发生变化。
因此不能把 DCT24 的四个 Cb 输出通道直接当作四个可读取的源通道。
块对齐 crop 自身不混合频率；恒等 resize 的分量也不因此需要所有源频率。
当前 reader 对整个变换统一保留 64 列，没有做逐分量依赖裁剪。

## 实验矩阵

五个模型：eFUN、MobileNetV2 DCT24 / DCT32、ResNet50 DCT24 / DCT64。
每条路径全量 50K、batch=64、FP32、TF32 关闭，独立运行三次，轮换路径顺序。

| 路径 | 读取/解码范围 | 在线 crop/resize 及输出 |
|---|---|---|
| crop_off / grid | 完整源图全部 rowgroup、全部向量 | 在线变换，物化全部输出频率，再选模型通道 |
| crop_off / projected | 完整源图全部 rowgroup、全部向量 | 在线变换，融合输出频率投影 |
| crop_on / grid | 在线 crop 的源支持范围，选择性读取/解码 | 与 crop_off 相同的变换和模型输入 |
| crop_on / projected | 在线 crop 的源支持范围，选择性读取/解码 | 与 crop_off 相同的变换，融合输出频率投影 |

`full-source-decode` 是本次新增的真正全源对照，仍执行相同 crop/resize，只关闭存储范围裁剪。
旧 `full-rowgroup-decode` 仍会跳过 crop 外 rowgroup，不作为 crop-off。

**源频率筛选与输出频率投影分开报告。** 在线 resize 依赖源高频，当前 reader 必须读取全部 64 个源频率，
不能把输出 DCT24/32/64 通道索引直接用作源列选择。`grid/projected` 比较的是在线变换后的输出投影下推，
不应称为源频率 I/O 剪枝。eFUN 本身保留全部 192 通道，只有输出布局/融合差异。
四条路径送入同一模型的形状、系数和计算量必须一致。

## Native 配置择优

配置候选为旧 reader M1，以及 B6 runtime 的 M1/M2/M4（每次 activation 处理 1/2/4 个 shard）。
B6 runtime 对齐 `galp/include/galp/profiles/direct_dct.hpp` 的 dynamic-crop policy，
采用 bounded io_uring（读取放大上限 1.1）、limited-overlap、低优先级流、异步完成、
16 层 rowgroup 预取、8 个预取 worker，以及 CNN B6 已采用的 32768 blocks/launch。
在线 crop/resize/projection 仍由 C++/CUDA 执行；每次一个 active activation 和一个 native future。
这是 B6 执行参数在推理 reader 上的使用，不调用带随机增强和 Mixup 的训练 PLS 接口。

每个模型在相同的前 8192 张完整源图上对候选各测三次，以 projected 路径 E2E 中位数选优。
先按双 activation 的 grid 对照内存需求排除放不下的配置。选定后，四条消融路径共用同一组参数，
避免把 activation 大小变化混入 crop 或输出投影收益。`calibration/selection.json` 保存各次实测及选择。
“最佳”仅指这些已实测候选中的最快配置，不宣称全局最优。

## 源数据生成并行度

`prepare_online_source.py --threads N` 现在表示总 CPU 预算，默认使用 CPU affinity 内的全部逻辑核（最多 256）。
本机 96 核使用 16 个并发 shard × 每 shard 6 个解码/编码线程，布局扫描 96 线程；校验器原生最多 16 线程。
不能只给底层工具传 `--threads 96`：该兼容参数不控制 FastLanes 编码并行度。
本次 50K 压缩 89.39 秒、校验 28.00 秒、索引 0.41 秒，系数差异为 0。
此前中断的单线程尝试耗时也保存在 `generation.json`，不计入这次成功压缩的 89.39 秒。

## 执行

在有可用 CUDA 驱动的主机上，从仓库根目录运行。数据与结果目录必须是新目录。
优先选择空闲 PRO 6000/H100；运行器每条路径启动前检查空闲，运行时以 100 ms 采样检查外部进程，发现即中止该次和后续测试。

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
export CUDA_VISIBLE_DEVICES=GPU-e796262d-3449-6af1-586d-8460d8836d1b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$HOME/tmp/dctnet" MPLCONFIGDIR="$HOME/tmp/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
EXP=galp/benchmarks/dct_models
SOURCE=galp/data/compressed/imagenet512_val_block_major
RESULT=galp/data/system_rgbnomore/e2e_v3/runs/cnn_online_pushdown_20260916

# 仅在创建新的源数据目录时执行；当前 SOURCE 已完成，无需重新压缩：
# "$PYTHON" "$EXP/prepare_online_source.py" --output-dir "$SOURCE" --threads 96
"$PYTHON" "$EXP/online_crop_suite.py" --source-data "$SOURCE" --output-dir "$RESULT"
```

前置依赖沿用现有实验环境：已编译的 `galp_jpeg_dct_tool`、`galp_block_major_access_tool`、
`_galp_direct_dct`、`dctnet_storage`，现有全源 compact 数据、DCTNet/RGB 权重、DALI、`nsys`。
需要编译时使用现有 build；不要为运行本实验重新生成目标几何副本。

CPU 检查：

```bash
"$PYTHON" -m unittest -v galp.benchmarks.dct_models.tests.test_online_crop
```

## 输出与判读

- `SOURCE/generation.json`：离线开销与大小；`verification.log` 必须 `exact: true`。
- `RESULT/report.md`、`summary.json`：全量精度、E2E、forward scope、重复测量、源读取计数。
- 每个模型 `calibration/selection.json`：候选配置、三次测量及最终使用的配置；每份 N 结果也保存完整 native 参数和 activation 大小。
- 每个模型 `verify/{crop_off,crop_on}/{grid,projected}/N_8.json`：独立 CPU 变换验证，各模式与全源 GPU grid 输入必须精确一致。
  CPU 浮点变换与 native 在半整数附近可能相差一个取整单位，检查对未取整参考的误差不超过 0.505。
- 每次完整评估保存全部预测，汇总要求同一模型四条路径和三次重复的 50K 预测一致。
- 每个模型 `profile/{crop_off,crop_on}/{grid,projected}/`：独立 4,096 图 Nsight 窗口及 breakdown。
  profile 耗时不混入 E2E 表。每次测试目录保存 GPU 进程采样。

不清理操作系统文件缓存，不能将结果称为冷盘吞吐。native 的 `full_vector_count` 仅以命中 rowgroup 为分母，
还应结合 `source_rowgroups` 与 `rowgroup_count` 判断整 rowgroup 跳过量。
若要声称模型计算减少，必须有模型结构/MAC 或实际模型 kernel 的证据；仅 E2E、H2D 改善不满足这个条件。
