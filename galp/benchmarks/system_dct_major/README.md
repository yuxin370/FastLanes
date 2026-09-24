# DCT-major locality benchmark

本目录测试 GALP 的物理顺序：

```text
component -> spatial DCT block -> image
```

storage API 名称为 `kSpatialMajorImageMinor`，benchmark 中简称 DCT-major。

## 正式矩阵

| Pipeline | 输入域 | 用途 |
| --- | --- | --- |
| `dct_major_pushdown` | DCT | GALP block-major 原生 crop pushdown，固定 `all`/K64 |
| `dct_major_coefficient_pushdown` | DCT | 同一 GALP adapter/profile/runtime，加可配置 raw coefficient pushdown |
| `rgbnomore` | DCT | RGB-no-more 严格语义参考 |
| `dali` | RGB | nvJPEG/GPU transform 部署参考 |
| `coordl`（可选） | RGB | CoorDL 单节点 MinIO JPEG 缓存，跨 repeat 保留 reader |
| `ffcv` | RGB | FFCV 顺序读取、中心裁剪和 GPU 输入参考 |
| `pytorch` | RGB | PIL/torchvision 部署参考 |

历史 `full`、`legacy_pushdown`、三代 image-major 和 planless/fixed planner A/B
不再属于正式 pipeline，也不能通过 Python contract 恢复。DCT 同域路径共享 DCT
checkpoint；RGB 路径共享 RGB checkpoint。

FFCV 需单独安装在运行 benchmark 的 Python 环境中。输入按 canonical JPEG 顺序离线转换为
`RGBImageField(write_mode="raw")` 的 `.beton`，保存解码后的像素而非重新压缩 JPEG；
转换时间不计入 pipeline 计时，文件比原始 JPEG 大。单次 `run` 默认转换到输出目录，
可用 `--ffcv-beton` 复用已经转换的同一数据集；`run_suite` 在正式运行前只转换一次，
供 smoke 和 formal 运行共用。FFCV 通过 ordinal 校验样本顺序，并参与 RGB 语义诊断和吞吐比值。

支持三个 workload：

- `feature-extraction`：输出 `[N,192]` penultimate features；
- `evaluation`：输出 `[N,1000]` logits 并统计 Top-1/Top-5。
- `training-throughput`：执行 FP32 forward、cross-entropy、backward 和 AdamW 更新，统计训练 step 吞吐。

## 固定输入的训练吞吐

`training-throughput` 复用七管线的固定裁剪、样本顺序和计时路径。默认仍使用前述
50K validation 输入作为性能数据，每轮遍历全部样本；这是带真实参数更新的吞吐测试，
不是 ImageNet 训练配方或收敛实验，不把训练后的准确率报告为 validation accuracy。
没有随机增强、shuffle、梯度累积、AMP 或 CUDA Graph。

各 RGB 路径从同一 RGB checkpoint 开始，DCT 路径从同一 DCT checkpoint 开始，
统一 seed、FP32、AdamW（lr=1e-4、betas=0.9/0.999、eps=1e-8、weight decay=0.05，
bias 和一维参数不衰减），固定学习率，不裁剪梯度。模型、优化器及 CoorDL reader
跨 repeats 保留；每个 batch 执行一次 zero_grad、forward、CE、backward、step。
首次 optimizer 状态分配和首次训练 kernel 执行计入首轮；后续轮作为热轮。

第一个 batch 在参数更新前以 eval 模式采集至多 `min(semantic_samples,batch_size)`
个样本，用于原有输入/输出语义对比。该采集及首次梯度/参数更新检查位于 batch-loop
计时之外，进程首轮总时间包含这些开销。每轮记录 loss、optimizer step 数和参数有限性；
`model_ms` 包含完整训练 step，loss 在 GPU 上累计，轮末读取。训练 loss 不是验证 loss。
K64 保留与 PyTorch DCT 的严格初始语义对比；RGB 保留 diagnostic 差异；
K32 仍是丢弃部分系数的近似路径，其既有全量 oracle 差异并未因训练模式而解决。

在空闲 RTX 4090 上运行 50K、5 轮完整训练吞吐对比（每管线 5,000 次参数更新）：

```bash
export PYTHONPATH="$PWD:$PWD/build/galp/torch${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 TMPDIR="$HOME/tmp" \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m galp.benchmarks.system_dct_major.run \
  --preset e2e --workload training-throughput \
  --pipelines dct_major_pushdown dct_major_coefficient_pushdown rgbnomore dali ffcv pytorch coordl \
  --block-major-access-dir galp/data/compressed/imagenet512_val_block_major/access \
  --sample-count 50000 --batch-size 50 --repeats 5 --dct-coeffs first:32 \
  --coordl-cache-size 25000 \
  --coordl-python "$HOME/tmp/coordl-build-cu128/coordl-python" \
  --ffcv-beton "$HOME/tmp/dct-major-coordl-full-20260924/ffcv.beton" \
  --output-dir "$HOME/tmp/dct-major-coordl-training-full-20260924"
```

已有七管线 GPU 功能 smoke：1024 张、batch50（含尾批）、2 轮，每管线 42 次更新，
结果位于 `$HOME/tmp/dct-major-coordl-training-smoke-20260924/results.json`，`ok=true`。
该次 GPU 存在其他计算任务，吞吐数值不用于正式排名。完整训练吞吐尚待空闲 GPU 测量。
`run_suite` 仍执行原有 feature/evaluation suite；训练吞吐使用上面的独立 `run` 命令。

## CoorDL baseline

对应论文为 Mohan et al., **Analyzing and Mitigating Data Stalls in DNN Training**,
PVLDB 14(5), 771–784, 2021，
[DOI](https://doi.org/10.14778/3446095.3446100) /
[论文全文](https://www.vldb.org/pvldb/vol14/p771-mohan.pdf)。
作者公开了 [DS-Analyzer](https://github.com/msr-fiddle/DS-Analyzer) 诊断工具和
[CoorDL](https://github.com/msr-fiddle/CoorDL) 数据加载库；这里比较后者。
接口核对版本为 `bcde72da21781aab0661eacdedbbd1dcca5f4cfe`（DALI `0.20.0dev`）。
CoorDL README 将新增代码声明为 MIT，仓库还保留 NVIDIA DALI 的 Apache-2.0 许可。

`coordl` 复用当前 RGB checkpoint、固定 512 输入的 224 中心裁剪、FP32 `[-1,1]`
归一化、样本 ordinal 和现有计时。第一版范围为 inference / feature extraction：
与现有 DALI、FFCV 一样使用 RGB 语义诊断；`rgbnomore` 是 PyTorch DCT 参考，
两条 block-major 路径继续使用现有 DCT 语义校验。RGB/DCT 跨域比值沿用部署对比口径。
缓存容量 `--coordl-cache-size` 的单位是 **JPEG 数量**，
不是字节；上游将超过样本数的容量截断为样本数。reader 在 repeats 间保留，首轮填充缓存，
后续轮次读取缓存；建议正式测量使用 `--repeats 5`。尾批在 reader 内 padding，iterator
只返回有效样本，并保证下一轮从 ordinal 0 开始。该对比只覆盖单节点 MinIO 缓存，
不测分布式 partitioned caching 或多个训练任务间的 coordinated preprocessing。
它是当前 feature/evaluation workload 下的部署对比，不是论文训练加速比的复现。

**环境要求：** CoorDL 与普通 DALI 共用 `nvidia.dali` 包名。用 `--coordl-python`
指定安装了 CoorDL 且能运行本 benchmark 的独立 Python 环境，普通 `--python`
继续运行现代 DALI、其余管线和结果验证。CoorDL 上游构建说明针对 Python 3.6、
PyTorch 1.0、CUDA 10；其原始容器不能直接运行当前 benchmark，需兼容当前 Python、
PyTorch 和 GPU 的 CoorDL 构建。`pip install nvidia-dali-*` 不会提供 CoorDL。
已用下述兼容构建在 Python 3.11 / CUDA 12.8 / RTX 4090 上跑完七管线的
feature-extraction 和 evaluation GPU smoke（1024 张图、batch size 50、5 repeats）。
样本顺序、尾批、DCT 严格校验和 `first:32` raw-mask oracle 检查通过。
CoorDL 对 PyTorch RGB 的特征余弦为 0.99913，logits 余弦为 0.99897；后者略低于
诊断阈值 0.999，像素级诊断也未通过。两者都使用默认 `use_fast_idct=False`，
这里保留上游解码实现及现有 RGB 诊断口径，不声明逐像素或 logits 严格等价。
两次运行的 `results.json` 均为 `ok=false`，失败项来自吞吐 CV 或首尾漂移超过
原有 5% 门槛；RGB 差异单独记录为 diagnostic。这次 smoke 不作为正式性能结论。

50K、5 轮的完整七管线结果见 [CoorDL 全量实验报告](docs/COORDL_FULL_EXPERIMENT_2026-09-24.md)。
报告保留首测和一次稳定性复测，并明确记录 K32 全量 raw-mask oracle 一致率未达标。

### 构建当前 Python 可用的 CoorDL

[coordl_compat.patch](coordl_compat.patch) 只修正现代 nvJPEG 版本检测、C++17
所需头文件和 `std::size` 名称冲突，不修改 CoorDL 缓存策略或预处理算法。
另外将 pybind11 固定到支持 Python 3.11 的 v2.13.6。此配方针对 CUDA **12.8**；
CUDA 13 已删除上游使用的部分 NPP 接口，不能直接替换。
当前 Python 环境需已有 benchmark 依赖，以及 OpenCV、libjpeg-turbo、Protobuf 的
C++ 开发文件。本次使用 OpenCV 4.13、Protobuf 6.31.1 和 CMake 3.28。

在仓库根目录执行，`coordl_cuda` 指向完整 CUDA 12.8 toolkit，包含 nvJPEG/NPP：

```bash
coordl_repo="$PWD"
coordl_env=/home/tangyuxin/miniconda3/envs/fastlanes-cuda
coordl_source="$HOME/tmp/coordl-baseline-source"
coordl_build="$HOME/tmp/coordl-build-cu128"
coordl_cuda="$HOME/tmp/coordl-cuda-12.8"
mkdir -p "$HOME/tmp"
git clone https://github.com/msr-fiddle/CoorDL.git "$coordl_source"
git -C "$coordl_source" checkout bcde72da21781aab0661eacdedbbd1dcca5f4cfe
git -C "$coordl_source" submodule update --init --depth 1 \
  third_party/boost/preprocessor third_party/dlpack third_party/rapidjson third_party/pybind11
git -C "$coordl_source/third_party/pybind11" fetch --depth 1 origin tag v2.13.6
git -C "$coordl_source/third_party/pybind11" checkout v2.13.6
git -C "$coordl_source" apply "$coordl_repo/galp/benchmarks/system_dct_major/coordl_compat.patch"
TMPDIR="$HOME/tmp" cmake -S "$coordl_source" -B "$coordl_build" -G Ninja \
  -DARCH=x86_64 -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_TOOLKIT_ROOT_DIR="$coordl_cuda" -DCUDA_TARGET_ARCHS=89 \
  -DPYTHON_EXECUTABLE="$coordl_env/bin/python" -DCMAKE_PREFIX_PATH="$coordl_env" \
  -DProtobuf_USE_STATIC_LIBS=OFF -DPYBIND11_TEST=OFF \
  -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF -DBUILD_NVTX=OFF -DBUILD_LIBTIFF=OFF \
  -DBUILD_NVOF=OFF -DBUILD_NVDEC=OFF -DBUILD_LIBSND=OFF -DBUILD_NVML=OFF -DBUILD_FFTS=OFF
TMPDIR="$HOME/tmp" cmake --build "$coordl_build" \
  --target install_backend_impl python_function_plugin -j8
TMPDIR="$HOME/tmp" "$coordl_env/bin/python" -m pip install \
  --target "$coordl_build/dali/python" future==1.0.0
cat > "$coordl_build/coordl-python" <<EOF
#!/usr/bin/env bash
export PYTHONPATH="$coordl_build/dali/python\${PYTHONPATH:+:\$PYTHONPATH}"
exec "$coordl_env/bin/python" "\$@"
EOF
chmod +x "$coordl_build/coordl-python"
export COORDL_PYTHON="$coordl_build/coordl-python"
```

`89` 对应本次 RTX 4090。构建输出和 `future` 均放在独立目录，通过启动器的
`PYTHONPATH` 选择 CoorDL，不覆盖环境中的普通 DALI。
当前机器已准备好 `/home/tangyuxin/tmp/coordl-build-cu128/coordl-python`，可直接复用。

在仓库根目录，用准备好的 CoorDL 环境检查必要接口；应输出 `CoorDL FileReader available`：

```bash
export COORDL_PYTHON=/path/to/compatible-coordl/bin/python
"$COORDL_PYTHON" -c 'from nvidia.dali import backend; assert "cache_size" in backend.GetSchema("FileReader").GetArgumentNames(); print("CoorDL FileReader available")'
```

第一版七管线 feature smoke（在仓库根目录执行，使用现有 fixed-512 数据和 sidecar；
1024 张图覆盖首个完整 shard，batch size 50 同时测试尾批；5 repeats 包含首轮及热缓存轮次）：

```bash
export PYTHONPATH="$PWD:$PWD/build/galp/torch${PYTHONPATH:+:$PYTHONPATH}"
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m galp.benchmarks.system_dct_major.run \
  --preset e2e --workload feature-extraction \
  --pipelines dct_major_pushdown dct_major_coefficient_pushdown rgbnomore dali coordl ffcv pytorch \
  --block-major-access-dir galp/data/compressed/imagenet512_val_block_major/access \
  --sample-count 1024 --batch-size 50 --repeats 5 \
  --dct-coeffs first:32 --coordl-cache-size 512 \
  --coordl-python "$COORDL_PYTHON" \
  --output-dir "$HOME/tmp/dct-major-coordl-feature-v1"
```

成功后检查输出目录的 `results.json` 中 `ok` 和各项语义诊断，以及
`pipeline_coordl.json` 的逐轮吞吐。正式运行使用 `--sample-count 50000` 和合适的缓存容量；
evaluation 沿用下文现有命令的 raw-mask oracle 要求。
本次 smoke 输出位于 `$HOME/tmp/dct-major-coordl-feature-v1` 和
`$HOME/tmp/dct-major-coordl-evaluation-v1`。上游生成在工作目录的 `0-512.log`
已分别收至对应输出目录的 `coordl-cache.log`。

在现有 `run_suite` 命令中附加
`--coordl-python "$COORDL_PYTHON" --coordl-cache-size 25000`，即可将 CoorDL 加入
contract、smoke 和 formal 阶段。未设置缓存容量时，suite 保持六管线矩阵。
结果包含 CoorDL 相对 DALI、PyTorch RGB、FFCV，以及两条 block-major 路径相对
CoorDL 的冷热吞吐比；PyTorch DCT (`rgbnomore`) 继续列在同一结果表中。
缺少真正的 CoorDL 接口时直接报错，不会退回普通 DALI。

上游硬编码使用 `/dev/shm/cache`，需预留足够共享内存；它不属于进程 RSS，不能仅用
现有 RSS 指标比较总内存占用。adapter 使用完整源路径作为相对缓存键，拒绝已有的同名
缓存，正常完成后只删除本次运行的缓存文件。异常退出后需在确认没有同数据集的 CoorDL
任务运行时，手动清理报错指出的缓存条目；不要并发运行使用相同 JPEG 的 CoorDL baseline。
上游 file-list 格式不支持文件路径包含空白字符。

## 不变量

- 所有 sampler/reader 都是 `shuffle=false`；
- `drop_last=false`，支持 partial tail；
- sample ordinal 等于物理 `galp_image_id`；
- 完整 DCT-major storage 始终保存 64 个系数；GALP baseline 读取 `all`，coefficient-pushdown pipeline 默认读取 `first:32`；
- coefficient selection 在 dequantization/frequency mixing 前应用，模型仍接收 dense-64 FP32 grid 和 N=196 tokens；
- 生产运行要求完整 manifest shard 和预先物化的 `BLOCK_MAJOR_ACCESS_V1` sidecar；
- transformed DCT 允许最多一个归一化整数级误差（`1/1020`）；
- physical bytes、vectors、source blocks、rowgroups 和 preads 必须写入结果；
- crop-pushdown 只有在 bytes、decoded vectors 和 source blocks 均小于完整输入时才成立。

## 语义 profile 与原生运行策略

两条 GALP pipeline 使用同一个语义 profile
`rgbnomore-validation-center-crop-512-v1`：输入是固定 64×64 luma block 网格，
保持 RGB-no-more `ResizedCenterCrop_DCT(32, 28)` 语义（中心裁 56×56 后缩小到
28×28，chroma 对应 28×28→14×14）和 FP32 归一化。其原生 runtime policy 固定为
`block-major-p4-scheduled-bounded-110-v1`：

- planless execution，decoded-rowgroup cache/plan cache 为 0；
- decode rowgroup batch 64，workset capacity 512 MiB；
- rowgroup prefetch `depth/workers/min-batches = 16/8/1`；
- limited-overlap、低优先级 stream、异步 completion；
- transform launch 512 blocks / 512 CTAs；
- scheduled bounded io_uring；
- global read amplification 1.10，local amplification 继承，run 上限为 rowgroup；
- double buffer 由 runtime policy 自动选择。

Python contract 只记录 runtime policy 身份，不记录上述 planner、allocator、I/O 或 kernel
细节。selection 是独立的 request 参数，支持 `all`、`first:N`、`list:i,j,...`；
benchmark 还把 `random:K:SEED` 一次性解析成确定性的 native `list`。contract 同时记录
原始 spec、zigzag column indices、natural 8×8 indices、K 和 seed。

## Quick start

运行以下 GALP 命令前先按 [run guide](docs/RUN_GUIDE.md) 物化首分片的
`active_output_schedule`；正式 `first:32` evaluation 还需将
`RAW_MASK_ORACLE_DIR` 指向已完成的 raw-mask 评估目录。

```bash
RAW_MASK_ORACLE_DIR="$PWD/galp/benchmarks/coefficient_mask_evaluator/runs/imagenet_val_k1_64_20260816_h100"
```

只生成 contract（manifest 与 labels 必须来自同一个 fixed-512 数据视图）：

```bash
PYTHONPATH=.:build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m galp.benchmarks.system_dct_major.run \
  --preset e2e \
  --workload feature-extraction \
  --sample-count 1024 --repeats 1 \
  --dct-major-manifest galp/data/compressed/imagenet512_val_block_major/manifest.bin \
  --dct-major-label-map galp/data/compressed/imagenet512_val_compact_v3/labels.json \
  --block-major-access-dir galp/data/compressed/imagenet512_val_block_major/access \
  --dct-coeffs first:32 \
  --output-dir /tmp/galp-dct-major-dry-run \
  --dry-run
```

六管线 feature smoke（正式 coefficient spec 必须显式给出）：

```bash
PYTHONPATH=.:build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m galp.benchmarks.system_dct_major.run \
  --preset e2e \
  --workload feature-extraction \
  --sample-count 1024 --repeats 1 \
  --dct-major-manifest galp/data/compressed/imagenet512_val_block_major/manifest.bin \
  --dct-major-label-map galp/data/compressed/imagenet512_val_compact_v3/labels.json \
  --dct-coeffs first:32 \
  --block-major-access-dir galp/data/compressed/imagenet512_val_block_major/access \
  --output-dir /tmp/galp-dct-major-feature-smoke
```

正式 50K evaluation：

```bash
PYTHONPATH=.:build/galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m galp.benchmarks.system_dct_major.run \
  --preset e2e \
  --workload evaluation \
  --raw-mask-oracle-dir "$RAW_MASK_ORACLE_DIR" \
  --dct-major-manifest galp/data/compressed/imagenet512_val_block_major/manifest.bin \
  --dct-major-label-map galp/data/compressed/imagenet512_val_compact_v3/labels.json \
  --dct-coeffs first:32 \
  --block-major-access-dir galp/data/compressed/imagenet512_val_block_major/access \
  --output-dir /tmp/galp-dct-major-eval-50k
```

生产 DCT-major 要求 `warmup-batches=0`，避免 sidecar/cold-I/O contract 在运行后
发生变化。详细说明见 [run guide](docs/RUN_GUIDE.md)。

## 完整 suite

`run_suite.py` 固定执行：只读 contract preflight、K64 regression 与 K32/K16/list
raw-mask semantic gate、约 1K 的六管线 feature/evaluation smoke、两次六管线 formal 和四个 DCT/RGB model-only
ceiling。它不再做 segment sweep、自动选优、legacy
ABBA、crop A/B 或 plan-audit compare。

```bash
PYTHONPATH=.:build/galp/torch \
  /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m galp.benchmarks.system_dct_major.run_suite \
  --raw-mask-oracle-dir "$RAW_MASK_ORACLE_DIR" \
  --dct-major-manifest galp/data/compressed/imagenet512_val_block_major/manifest.bin \
  --dct-major-label-map galp/data/compressed/imagenet512_val_compact_v3/labels.json \
  --block-major-access-dir galp/data/compressed/imagenet512_val_block_major/access \
  --dct-coeffs first:32 \
  --output-dir /tmp/galp-dct-pushdown-k32-fixed \
  --dry-run
```

移除 `--dry-run` 后执行；中断后可使用相同参数加 `--resume`。runner 不覆盖不完整
phase。

## 测试

```bash
PYTHONPATH=. \
  /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python \
  -m unittest discover -s galp/benchmarks/system_dct_major/tests -v
```

`diagnostics/` 用于分析，不定义生产 API。历史 crop ABBA runner 已删除。
