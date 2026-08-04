# DCT / Block-major 端到端实验汇总（2026-08-04）

## 1. 结论摘要

目前最新、全绿、最适合作为当前代码基线的是物理 GPU 0（NVIDIA GeForce RTX
4090）上的 50K、no-shuffle、feature-extraction 六路端到端实验。按排除 repeat 0
后的 hot p50：

1. image-major v2 pushdown：4331.5 images/s；
2. DALI：3561.0 images/s；
3. DCT block-major pushdown：1432.7 images/s；
4. image-major v3 pushdown：1402.0 images/s；
5. RGB-no-more：1281.1 images/s；
6. PyTorch：1250.5 images/s。

核心判断如下。

- image-major v2 比 DALI 快 21.6%，但二者分别使用 DCT 与 RGB 模型；这是部署级
  吞吐对比，不是同模型逐元素对比。旧 H100 结果中 DALI 比 v2 快 5.1%，说明这两条
  pipeline 的相对排名对 GPU 和执行环境敏感，不能外推为硬件无关结论。
- 在严格同域 DCT pipeline 中，image-major v2 当前最快，是 block-major 的 3.02 倍。
- block-major 比 image-major v3 快 2.2%，比 RGB-no-more 快 11.8%；它与 v3 已很接近。
- block-major 的新 active-output schedule 已不是主瓶颈：50K 下约 3.10 ms/batch；
  总 planning 约 19.07 ms/batch，达到原定 10–20 ms/batch 目标。
- block-major 当前主要瓶颈是物理访问：每个 hot repeat 实际读取 31.56 GiB，
  相对其 `full_compressed_payload_bytes` 只减少 0.142%，producer active 达
  34.65 ms/batch。
- image-major v2 只读取 10.52 GiB，且 rowgroup/pread 粒度规整，因此达到
  4331.5 images/s。
- image-major v3 的持久化存储最小、读取量也最低（7.94 GiB），但产生
  338,991 个 rowgroup 和 88,974 次 pread，producer 达 35.75 s/repeat，
  碎片化抵消了字节优势。
- 10K segment sweep 明确选择 `segment_size=1000`。segment 50 虽然 CPU 预览
  只选择 59.0% 坐标，却因反复读取相同物理向量而使每图 I/O 达到 segment 1000
  的 10.4 倍。

新 50K artifact 的 `ok=true`、`failures=[]`、`source_changes=[]`；四组 DCT strict
语义比较全部通过，DALI/PyTorch 的差异只作为 diagnostic。新旧结果的逻辑配置和
物理工作计数一致，但使用了不同物理 GPU，binding 二进制 SHA 也不同，因此不能将
绝对吞吐差直接解释为旧运行受干扰。新结果应作为 RTX 4090 当前代码基线；旧 H100
结果保留为硬件特定参考，若需要判断旧 H100 是否受干扰，应在 H100 上用当前同一
binary 重跑第 15.1 节命令。

## 2. 证据等级与工件

| 证据 | 物理 GPU | 用途 | 当前状态 |
| --- | --- | --- | --- |
| 最新 50K 六路 feature extraction | GPU 0，RTX 4090 | 当前代码的主要端到端吞吐、cold/hot、存储和 I/O | `ok=true`；0 failures；0 source changes；全部 stability gate 通过 |
| 旧 50K 六路 feature extraction | GPU 1，H100 80GB | 跨硬件历史参考 | 语义通过；旧 validator/source fingerprint；v3/PyTorch drift 超 5%，不作为当前主排名 |
| 1K 六路 feature extraction r2 | GPU 1，H100 80GB | 短跑正确性和启动敏感性筛查 | 正确 v3 数据；语义全通过；不作为主要吞吐结论 |
| 10K block-major segment sweep | GPU 0，RTX 4090 | segment 参数内部 A/B | 50/100/250/1000 通过；500 仅 stability gate 失败 |
| 1K 六路 r1 | GPU 1，H100 80GB | 历史错误实验 | v3 manifest 指向错误数据，已由 r2 替代，不使用 |

主要工件：

- 最新 50K：`/tmp/galp-e2e-sixway-feature-noshuffle-50k-gpu0-20260804-new`
- 旧 H100 50K：`/tmp/galp-e2e-sixway-feature-noshuffle-50k-gpu1-20260804-r1`
- 1K r2：`/tmp/galp-e2e-sixway-feature-noshuffle-1k-gpu1-20260804-r2`
- segment 50/100/250/500：目录名中写了 `gpu1`，但实际命令是
  `CUDA_VISIBLE_DEVICES=0`，因此物理 GPU 是 RTX 4090。
- segment 1000 最新工件：
  `/tmp/galp-blockmajor-segment-sweep-10k-gpu0-s1000-20260804-r1`

## 3. 硬件与软件环境

### 3.1 GPU

| 物理编号 | GPU | 显存 | 本报告用途 |
| ---: | --- | ---: | --- |
| 0 | NVIDIA GeForce RTX 4090 | 24,564 MiB | 最新 50K 六路比较；10K segment sweep |
| 1 | NVIDIA H100 80GB HBM3 | 81,559 MiB | 旧 1K/50K 六路比较；可选同 binary 复测 |
| 2 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97,887 MiB | 未使用 |

NVIDIA driver 为 590.48.01。命令中的 `--device cuda:0` 是
`CUDA_VISIBLE_DEVICES` 过滤后的逻辑设备：最新六路实验和 segment sweep 使用
`CUDA_VISIBLE_DEVICES=0`，所以逻辑 `cuda:0` 对应物理 GPU 0；旧六路实验使用
`CUDA_VISIBLE_DEVICES=1`，对应物理 GPU 1。

当前 contract 没有记录物理 GPU UUID，只记录逻辑 `cuda:0`。物理 GPU 归属来自
实际启动命令，这是后续 harness 应补充的可重复性字段。

### 3.2 Host 与软件

- CPU：2 × Intel Xeon Gold 5318Y，24 cores/socket，96 logical CPUs；
- RAM：2.0 TiB；
- OS：Linux 6.14.0-37-generic x86_64；
- Python：3.11.15；
- PyTorch：2.11.0+cu128；
- PyTorch CUDA runtime：12.8；
- NVIDIA DALI：2.2.0；
- CMake：3.28.3。

数据所在文件系统不同：

- DCT-major、image-major v2/v3 payload 位于 `/dev/nvme2n1` ext4；
- canonical JPEG 根目录和 block-major sidecar 位于 `/dev/nvme0n1p3` ext4。

因此 DCT 与 RGB pipeline 的裸盘 I/O 不是同一块 NVMe。机器有 2 TiB RAM，所有
6–24 GiB 数据集都能完整进入 OS page cache；现有 hot 结果应解释为长驻进程、
多 epoch 或 warmed-cache 吞吐，而不是强制清 page cache 后的首读盘性能。

## 4. 统一工作负载与模型

50K 正式实验与 1K 筛查使用相同逻辑配置：

- 数据集：ImageNet-1K validation；
- 顺序：`galp_image_id` 升序；
- `shuffle=false`；
- `drop_last=false`；
- batch size：50；
- workers：8；
- precision：FP32；
- seed：11997733；
- workload：feature extraction；
- feature stage：penultimate；
- 输出：`classhead.ch_tanh` 后的 `[N,192]`；
- `materialize_features=false`，因此计时不包含将 feature 写入磁盘；
- throughput scope：load + 顶层 H2D + model + feature statistics；
- model stream 使用 CUDA greatest priority。

### 4.1 DCT 模型

- Architecture：RGB-no-more JPEG-Ti ViT-Ti DCT；
- Checkpoint：`imgnetDCTViTTi_ep300_75.1.pth`；
- SHA-256：`bdbb1110b1d5383ab2774071834ee9597ceff85236b1ce212b2a0dc0af47c2cb`；
- 输入：全部 64 个 DCT coefficients；
- Y：`[N,1,28,28,8,8]`；
- CbCr：`[N,2,14,14,8,8]`；
- preprocess：RGB-no-more `ResizedCenterCrop_DCT(32,28)`，范围 `[-1,1]`。

### 4.2 RGB 模型

- Architecture：RGB-no-more ViT-Ti RGB；
- Checkpoint：`imgnetRGBViTTi_ep300_74.1.pth`；
- SHA-256：`a5aedaa7231fb4e756f032246aa05366363d250899faef255f12a90a4fe07ee5`；
- preprocess：resize shorter side 到 256，224×224 center crop，范围 `[-1,1]`。

两个模型 checkpoint 不同，所以 DCT 与 RGB pipeline 之间只能比较部署吞吐、TTFT
和资源，不能声称模型逐元素等价。

## 5. 六条 pipeline 与 baseline 定义

| Pipeline | 输入域 | 数据与执行路径 | Baseline 角色 |
| --- | --- | --- | --- |
| `dct_major_pushdown` | DCT | v1 DCT-major `component -> spatial block -> image`；block-major sidecar；native planless fixed-grid pushdown；contiguous segment；device output | 本次优化对象 |
| `image_major_v2_pushdown` | DCT | v2 image-major；每图 rowgroup selected decode；segment 50 | 主要严格同域布局 baseline |
| `image_major_v3_pushdown` | DCT | Compact-v3 tiled-z32；image-major vector rowgroups；segment 50 | 存储优先的新布局 baseline |
| `rgbnomore` | DCT | RGB-no-more canonical CPU-DCT loader/transform，再送入同一个 DCT checkpoint | 严格 DCT 语义参考和 CPU-DCT 部署 baseline |
| `dali` | RGB | JPEG reader；nvJPEG mixed decode；GPU resize、crop、normalize；RGB checkpoint | GPU JPEG/RGB 部署 baseline |
| `pytorch` | RGB | PIL/torchvision CPU decode、resize、crop；pinned H2D；RGB checkpoint | 标准 PyTorch RGB baseline |

GALP 三条路径的共同关键配置：

- cache capacity：0 MiB；
- exact plan cache capacity：0；
- decode workset capacity：512 MiB；
- decode batch rowgroups：64；
- rowgroup prefetch depth：16；
- rowgroup prefetch workers：4；
- scheduling：limited-overlap；
- low-priority decode/transform streams：开启；
- block-major double buffer：开启；
- crop execution：block-major 为 `auto`，v2/v3 为
  `rowgroup-read-selected-decode`。

## 6. Cold、hot 与 steady 的定义

### 6.1 Cold process scope

`cold_start` 使用 repeat 0 的 process-scope 统计，从 `run_pipeline` 入口开始，包含：

- contract 和样本身份验证；
- reader/adapter 构造；
- binding/profile 导入的 adapter-local 部分；
- checkpoint 和模型构造；
- loader preparation；
- CUDA device/context 相关准备；
- 显式 model prime；
- 第一个 measured repeat 的数据、模型与指标计算。

它不包含 Python interpreter 启动和进入 `run_pipeline` 之前的顶层 import。

终端打印数组中的 repeat 0 `throughput_images_per_s` 只是 batch-loop 吞吐。因为
loader prime 和 model prime 在 batch loop 前发生，不能把它当成 cold end-to-end。
报告中的 cold throughput 必须使用 `aggregate.cold_start.throughput_images_per_s`。

### 6.2 Hot aggregate

5-repeat 正式实验设置 `aggregate_exclude_first_repeat=true`。hot p50/CV/p95 使用
repeats 1–4：

- 模型、checkpoint、CUDA context 已驻留；
- PyTorch/RGB-no-more DataLoader 使用 persistent workers；
- CUDA allocator/native arenas 已增长并可复用；
- metadata mmap、动态库和 kernel 初始化已完成；
- OS page cache 很可能已被 repeat 0 填充。

hot 比 cold 更高是预期行为。最新 50K hot/cold-process 吞吐比为：block-major
1.25×、v2 1.59×、v3 1.41×、RGB-no-more 1.11×、DALI 1.36×、PyTorch 1.11×。

### 6.3 Steady within-repeat

steady throughput 只去掉每个 repeat 的第一个 batch。它不是另一个完整的 warm
repeat。1K block-major 只有一个 1000-image segment，去掉第一 batch 会把这个
segment 的主要 planning/I/O 成本几乎全部去掉，因此 1K steady 4687 images/s
严重高估真实 segment 周期吞吐。50K 有 50 个 block-major segments，去掉一批不会
去掉后续 49 次 segment 成本，故其 steady 1452 与 hot p50 1433 接近。

### 6.4 哪个指标更指导真实负载

- 长驻 feature extraction、训练式多 epoch：以 50K hot p50 为主；
- 单次工具调用、服务启动或短任务：同时看 process-cold TTFT 和 cold throughput；
- 真正“机器重启后/强制清 page cache 后的第一 epoch”：当前实验没有控制该条件，
  不能用名为 cold 的 process scope 代替；
- 1K hot 和 1K steady 只适合 smoke/诊断，不应替代 50K hot 端到端结论。

## 7. 最新 50K RTX 4090 六路端到端结果

配置：50,000 images，batch 50，1000 measured batches，warmup 0，5 repeats；物理
GPU 0（RTX 4090）；pipeline 顺序为 block-major、v2、v3、RGB-no-more、DALI、
PyTorch。repeat 0 是 cold process scope，hot 聚合使用 repeats 1–4。

| Pipeline | Cold process images/s | Cold TTFT ms | Hot p50 images/s | Hot CV | Endpoint drift | Hot first-batch p50 ms | Steady p50 images/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| image-major v2 | 2715.8 | **4216** | **4331.5** | 1.14% | **0.05%** | 22.8 | 4336.1 |
| DALI | 2609.0 | 5107 | 3561.0 | **0.18%** | 0.30% | **10.2** | 3560.0 |
| DCT block-major | 1142.8 | 4282 | 1432.7 | 0.69% | 1.79% | 507.3 | 1452.4 |
| image-major v3 | 994.9 | 4310 | 1402.0 | 1.70% | 3.78% | 45.8 | 1402.4 |
| RGB-no-more | 1154.5 | 4557 | 1281.1 | 0.62% | 1.72% | 336.5 | 1291.3 |
| PyTorch | 1125.0 | 4778 | 1250.5 | 0.45% | 0.90% | 403.3 | 1262.3 |

所有 pipeline 的 CV 都不超过 1.70%，endpoint drift 都不超过 3.79%；本次 stability
gate 全部通过。

block-major 与 v3 的 p50 差距只有 2.2%，且两者 hot repeat 区间有重叠；当前应表述为
“block-major 观测值略高、总体接近”，若要声称稳定胜出，需要交替顺序或随机化顺序
的 paired A/B，而不能只看这一次顺序执行的 p50。

关键 hot 比值：

- v2 / DALI：1.216×；
- v2 / block-major：3.023×；
- block-major / v3：1.022×；
- block-major / RGB-no-more：1.118×；
- block-major / PyTorch：1.146×；
- DALI / PyTorch：2.848×。

### 7.1 为什么 v2 和 DALI 的 hot 更高

DALI 使用 nvJPEG mixed decode 以及 GPU resize/crop/normalize；在模型和 CUDA 已经
驻留、JPEG 已进入 page cache 后，CPU 解码和 H2D 不再形成 PyTorch 那样的瓶颈。
但其 RGB checkpoint 与 DCT checkpoint 不同，因此这是部署 baseline。

image-major v2 每张图形成规整的 single-image rowgroup，50K hot repeat 只需要
50,000 rowgroups、50,000 preads，实际读取 10.52 GiB，planning 总计仅 0.22 s。
它以较大的持久化存储换取最直接的访问粒度，是当前 DCT 同域性能 baseline。

block-major 每 1000 张图把 transform/support 合并为一个 plan。跨不同图像尺寸的
crop 坐标并集覆盖大部分 spatial vectors，最终读取 31.56 GiB；虽然 double buffer
重叠了一部分 planning/I/O，但 producer 仍接近整个 repeat 的关键路径。

v3 读取字节更少，却使用一 vector 一 rowgroup 的细粒度 Compact-v3 布局；大量
rowgroup/pread、workset build 和 pinned staging 使其 producer 比 v2 高 3.10×。

### 7.2 与旧 H100 结果的吻合度及干扰判断

新旧 contract 除输出目录、validator 指纹和 native binding 工件外，逻辑配置完全
一致；sample manifest SHA 一致。三条 DCT pipeline 的实际读取字节、pread、
rowgroup、planned vectors、saved vectors 也逐项完全相同。block-major 两次都是：

- source contributions：168,473,070；
- contribution visits：336,946,140，即严格 `2C`；
- output/workset ownership：59,494,563；
- worksets：1,120。

因此两次执行了同一负载，结果不是由 sample、shuffle、segment 或访问计划改变造成。

| Pipeline | 旧 H100 hot p50 | 新 RTX 4090 hot p50 | 新相对旧 | 解释边界 |
| --- | ---: | ---: | ---: | --- |
| DALI | 4925.5 | 3561.0 | -27.7% | mixed decode/GPU preprocess，GPU 相关性强 |
| image-major v2 | 4687.2 | 4331.5 | -7.6% | 新 GPU model 更慢，但 host loader 更快 |
| block-major | 1456.9 | 1432.7 | -1.7% | 主要受 planning/I/O 限制，跨 GPU 最接近 |
| image-major v3 | 1366.4 | 1402.0 | +2.6% | 新 run 的 host producer 更快，抵消 model 变慢 |
| RGB-no-more | 1182.9 | 1281.1 | +8.3% | 新 run 的 CPU loader 明显更快 |
| PyTorch | 1090.3 | 1250.5 | +14.7% | 新 run 的 CPU JPEG loader 明显更快 |

阶段计时显示，新 run 中所有模型阶段都比旧 H100 慢 5.3%–21.4%，这与 RTX 4090
和 H100 的 GPU 差异方向一致；但 CPU-oriented loader 比旧 run 快：block-major
3.8%、v2 30.5%、v3 9.4%、RGB-no-more 22.0%、PyTorch 28.3%。这提示旧 H100 run
可能存在持续的 host 竞争、page-cache 状态或 worker 调度差异；DALI 的 loader 包含
mixed GPU decode，不能按纯 CPU loader 解读。

这仍不是“旧 run 确认受干扰”的充分证据，原因有二：物理 GPU 不同；native
binding 从 SHA `4b45d2fd...` 变为 `d4d62f0e...`，并非同一二进制。能确认的是：

- 新 run 更稳定且全绿，应作为当前 RTX 4090 基线；
- 旧 H100 run 的 DALI/v2 绝对值不能拿来覆盖新 GPU0 报告；
- 若要判断旧 H100 是否受干扰，必须在 H100 上使用当前 binary 和同一 contract
  再跑一次，并记录 GPU UUID、时钟/功耗和并发进程。

## 8. 严格语义结果

50K 实验从相同 sample manifest 捕获语义样本，sample identity 全部一致。

| Pair | Enforcement | 输入 max abs | 输出 cosine mean | 输出 max abs | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| block-major vs v2 | strict | Y 0；CbCr 0.000980394 | 0.999996861 | 0.0202685 | 通过 |
| block-major vs v3 | strict | Y 0；CbCr 0.000980394 | 0.999996861 | 0.0202685 | 通过 |
| v2 vs v3 | strict | 0 | 1.0 | 0 | 完全一致 |
| block-major vs RGB-no-more | strict | Y 1.19e-7；CbCr 0.000980451 | 0.999996861 | 0.0202640 | 通过 |
| DALI vs PyTorch | diagnostic | RGB 0.886275 | 0.998841 | 0.315141 | 仅诊断，不进入 failure |

DCT 差异不超过约一个 normalized integer DCT level（1/1020）。本次没有调整
tolerance 来掩盖差异。

## 9. 持久化存储

| 数据表示 | Persistent size | 相对 JPEG | 说明 |
| --- | ---: | ---: | --- |
| Canonical JPEG | 6.246 GiB | 1.00× | 当前文件系统快照，50,000 files；不是 contract 内 DCT storage 指标 |
| DCT-major v1 base | 13.138 GiB | 2.10× | manifest + DCT payload/metadata |
| DCT-major + block-major sidecar | 13.196 GiB | 2.11× | 本次 block-major 实际持久化占用 |
| image-major v2 | 22.889 GiB | 3.66× | 当前最快 DCT pipeline，存储最大 |
| image-major v3 tiled-z32 | **10.311 GiB** | 1.65× | DCT 布局中最小 |

block-major sidecar 为 62,997,320 bytes，令 base DCT-major 增加 0.447%。

布局间节省：

- block-major 总存储比 v2 少 42.35%；
- v3 比 v2 少 54.95%；
- v3 比 block-major 少 21.86%。

JPEG、DCT-major、v2、v3 的语义内容和运行路径不同，存储倍数反映部署占用，不是
同一种编码格式下的压缩率。DALI、PyTorch 和 RGB-no-more 使用 canonical JPEG；
它们没有额外预计算 DCT storage。

## 10. 50K 实际 I/O、planning 与资源

下表为 repeats 1–4 的 native hot mean，每一行对应完整 50K repeat。

| Pipeline | Read GiB | KiB/image | Reported read/full | pread | rowgroups | Vector saved | Planning ms/batch | Producer ms/batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| block-major | 31.558 | 661.8 | 99.858% | 33,230 | 18,307 | 10.59% | 19.07 | 34.65 |
| image-major v2 | 10.525 | 220.7 | 100% | 50,000 | 50,000 | 31.52% | 0.22 | 11.52 |
| image-major v3 | **7.942** | **166.5** | 100% | 88,974 | **338,991** | 31.52% | 0.73 | 35.75 |

实际读取量对比：

- v2 比 block-major 少读 66.65%；
- v3 比 block-major 少读 74.83%；
- v3 比 v2 少读 24.54%；
- block-major 的读取量是 v2 的 3.00×、v3 的 3.97×。

`reported read/full` 是 native 计数器中本次 read plan 相对其报告 full payload 的
比例，不等于统一 full-image pipeline 的端到端对照。block-major 该计数器显示
只节省 0.142% bytes；v2/v3 虽减少 31.52% vectors，但其 rowgroup 读取路径对所选
rowgroup 是 full read，因此计数器为 100%。

本次六路正式实验没有启用 `dct_major_full`，所以 `physical_evidence=null`。现有数据
足以报告“实际读取了多少”和布局间差异，但若要发表“pushdown 相对同布局 full
decode 节省 X% I/O”，必须补跑第 15.2 节的 full-vs-pushdown 对照。

### 10.1 Block-major planner / schedule 分解

50K block-major 每个 hot repeat包含 50 个 1000-image segments：

| 阶段 | Total per repeat | ms/batch |
| --- | ---: | ---: |
| 总 planning | 19.068 s | 19.07 |
| active-output schedule total | 3.096 s | 3.10 |
| group→workset build | 0.052 s | 0.052 |
| count | 1.357 s | 1.357 |
| prefix | 0.049 s | 0.049 |
| fill | 1.634 s | 1.634 |
| planless GPU kernel | 1.365 s | 1.365 |
| producer active | 34.648 s | 34.65 |

结构性优化证据保持成立：

- source contribution count：168,473,070；
- visit count：336,946,140，严格等于 `2C`；
- `host_expanded_transform_items_created=0`；
- `host_output_block_source_lists_created=0`；
- `host_global_transform_sort_items=0`。

因此下一步不应继续微调 count/prefix/fill；收益更大的方向是减少物理 payload
读取、缩短 producer 和改善 block-major 图像分组/物理 tile 粒度。

### 10.2 内存

| Pipeline | Host RSS p50 | Native GPU p50 | Native pinned p50 | Torch allocated p50 |
| --- | ---: | ---: | ---: | ---: |
| block-major | 3091 MiB | 1625 MiB | 79 MiB | 148 MiB |
| image-major v2 | 3473 MiB | 286 MiB | 71 MiB | 148 MiB |
| image-major v3 | 3271 MiB | 289 MiB | **1116 MiB** | 148 MiB |
| RGB-no-more | 1698 MiB | 未采集 | 未采集 | 164 MiB |
| DALI | 1950 MiB | 未纳入 GALP native counter | 未纳入 GALP native counter | 141 MiB |
| PyTorch | 1895 MiB | 未采集 | 未采集 | 169 MiB |

不能用 Torch allocated 数值代表 DALI 的全部 GPU 内存；DALI 管理的 native buffers
未包含在该列。v3 的 1.09 GiB native pinned peak 是当前明显资源问题，需要在后续
Compact-v3 producer 优化中单独调查。

## 11. 1K H100 筛查结果

配置：1000 images，20 batches，warmup 0，5 repeats，物理 GPU 1。r2 使用正确的
`compact_v3_tiled_z32` manifest，六路严格语义通过。

| Pipeline | Cold process images/s | Cold TTFT ms | Hot p50 images/s | Hot CV | Endpoint drift | Steady p50 images/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| block-major | 274.2 | 3435 | 984.7 | 3.57% | **9.02%** | 4687.4 |
| image-major v2 | 247.9 | 3433 | 2381.5 | 1.58% | 3.31% | 2464.9 |
| image-major v3 | 195.3 | 3573 | 886.0 | 1.72% | 2.47% | 901.1 |
| RGB-no-more | 201.6 | 4463 | 883.4 | 1.62% | 4.02% | 1217.5 |
| DALI | 184.2 | 4876 | 1639.5 | 1.02% | 0.73% | 1698.3 |
| PyTorch | 201.3 | 4396 | 833.8 | 3.88% | **5.41%** | 1223.3 |

1K 只包含一个 block-major segment。第一个 batch 承担整个 1000-image segment 的
planning/read，后续 19 batches 从已生成的 segment tensor 取数据，所以 block-major
steady 4687 images/s 不代表 50K 多 segment 负载。1K hot 也受 pipeline fill、worker
ramp 和单 segment 边界影响。其用途是快速发现 v3 数据错误、验证语义和观察启动，
不能用于最终排名。

## 12. Segment size 的含义

`dct-major-segment-size` 是 GALP native reader 一次共同规划、预取、读取、解码并
转换的连续图像数，不是模型 batch size。模型 batch 始终为 50。

以 segment 1000 为例：

- planner 为连续 1000 张图构造一个 block-major plan；
- 合并这些图的 crop/resize support 坐标；
- 构造 coordinate→group、group→workset 和 active-output schedule；
- reader/prefetch producer 读取并生成 1000 张图的 DCT tensor；
- 上层依次消费 20 个 model batches；
- double buffer 尝试让下一 segment producer 与当前 segment 消费重叠。

segment size 同时控制四个互相冲突的因素：

1. 较大 segment：同一物理 vector 被更多图像共同利用，减少重复 I/O 和重复
   planning；
2. 较大 segment：不同图像的坐标并集扩大，选择率变差；
3. 较大 segment：workset 和 native GPU memory 增大；
4. 较小 segment：坐标并集更稀疏，但可能反复读取同一个 1024-row 压缩 vector。

1000 是 batch 50 的倍数，也是低于 FastLanes 1024-row vector width 的最大 batch
倍数，避免跨 vector boundary，同时利用 97.7% 的 row width。

## 13. Segment size 实验

### 13.1 设置

- 物理 GPU：GPU 0，RTX 4090；
- workload：feature extraction；
- block-major only；
- sample count：10,000；
- batch：50，200 batches/repeat；
- warmup：0；
- repeats：3；
- hot aggregate：排除 repeat 0，只剩 2 个 hot samples；
- segment：50、100、250、500、1000；
- no shuffle；
- cache/plan cache：0；
- workset capacity：512 MiB；
- double buffer：on。

注意：50/100/250/500 的目录名误写为 `gpu1`，实际启动环境为
`CUDA_VISIBLE_DEVICES=0`；结果内部 A/B 有效，且与最新 50K 六路结果使用同一物理
GPU 型号。10K sweep 仍只用于 segment 内部选型，不用其绝对吞吐替代 50K 排名。

### 13.2 CPU preview

| Segment | Planned selected vector ratio |
| ---: | ---: |
| 50 | 59.0% |
| 100 | 64.5% |
| 250 | 84.3% |
| 500 | 91.3% |
| 1000 | 92.3% |

preview 证明小 segment 的坐标并集更稀疏，但它不执行物理读取，不能单独决定性能。

### 13.3 10K GPU A/B 结果

| Segment | Hot p50 images/s | 相对 1000 | Read/10K | MiB/image | Planning ms/batch | Schedule ms/batch | Producer ms/batch | Upload ms/batch | Native GPU | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 50 | 167.0 | 12.3% | 62.94 GiB | 6.445 | 202.7 | 3.05 | 299.5 | 56.0 | 293 MiB | pass |
| 100 | 302.0 | 22.2% | 34.59 GiB | 3.542 | 110.1 | 3.02 | 165.4 | 31.0 | 578 MiB | pass |
| 250 | 665.0 | 49.0% | 15.87 GiB | 1.625 | 46.9 | 3.03 | 74.9 | 14.2 | 1178 MiB | pass |
| 500 | 995.0 | 73.3% | 9.22 GiB | 0.944 | 29.8 | 3.34 | 49.8 | 9.25 | 1323 MiB | endpoint drift 5.22% |
| **1000** | **1358.0** | **100%** | **6.08 GiB** | **0.622** | **19.9** | 3.42 | **35.8** | **7.25** | 1616 MiB | pass |

### 13.4 讨论

吞吐随 segment size 单调增加，实际读取量随 segment size 单调下降。segment 50
每图读取 6.445 MiB，是 segment 1000 的 10.35×；这说明重复读取成本远大于坐标
选择率收益。

schedule 一直保持在约 3.0–3.4 ms/batch，几乎不随 segment size 增长；下降的是
重复 planning、producer 和 upload。只有 segment 1000 达到 planning 10–20
ms/batch 目标。

segment 500 的唯一 failure 是两次 hot samples 从 1021.6 降到 968.3 images/s，
endpoint drift 5.219%；CV 为 2.68%。即使取其更高样本，也明显落后于 segment
1000，所以不改变选型结论。

最终建议：

- 吞吐优先：segment 1000；
- 若 native GPU budget 必须下降：segment 500 将 1616 MiB 降到 1323 MiB，
  但损失约 26.7% hot throughput；
- 不建议 segment ≤250；
- 后续优化应面向物理 row/image tiling、producer 和 I/O，而不是继续缩小 segment。

## 14. 当前不可直接使用或不可声称的结果

### 14.1 最新 50K artifact 可用；旧 H100 artifact 仅作历史参考

最新 GPU0 结果在当前 validator 和当前 binding 下完成，`ok=true`、`failures=[]`、
`source_changes=[]`，吞吐、稳定性、严格 DCT 语义和 descriptor storage gate 都通过，
可以作为当前 RTX 4090 publication artifact。

旧 H100 run 在 GPU 执行后修复了 validator 的多 segment compact-plan 逻辑；旧
validator 曾把 50 个 segment 的累计 bytes 与单 segment peak 比较，造成 5 个伪
失败。修复后旧工件仍有 validator source fingerprint 变化以及 v3/PyTorch endpoint
drift 超 5%。因此旧工件可以辅助解释 H100 行为，但不再承担当前正式结果角色。

### 14.2 缺少同布局 full-decode I/O 对照

六路命令没有包含 `dct_major_full`，因此不能把现有 `read/full` counter 宣称为严格
full-image pipeline 的节省比例。第 15.2 节给出补跑命令。

### 14.3 RGB baseline 的实际 read bytes 未采集

DALI/PyTorch/RGB-no-more 只记录吞吐、latency 和内存，没有记录 per-repeat JPEG
physical read bytes。6.246 GiB 是 dataset persistent size，不等于每次运行的实际
NVMe bytes。加之未清 OS page cache，当前不能声称 DALI/PyTorch 的实际 I/O 节省。

### 14.4 新旧 50K 不能作为干扰的严格 A/B

最新 50K 与 10K segment sweep 都在 RTX 4090 上，二者可以共同支持 segment 1000
的当前 GPU0 选型。旧 50K 则在 H100 上，且 binding SHA 与新 run 不同；它与新 run
不能组成“只有干扰状态不同”的控制实验。若要确认旧 H100 是否受干扰，应使用第
15.1 节命令在 H100 上复测。

## 15. 建议补跑命令

以下命令均由用户在空闲 GPU 上执行；报告生成过程没有重新占用 GPU。

### 15.1 可选：同 binary 重跑 H100，判断旧 run 是否受干扰

最新 RTX 4090 artifact 已全绿，不需要为了 GPU0 报告再次运行。只有需要得到当前
binary 的 H100 结论，或严格判断旧 H100 run 是否受干扰时，才运行以下命令。运行前
应确认没有其他 GPU/CPU/I/O 重负载，并保留 `nvidia-smi` 与并发进程快照。

```bash
cd /home/tangyuxin/gfastlanes/FastLanes

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python -B \
  galp/benchmarks/system_dct_major/run.py \
  --preset e2e \
  --workload feature-extraction \
  --pipelines \
    dct_major_pushdown \
    image_major_v2_pushdown \
    image_major_v3_pushdown \
    rgbnomore \
    dali \
    pytorch \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --image-major-v3-manifest \
    galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32/manifest.bin \
  --image-major-v3-label-map \
    galp/data/system_rgbnomore/e2e_v3/compact_v3_tiled_z32/labels.json \
  --device cuda:0 \
  --dct-major-segment-size 1000 \
  --image-major-segment-size 50 \
  --block-major-double-buffer on \
  --output-dir \
    /tmp/galp-e2e-sixway-feature-noshuffle-50k-h100-current-20260804-r1
```

成功标准：`RESULT_JSON {"failures": 0, "ok": true}`。如果仅 endpoint drift 失败，
不要放宽阈值，保留所有 repeat 数组用于分析。

### 15.2 必需于严格 I/O 节省 claim：10K full-vs-pushdown

```bash
cd /home/tangyuxin/gfastlanes/FastLanes

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=build/galp/torch:galp/torch \
/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python -B \
  galp/benchmarks/system_dct_major/run.py \
  --preset e2e \
  --workload feature-extraction \
  --pipelines dct_major_full dct_major_pushdown \
  --block-major-access-dir /tmp/galp-block-major-access-v1-real \
  --device cuda:0 \
  --sample-count 10000 \
  --repeats 5 \
  --dct-major-segment-size 1000 \
  --block-major-double-buffer on \
  --output-dir \
    /tmp/galp-dct-full-vs-blockmajor-pushdown-10k-gpu0-20260804-r1
```

成功时 `results.json.physical_evidence` 应非空且 `ok=true`，并同时给出 physical
bytes、vectors 和 source blocks 的严格节省比例。

### 15.3 可选：需要 H100 segment curve 时确认 500/1000

当前 RTX 4090 sweep 已足以选 1000；只有需要 H100 上的 publication curve 时再跑：

```bash
cd /home/tangyuxin/gfastlanes/FastLanes

for segment_size in 500 1000; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=build/galp/torch:galp/torch \
  /home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python -B \
    galp/benchmarks/system_dct_major/run.py \
    --preset e2e \
    --workload feature-extraction \
    --pipelines dct_major_pushdown \
    --block-major-access-dir /tmp/galp-block-major-access-v1-real \
    --device cuda:0 \
    --sample-count 10000 \
    --repeats 5 \
    --dct-major-segment-size "${segment_size}" \
    --block-major-double-buffer on \
    --output-dir \
      "/tmp/galp-blockmajor-segment-10k-h100-s${segment_size}-20260804-r1"
done
```

## 16. 最终判断

现阶段可以确认：

- block-major planner 的连续 coordinate/group/workset 索引和两遍 schedule 优化已经
  达到预期结构目标，并将 schedule 压到约 3.10 ms/batch；
- 真实 50K 端到端性能不再受 contribution hash lookup 主导，而受物理 I/O、
  producer、workset upload 和 block-major 坐标并集覆盖率主导；
- segment 1000 是当前物理布局下正确选择；小 segment 会因重复读取而显著变慢；
- v2 是当前 DCT 性能最优方案，v3 是当前 DCT 存储最优方案；
- block-major 位于两者之间：存储远小于 v2、吞吐略高于 v3，但距离 v2 仍有
  3.02× 差距；
- 最新 GPU0 结果与旧报告的瓶颈、I/O、存储和 DCT 内部排序结论吻合，但顶层第一名
  从 H100 上的 DALI 变为 RTX 4090 上的 v2，且 block-major 对 v3 的优势从 6.6%
  缩小到 2.2%；这两项必须按硬件分别报告；
- 下一阶段若要提升 block-major，应改进 physical tile/image grouping 或 producer
  access，而不是继续优化已经较小的 active-output schedule。
