# 给开发 agent 的起始任务

在实际 FastLens 仓库中实施 `IMPLEMENTATION_PLAN.md`。这是一份研究复现任务，不允许预设哪种方法应该更快。当前包仅有规格、配置示例和任务依赖，没有已经实现的 backend 或实测结果。

## 第一轮必须完成

1. 读取仓库现有贡献/运行规则，定位 CPU-DCT reference、FastLens writer、model profiles、推理和训练脚本、现有数值检查。创建 integration_map，使用真实路径和 commit。
2. 解析 `contracts/matched.example.yaml` 的 unresolved/null 字段。缺少数据、checkpoint 或硬件时先完成可独立的格式/reference 工作，记录阻塞项；不能编造路径、profile/source-K、模型准确率或运行日志。
3. 冻结 SourceManifest、ModelProfile 和逐 occurrence 的 RequestTrace。直接提取源 JPEG 的 q/Q，禁止 RGB 解码再 DCT 冒充同一源；禁止把输出 DCT24 当作 24 个源频率。
4. 写小规模独立 oracle 与验证集合。不要将同一有错误的 planner 同时用作 expected 和 actual。只有 q、metadata、transform、model/更新通过匹配验证，才跑正式速度实验。
5. 接口冻结后并行派 IC、FS、harness 三个任务，公共接口和 reference 由单一 owner 维护。

## 不可改动的研究约束

- 原生 IC 与当前 JPEG-DCT matched adaptation 分表。IC 的 `u+v<k` 三角裁成 k×k 后按 block 拼为三平面 mosaic；B8,k8 是36个有效系数，非all64。
- 原生 FS B32/Q50；matched FS B8 保留原 JPEG q/Q，用15条反对角线，默认每256图一组。不能偷偷改成64个单系数列、bit-plane或二维空间索引。
- IC/FS 主复现压缩采用 numeric NPZ/DEFLATE；Zstd、LZ4、GPU解压、额外seek索引属于单独 enhanced variant。
- 保留 IC 固定 source-K 的离线专属格式能力，同时计入所有副本/转换成本。动态 crop/resize 不可离线预固化进主表。
- FS 读取完整所需 diagonal/group，再在计时内 gather；额外读到的图片、空间和频率必须计量。多消费者用有界共享缓存，所有可共享系统享有相同机会。
- 各实现允许合理 workers、prefetch、pinned H2D 与重叠；共同资源预算、相同模型 batch、相同 trace。不能以单线程 Python 热循环制造慢基线。
- measured、emulated、modeled 三类结果分开；warm OS cache 下的 application bytes 不等于物理 disk bytes。
- 计时从首个被测请求前开始，到最后 CUDA 工作完成；warm-up 后恢复正式训练起始状态。不得在计时外准备被测首 pool。
- 失败 gate 不进入主表；不自动放宽数值容差，不遗漏失败样本，不调参追逐原论文速度倍数。

## 完成后交付

提交实现、锁定的 source/profile/trace/checkpoint/tuning manifest、测试和验证日志、原始逐 run JSON/JSONL、自动汇总表、存储和离线成本、ASSUMPTIONS/DEVIATIONS、可复跑命令。先交三个代表模型的完整50K inference与一epoch training，再补齐已有七个模型和第二GPU。提供 IC-b、位宽、选择粒度、局部性和多消费者关键角点。

若 scope 扩到 native IC 搜索或质量 Pareto，单独启用 native contract；不要为 matched throughput 训练6048个网络。完整搜索和B32 FastLens原生匹配扩展是后续工作包，不是本轮所有任务的共同前置。
