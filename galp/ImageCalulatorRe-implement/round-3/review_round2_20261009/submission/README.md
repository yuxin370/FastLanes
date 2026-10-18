# round-2：判断、修复与可复核证据

审查中的 R01–R06 合理，本轮已实现对应修复。两点需要限定：附件缺少外部 patch 是交付材料缺口，不能据此断言仓库没有接入；标准 batch64/pool4096/accumulation16 在 1,281,167 图像上本来就是每 epoch 1252 次更新，R03 影响的是允许的其他 pool/tail 组合。

本包基于 `13fabad2b337945ea32d50dd6b6cbebc5563b031`，提供 [完整集成补丁](integration.patch)、[实测依赖与上游版本](dependencies.json)、[复现脚本](reproduce.py)、原始 CPU/H100 JSON/日志及小型 JPEG/NPZ/trace fixture。补丁包含此前未随附件交付的 C++ bridge、CNN/Transformer 正式训练入口和报告器修改，不包含用户已有的无关 inference 修改。请在该 base 的干净 checkout 上应用；当前工作区已经含有修改，无需再次 apply。

## 修复与验收

| 问题 | 当前修改 | 验收证据 |
| --- | --- | --- |
| R01 内容身份 | 每个 q/Q NPZ 保存 size/SHA-256；完整 store fingerprint 同时进入证据、结果和正式 run manifest。计时前核验全部内容，实际 decode 再核验读取字节 | 同 shape/dtype 的合法 q、Q 重写均被拒绝；正常 stores 和真实模型验证通过 |
| R02 活跃工作集 | 限制 decode/scatter chunk 的字节数和 future 数；uncached 完整数组在 worker 返回前释放，输出和 LRU 另有预算 | cache=0、跨八组、四 workers 的弱引用测试检查实际数组生命周期；数值与文件次数正确；不足预算明确失败 |
| R03 schedule | 同一 pool/epoch/tail 规则推导完整人口的每 epoch 更新数；显式 total_updates override 写入契约 | 8/2/2/4 得到 4 次，100/4/12/4 得到 9 次，标准配置 1252 次；默认与显式 override 的完整输入校验通过 |
| R04 CPU 计时 | CPU 与 wall 同时起止，完成最终 CUDA drain 后结束；初始化、内容核验、容量扫描另记 | 窗口外注入 25+75ms 不进入 CPU 指标，窗口内两次 50ms 被计入 |
| R05 聚合身份 | 本地 `0.2-diagnostic` schema 声明 store/code/binding fingerprint；分组包含这些身份，summary 保留 model/profile、device、config/checkpoint、运行参数和 raw 行引用 | 完整 JSON Schema 校验；相同实现合并，修改实现/store 或 model/device 分组；每组能回溯原始行 |
| R06 操作顺序 | training 与 inference 分别生成实际顺序：几何 → integer round/clamp → RandAugment/逐步 clamp → model normalization → Mixup | CNN、Transformer 执行顺序检查；非恒等 resize、非零 chroma；独立 normalization/Mixup/soft-target golden |

其他审查问题也已处理：完整类别映射单独保存并 hash；标签必须是有效整数且符合冻结映射；校准和验证共用 microbatch/pool/tail validator；模型请求携带实际 crop 的分量 block regions；分别记录实际需求与 full-grid 需求；推理拒绝未实现的 BF16；IC/FS serialization_order 分别描述；CUDA first-batch event 已接入结果计算，但尚需宿主验证其计量。

默认 reader 限制为 512 MiB 活跃 decoded 数组、1 GiB reader 数组总预算。诊断 replay 另以 4 GiB host input 预算约束 current/next pool、stacking 和 CUDA pinned staging。**这些不是 RSS 保证**：manifest/index 对象、NumPy inflate/transform 临时空间、Torch 运算临时空间、模型/optimizer 不在这些数组预算内。内容 preflight 会预热 OS file cache，其 CPU/wall/bytes 单独记录；当前不声称 cold-cache 性能。
详细结果中的 initialization 总时长包含内容 preflight，content_verification 是其中的子项，两者不能相加。

## 与已有实验兼容

IC/FS 复用既有模型、源映射、训练 recipe、精度、pool/accumulation、checkpoint/resume、验证和报告器：

- CNN/eFUN：`dct_models.training_pls --input-backend ic|fs --baseline-store DIR`；已有 `cnn_training_baselines.sh` 和 `report_cnn_suite --additional-training` 接受这些结果。
- ViT/Swin：`training_pls.run_matrix --execution-backend ic-matched-dct|fs-matched-dct --baseline-store DIR`；保留既有 immutable run manifest、epoch `metrics.jsonl`、checkpoint 和条件检查。已有性能/收敛报告器接受 DCT reference，并保留具体 store variant。

两条正式 DataLoader 路径现在通过 `__getitems__` 每个真实 model microbatch 合并请求，再使用同一个有界 reader。重复样本/顺序/tail 正确，一批触及的相同 member 只解压一次。CPU JPEG 对照检查已覆盖两条实际 dataset 路径。正式 CNN 和 Transformer 的 consumed-batch I/O、decode 和峰值进入各自原始结果；没有把 pooled replay 的 counters 冒充正式路径测量。

诊断 replay 的新 schema 不替代已有正式实验 schema。现有条件、source/model/recipe/初始状态和 GPU 配对检查继续有效；不同 GPU 或不同训练条件的历史结果可展示，但不能直接拼成 matched speedup。旧 baseline stores/config/evidence 必须重新生成，因为此前没有内容身份或完整 label contract。

## 已运行的证据

- 用户在宿主标准 unittest discovery 中执行 **34 tests，13.418s，OK**；来源明确记在 [host-unittest-result.txt](host-unittest-result.txt)。之前的上游 `datasets` 导入冲突通过将真实 DCTNet 模型检查放在独立进程解决，Tiny 集成测试不再导入冲突的上游模型包。
- 最后消除初始化哈希对 manifest 的重复读取，直接 hash 已读取字节；[定向回归](reader-initialization-regression/result.json) 验证实际读取次数/字节与 initialization counters 相等，同时再次验证多组 decode 生命周期。该改动后九份真实模型 CPU 证据已重新生成。
- [regression-final/regressions.json](regression-final/regressions.json)：相同 34 个测试体另保存持久 fixture；10 个无需临时目录的测试用标准 unittest 执行，其余 24 个测试体显式导出输入并执行原断言。这不替代上面的标准宿主 discovery。包含最终 schedule override / inference BF16 检查。
- [regression-cuda-fix/regressions.json](regression-cuda-fix/regressions.json)：CUDA oracle 修复后的最新 fixture 导出，34 项 CPU 测试通过、新增的一项 CUDA regression 在沙箱明确 skip。校准过期负例现在使用独立 stale-calibration 文件，原始 passing calibration 保持不变。
- [evidence-v2/completion.json](evidence-v2/completion.json)：实际 MobileNet24、ResNet24、ViT-Ti，分别对 IC、IC-b、FS 执行校准和验证各十次更新；九份 `models/*/*.verified.json` 全部 passed，inputs/logits/loss/gradients/parameters/AdamW moment 最大观测差异均为 0。
- [evidence-h100/completion.json](evidence-h100/completion.json)：宿主 H100 的同样九组校准及九组验证全部通过，每组十次更新，inputs/model/state 的 atol/rtol 全部为 0，所有观测差异为 0。MobileNet24/ResNet24 使用 BF16 autocast，ViT-Ti 使用 FP32。已核对当前实现、config、store、calibration 和原始 snapshot 哈希；每个模型的初始参数状态与 CPU 证据相同。原始 JSON/日志逐字节复制归档，不改写其绝对路径。
- 真实模型使用三张脚本生成的 512×512 JPEG：一张校准、两张验证，十个 epoch 的冻结 trace，完整 from-scratch model/optimizer/decay/scheduler/RNG snapshot。不是 ImageNet 质量或性能结果。原始 config/profile/source/labels/trace、stores、校准 JSON 和 stdout/stderr 均在 `evidence-v2`。
- NativeSourceTests 从当前真实 FastLens block-major source 读取 image IDs 2、0、2，与相应真实 JPEG 的 q/Q/geometry 精确一致。
- 已有 PLS 的 pool 边界、锁定 recipe 和注册 DCT model contract 三项定向测试通过。
- [正式契约 fixture](formal-contract-fixture.json) 直接调用现有 contract builder，核验 IC/FS 保持 optimizer/model recipe、后端身份分别冻结且不可进入 headline。这是契约生成回归，不是正式训练运行。

聚合回归 fixture 中 100/120/900/300 images/s 是人工反例输入；计时注入值也是控制流检查，不是性能数据。负例 fixture 有意含合法重写但 fingerprint 过期的 NPZ。`evidence` 是缺少 jsonschema 时的未完成首次尝试，`regression-suite` 是添加最后两个契约断言前的导出，均不作为最终证据或放入 review zip。

`*.pt` 初始快照保留在本机证据目录，原始 JSON 保存其 SHA-256 和 initial model state SHA-256；review zip 不携带大 checkpoint。迁移到另一目录/机器时请重新运行复现脚本，不能仅修改已 hash 的 JSON 路径。

## 复现与下一步

2026-10-09 的首次 H100 运行在 MobileNet24/IC 校准输入阶段停止：最大绝对差为 `0.00048828125`，未完成 GPU 更新。日志和诊断保存于 [cuda-normalization-fix](cuda-normalization-fix/diagnosis.json)。原因是独立 oracle 使用 Python scalar std，而实际 normalize 使用设备 FP32 tensor std；[PyTorch 2.11 CUDA division](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu) 对 CPU scalar 除数采用 rounded reciprocal multiplication。现在 oracle 的 mean/std 显式构造为设备 FP32 tensor，仍逐通道独立计算，没有调用主路径的 organizer 或放宽容差。新增 CUDA regression；本地 CPU normalization/Mixup、正式 dataset 对照和 Tiny 十次更新通过，修复后宿主 `dct-round2-h100-20261009-205546` 的九组 GPU 校准和验证也全部通过。
[同一输入的 CPU 算术复现](cuda-normalization-fix/scalar-div-reproduction.json) 使用 FP32 direct division 和 FP64 reciprocal → FP32 multiplication，完整重现原 H100 的 79376/301056 个不同元素及上述最大差；这是故障原因诊断，不是新的 GPU 测量。

在有既有 RGB-no-more/DCTNet/FUN 树、checkpoint、jpeg2dct 和 dct_manip 扩展的完整仓库上运行。依赖版本见 `dependencies.json`；CPU 报告首次借用系统 jsonschema 4.10.3，用户随后在目标环境安装 4.26.0 并通过标准测试。现有 build 为 Release/clang++，bridge 构建命令：

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
export PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
cmake --build build --target galp_dct_storage -j2
"$PYTHON" -B -m unittest discover -s galp/benchmarks/dct_baselines/tests -t . -v
"$PYTHON" -B -m unittest \
  galp.benchmarks.training_pls.tests.test_training_pls.RecipeAndContractTests.test_ic_fs_reference_contracts_preserve_the_training_recipe \
  galp.benchmarks.training_pls.tests.test_training_pls.RecipeAndContractTests.test_standard_rgbnomore_dct_backend_has_distinct_contract_identity -v
"$PYTHON" -B galp/ImageCalulatorRe-implement/round-2-response/reproduce.py \
  --output "$HOME/tmp/dct-round2-cpu-$(date +%Y%m%d-%H%M%S)"
```

H100 的 IC/IC-b/FS 小型输入数值校准和十次更新已通过。以下命令用于复现；同一代码无需重复运行。脚本只声明零容差，首次失败即停止；校准通过才执行独立图像的验证，不自动重试或放宽限值：

```bash
cd /home/tangyuxin/gfastlanes/FastLanes
export PYTHON=/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python
export CUDA_VISIBLE_DEVICES=GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76
"$PYTHON" -B -m unittest \
  galp.benchmarks.dct_baselines.tests.test_round2.GoldenOrganization.test_cuda_normalization_and_mixup_match_independent_fp32_oracle -v
bash galp/ImageCalulatorRe-implement/round-2-response/run_cuda_gates.sh \
  "$HOME/tmp/dct-round2-h100-$(date +%Y%m%d-%H%M%S)"
```

成功信号为 `Passed: 3 models x 3 store variants x 10 CUDA updates`。输出目录包含九份 `.verified.json` 和对应校准/日志，host 重新生成的 snapshot 包含可见 CUDA 设备 RNG。

下一步先补齐同一真实 source/trace 上 FastLens 与基线的 CUDA 数值对照，再完成 G4 计量审查，包括正式 DataLoader 未消费的 lookahead I/O、测量窗口和资源预算。合成 fixture 不带真实 `galp_image_id`，不能用于 FastLens CUDA 对照；真实共同源的 production recipe/provenance 仍待确认。G4 通过后，按 round-2 顺序运行三个代表模型的完整验证集和训练 epoch。当前尚未完成 FastLens 的这项 GPU 对照、G4、G5、完整 50K/epoch、共享消费者或第二 GPU；所有当前结果保持 `headline_eligible=false`。
