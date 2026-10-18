# FastLens baseline agent pack

这份开发包用于给 FastLens 增加 Image Calculator 和 Frequency-Store 的有证据复现与 matched DCT 对照。它包含实施计划和建议合同，没有运行中的基线代码或新实验结果。

## 建议阅读顺序

1. `IMPLEMENTATION_PLAN.md`：完整研究依据、实现机制、adaptation 边界、数值验收与实验方案。
2. `AGENT_BRIEF.md`：可直接作为开发 agent 的起始任务。
3. `contracts/matched.example.yaml`：需要在实际仓库接入时解析并锁定的示例配置。
4. `contracts/result.schema.json`：建议的正式结果结构。
5. `tasks/task_graph.json`：任务依赖与验收门。

示例配置的 null 字段有意保留，表示本次未提供实际仓库/data/profile/checkpoint；运行前必须解析，不是可以默默跳过的可选项。计划中提出的CLI名称属于待实现接口。

## 三项最重要的区别

- FS 原生 B32/Q50 与 JPEG B8 的15条反对角线适配不同。
- IC 原生三通道DCT mosaic 与当前FastLens模型输入不同；全64系数不是原生B8,k8三角候选。
- 相同模型/最终tensor的系统对比、原生机制验证、准确率约束下的部署对比分开报告。

来源以用户提供的三篇PDF为准，完整文件hash、论文节号及已核查公开原始链接见实施计划末尾。未读到的TechRep/slides未用来推断实现事实。
