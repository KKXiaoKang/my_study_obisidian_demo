---
type: concept
node_id: bench
topic: practice
status: "原笔记归纳"
updated: 2026-10-27
tags: [论文图谱, practice, concept]
---

# Benchmark 与消融 — 让比较可解释

## 节点摘要

**概念 · 08 工程实验与评测**  
怎样区分模型进步与数据、环境或评测设置带来的变化？

把资产、场景、任务／episode、策略和评测器分层；按场景、对象、任务结构、时序与 embodiment 定义泛化。

## 训练、推理与数据流

固定数据划分和预算，记录随机种子、成功定义、失败类型及不确定性；消融回答组件贡献，探针回答表征可读性。

## 证据与适用边界

**整理状态：原笔记归纳。**

不同预训练规模、任务分布与观测输入下的成功率不能直接排行；单次 rollout 和代码可运行都不代表可靠收益。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 评测 | [[论文图谱/节点/动作头分解 — 尺度、共享与双臂协调\|动作头分解 — 尺度、共享与双臂协调]] | 消融分头、加权和协调 |
| → 本节点 | 评测 | [[论文图谱/节点/OOD Trigger — 不确定性与恢复行为\|OOD Trigger — 不确定性与恢复行为]] | 检测与恢复共同验收 |
| 本节点 → | 评测 | [[论文图谱/节点/JEPA-WAM — 联合嵌入世界动作模型\|JEPA-WAM — 联合嵌入世界动作模型]] | 同训练预算比较消融和泛化 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/Benchmark/设计方式|设计方式]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/JEPA-WAM 补充 - 消融实验清单 与「探针 probe」到底是什么|JEPA-WAM 补充 - 消融实验清单 与「探针 probe」到底是什么]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/Exploring the Limits of Vision-Language-ActionManipulation in Cross-task Generalization|Exploring the Limits of Vision-Language-ActionManipulation in Cross-task Generalization]]
- [[RobotVLA/physicalintelligence/Emergence of Human to Robot Transfer in VLAs|Emergence of Human to Robot Transfer in VLAs]]

## 下一步阅读或验证

每个研究提案绑定一个对照表与验收指标，再填写结果。

返回 [[论文图谱/主题/08 工程实验与评测|08 工程实验与评测]] · [[00-论文知识图谱|知识图谱首页]]
