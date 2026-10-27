---
type: proposal
node_id: ood
topic: practice
status: "方案归纳；未新增实验验证"
updated: 2026-10-27
tags: [论文图谱, practice, proposal]
---

# OOD Trigger — 不确定性与恢复行为

## 节点摘要

**研究设想 · 08 工程实验与评测**  
策略何时应该重新观测、重采样、缩短动作块或降低速度？

原笔记提出多次动作采样差异、ensemble 分歧与动态一致性三类信号，并讨论触发后的行为调整。

## 训练、推理与数据流

校准触发分数与真实失败的关系，记录阈值、误报漏报和恢复策略；与关节限位、碰撞约束分别管理。

## 证据与适用边界

**整理状态：方案归纳；未新增实验验证。**

多样动作可能都是合理解，因此采样方差不能直接当失败概率；单模型也可能在 OOD 下自信出错。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 双向比较 | 辨析 | [[论文图谱/节点/Uncertainty Weighting — 多任务损失加权\|Uncertainty Weighting — 多任务损失加权]] | 任务级尺度与样本级置信度不同 |
| 本节点 → | 评测 | [[论文图谱/节点/Benchmark 与消融 — 让比较可解释\|Benchmark 与消融 — 让比较可解释]] | 检测与恢复共同验收 |

## 原始笔记

- [[RobotVLA/OOD情况下如何定义 trigger进行重采样及行为降级（缩短 horizon，减小动作幅度， 降低末端速度）|OOD情况下如何定义 trigger进行重采样及行为降级（缩短 horizon，减小动作幅度， 降低末端速度）]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/Bimanual Cross-Attention|Bimanual Cross-Attention]]

## 下一步阅读或验证

在明确的分布变化集上测检测与恢复效果，并报告延迟和任务完成率。

返回 [[论文图谱/主题/08 工程实验与评测|08 工程实验与评测]] · [[00-论文知识图谱|知识图谱首页]]
