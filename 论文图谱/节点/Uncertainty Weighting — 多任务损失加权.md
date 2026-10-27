---
type: paper
node_id: uncertainty
topic: foundation
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, foundation, paper]
---

# Uncertainty Weighting — 多任务损失加权

## 节点摘要

**论文 · 01 表征与生成基础**  
位置、旋转、夹爪等任务量纲和噪声不同，损失权重怎样确定？

从任务似然出发，用可学习的同方差不确定性调节各项权重，并保留对不确定性参数的正则项，避免任意把任务权重压到零。

## 训练、推理与数据流

联合优化模型参数与各任务的尺度参数；实现中的 logσ 或 logσ² 会改变公式系数，必须和代码定义对齐。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

这是任务级尺度建模，不等于样本级 OOD 置信度；重加权也不能保证消除共享特征的梯度冲突。

**对应文献**：Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics  
**作者**：Alex Kendall et al.  
**首次预印本年份**：2017（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/1705.07115)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 本节点 → | 应用 | [[论文图谱/节点/动作头分解 — 尺度、共享与双臂协调\|动作头分解 — 尺度、共享与双臂协调]] | 不同动作任务的尺度加权 |
| 双向比较 | 辨析 | [[论文图谱/节点/OOD Trigger — 不确定性与恢复行为\|OOD Trigger — 不确定性与恢复行为]] | 任务级尺度与样本级置信度不同 |

## 原始笔记

- [[RobotVLA/模型optimize/MLP multi head v1.0|MLP multi head v1.0]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/GROOT N1.5 action expert改动一览|GROOT N1.5 action expert改动一览]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/Uncertainty-Weighted Multi-Task Loss|Uncertainty-Weighted Multi-Task Loss]]

## 下一步阅读或验证

和等权重、手工权重比较，记录各项原始 loss、有效权重与最终任务成功率。

返回 [[论文图谱/主题/01 表征与生成基础|01 表征与生成基础]] · [[00-论文知识图谱|知识图谱首页]]
