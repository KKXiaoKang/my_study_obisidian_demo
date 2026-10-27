---
type: paper
node_id: rotation
topic: foundation
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, foundation, paper]
---

# Rotation 6D — 连续旋转表示

## 节点摘要

**论文 · 01 表征与生成基础**  
怎样减少欧拉角或四元数表示带来的连续性问题？

用两个三维向量表示旋转，通过正交化恢复旋转矩阵。论文讨论连续表示的拓扑性质，并提出 5D／6D 表示。

## 训练、推理与数据流

网络输出 6D 向量，部署时映射到 SO(3)。评估分别记录表示空间误差与恢复后的旋转误差；归一化统计应与动作定义一致。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

连续表示不保证处处梯度平坦，也不保证 6D MSE 等价于 SO(3) 测地距离。接近共线的向量仍需数值处理。原“Normalization”笔记主要谈相对位置尺度，不能只按标题理解。

**对应文献**：On the Continuity of Rotation Representations in Neural Networks  
**作者**：Yi Zhou et al.  
**首次预印本年份**：2018（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/1812.07035)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 本节点 → | 应用 | [[论文图谱/节点/动作头分解 — 尺度、共享与双臂协调\|动作头分解 — 尺度、共享与双臂协调]] | 旋转表示影响动作头与损失 |
| 本节点 → | 基础 | [[论文图谱/节点/VR Retarget — 人体到机器人目标映射\|VR Retarget — 人体到机器人目标映射]] | 坐标与姿态表示一致性 |

## 原始笔记

- [[RobotRL/描述物体姿态 - 让Network学习到平坦可微的表示/On the Continuity of Rotation Representations in Neural Networks|On the Continuity of Rotation Representations in Neural Networks]]
- [[RobotVLA/EEF Train trick/Rotation MSE on 6D Vectors|Rotation MSE on 6D Vectors]]
- [[RobotVLA/EEF Train trick/Rotation(6D) Normalization way|Rotation(6D) Normalization way]]

## 下一步阅读或验证

固定数据和模型，对比 6D MSE、旋转测地误差，以及绝对／相对动作统计。

返回 [[论文图谱/主题/01 表征与生成基础|01 表征与生成基础]] · [[00-论文知识图谱|知识图谱首页]]
