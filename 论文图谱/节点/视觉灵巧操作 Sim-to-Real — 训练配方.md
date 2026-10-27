---
type: paper
node_id: sim2real
topic: rl
status: "原笔记归纳；原文细节待核对"
updated: 2026-10-27
tags: [论文图谱, rl, paper]
---

# 视觉灵巧操作 Sim-to-Real — 训练配方

## 节点摘要

**论文 · 03 价值学习与强化学习**  
怎样把仿真中的视觉灵巧操作策略迁移到真实人形机器人？

原笔记整理了相机对齐、真实到仿真调优、奖励设计、样本高效学习，以及稀疏和稠密对象表示。

## 训练、推理与数据流

分别记录 actor 观测、critic 特权信息和真实部署可获得的输入；将仿真随机化与真机标定区分。

## 证据与适用边界

**整理状态：原笔记归纳；原文细节待核对。**

特权状态只可在允许的训练分支使用；仿真成功不等于真机验收，原性能需对齐任务和硬件。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2502.20396)。链接由已有笔记或书目线索整理，未逐项复核正文。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 工程 | [[论文图谱/节点/仿真资产与工具 — 从重建到实验环境\|仿真资产与工具 — 从重建到实验环境]] | 仿真与现实输入匹配 |
| → 本节点 | 基础 | [[论文图谱/节点/标定与状态估计 — 几何可信度\|标定与状态估计 — 几何可信度]] | 几何和相机误差影响迁移 |
| 双向比较 | 对比 | [[论文图谱/节点/ViTAL — 视觉与触觉残差学习\|ViTAL — 视觉与触觉残差学习]] | 接触感知与现实泛化 |

## 原始笔记

- [[RobotRL/☆  基于视觉的Sim2real的抓取manipulation RL/Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids|Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids]]
- [[RobotSim/isaac-sim碰撞对比/碰撞对比表格|碰撞对比表格]]
- [[RobotRL/Reward设计/manipulation reward 设计|manipulation reward 设计]]

## 下一步阅读或验证

按视觉误差、动力学误差和控制延迟划分迁移失败类型。

返回 [[论文图谱/主题/03 价值学习与强化学习|03 价值学习与强化学习]] · [[00-论文知识图谱|知识图谱首页]]
