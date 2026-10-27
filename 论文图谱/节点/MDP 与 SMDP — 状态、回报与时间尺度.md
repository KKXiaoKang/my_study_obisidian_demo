---
type: concept
node_id: mdp
topic: rl
status: "原笔记归纳"
updated: 2026-10-27
tags: [论文图谱, rl, concept]
---

# MDP 与 SMDP — 状态、回报与时间尺度

## 节点摘要

**概念 · 03 价值学习与强化学习**  
控制决策如何形式化，动作持续多步时该怎样描述？

MDP 用状态、动作、转移和奖励描述逐步决策；SMDP 允许决策之间经历可变时长，适合时间扩展动作和分层技能。

## 训练、推理与数据流

记录 observation 与真实 state 的区别；动作块需要明确执行时长、累计回报和下一决策点。

## 证据与适用边界

**整理状态：原笔记归纳。**

图像观测通常不直接满足完全可观测假设；动作块长度变化时折扣不能仍无条件当单步处理。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 本节点 → | 基础 | [[论文图谱/节点/Bellman 与 DQN — 价值估计基础\|Bellman 与 DQN — 价值估计基础]] | 状态动作与价值递推 |
| 本节点 → | 基础 | [[论文图谱/节点/Q-chunking — 动作块空间中的 RL\|Q-chunking — 动作块空间中的 RL]] | 多步决策和块末 bootstrap |

## 原始笔记

- [[RobotRL/Agent - base - MDP马尔可夫决策过程|Agent - base - MDP马尔可夫决策过程]]
- [[RobotRL/Agent - explore - SMDP半马尔可夫决策过程|Agent - explore - SMDP半马尔可夫决策过程]]

## 下一步阅读或验证

从逐步 Bellman backup 推到 k 步 backup，再理解 Q-chunking 的动作空间。

返回 [[论文图谱/主题/03 价值学习与强化学习|03 价值学习与强化学习]] · [[00-论文知识图谱|知识图谱首页]]
