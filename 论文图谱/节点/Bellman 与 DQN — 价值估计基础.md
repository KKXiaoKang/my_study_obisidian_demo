---
type: concept
node_id: bellman
topic: rl
status: "原笔记归纳"
updated: 2026-10-27
tags: [论文图谱, rl, concept]
---

# Bellman 与 DQN — 价值估计基础

## 节点摘要

**概念 · 03 价值学习与强化学习**  
为什么未来回报可以通过自举目标迭代估计？

Bellman 方程联系当前价值与下一状态价值；Q-learning 学动作价值，DQN 用神经网络近似 Q，并借助经验回放和目标网络稳定训练。

## 训练、推理与数据流

分清 V(s)、Q(s,a)、当前网络与目标网络；用小型表格例子计算 TD target，再对应到神经网络损失。

## 证据与适用边界

**整理状态：原笔记归纳。**

DQN 不是把所有连续状态穷举完的 Q 表。三组同内容笔记在仓库中重复存在，本节点统一引用但保留原文件。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 基础 | [[论文图谱/节点/MDP 与 SMDP — 状态、回报与时间尺度\|MDP 与 SMDP — 状态、回报与时间尺度]] | 状态动作与价值递推 |
| 本节点 → | 扩展 | [[论文图谱/节点/Critic 谱系 — DQN 到连续控制\|Critic 谱系 — DQN 到连续控制]] | 比较不同 bootstrap 目标 |
| 本节点 → | 基础 | [[论文图谱/节点/Model-based RL — 模型用于哪里\|Model-based RL — 模型用于哪里]] | 模型是否参与转移预测和规划 |

## 原始笔记

- [[RobotRL/Q-learning DQN/DQN 变换表|DQN 变换表]]
- [[RobotRL/Q-learning DQN/Q-learing变化表|Q-learing变化表]]
- [[RobotRL/Q-learning DQN/q-learning 方法一览|q-learning 方法一览]]
- [[RobotRL/Q-learning DQN/公式探索/DQN_vs_贝尔曼.md|DQN_vs_贝尔曼.md]]
- [[RobotRL/Q-learning DQN/公式探索/Deep Q Net-work|Deep Q Net-work]]
- [[RobotRL/Q-learning DQN/公式探索/贝尔曼方程 - Bellman Optimality Equation|贝尔曼方程 - Bellman Optimality Equation]]
- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/公式探索/DQN_vs_贝尔曼.md|DQN_vs_贝尔曼.md]]
- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/公式探索/Deep Q Net-work|Deep Q Net-work]]
- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/公式探索/贝尔曼方程 - Bellman Optimality Equation|贝尔曼方程 - Bellman Optimality Equation]]

## 下一步阅读或验证

沿 Critic 谱系比较 max、双 Q、actor 替代动作搜索与熵正则。

返回 [[论文图谱/主题/03 价值学习与强化学习|03 价值学习与强化学习]] · [[00-论文知识图谱|知识图谱首页]]
