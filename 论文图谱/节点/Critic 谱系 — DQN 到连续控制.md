---
type: concept
node_id: critic
topic: rl
status: "原笔记归纳"
updated: 2026-10-27
tags: [论文图谱, rl, concept]
---

# Critic 谱系 — DQN 到连续控制

## 节点摘要

**概念 · 03 价值学习与强化学习**  
目标网络、双 Q 和 actor 分别解决什么问题？

按“动作选择—价值评估—目标构造”比较 DQN、Double DQN、DDPG、TD3 和 SAC，避免把所有双网络机制混为一谈。

## 训练、推理与数据流

为每种算法写 online 与 target 网络输入输出、更新频率及 TD target；原笔记的扩展表可作阅读入口。

## 证据与适用边界

**整理状态：原笔记归纳。**

原表中早期 DQN 的年份与目标网络叙述需要对应具体版本核对。博客的可扩展性观点也不等于已证明所有 Q-learning 都不可扩展。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 扩展 | [[论文图谱/节点/Bellman 与 DQN — 价值估计基础\|Bellman 与 DQN — 价值估计基础]] | 比较不同 bootstrap 目标 |
| 本节点 → | 基础 | [[论文图谱/节点/SAC — 最大熵 Actor-Critic\|SAC — 最大熵 Actor-Critic]] | actor、双 Q 与熵正则分工 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/Deep RL Critic 谱系总览表|Deep RL Critic 谱系总览表]]
- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/常见QA|常见QA]]
- [[RobotRL/Q-learning DQN/探索底层文章 - Q-learning is not yet scalable|探索底层文章 - Q-learning is not yet scalable]]

## 下一步阅读或验证

用相同符号重写 TD target，并将算法定义与本地代码选择分列。

返回 [[论文图谱/主题/03 价值学习与强化学习|03 价值学习与强化学习]] · [[00-论文知识图谱|知识图谱首页]]
