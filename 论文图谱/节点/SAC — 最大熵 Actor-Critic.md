---
type: paper
node_id: sac
topic: rl
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, rl, paper]
---

# SAC — 最大熵 Actor-Critic

## 节点摘要

**论文 · 03 价值学习与强化学习**  
如何复用交互数据，同时优化回报与探索？

SAC 采用 off-policy actor-critic 与最大熵目标。当前常见实现包含随机 actor、双 Q、目标网络与温度参数。

## 训练、推理与数据流

从 replay buffer 采样，更新 critic、actor、温度及目标网络；保存和恢复时还需对应观测归一化状态。早期 SAC 论文与后续常见实现有版本区别。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

Q 的绝对数值不能单独解释 actor 是否学得动；需检查动作方向梯度、熵项、reward 尺度与终止语义。

**对应文献**：Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor  
**作者**：Tuomas Haarnoja et al.  
**首次预印本年份**：2018（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/1801.01290)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 基础 | [[论文图谱/节点/Critic 谱系 — DQN 到连续控制\|Critic 谱系 — DQN 到连续控制]] | actor、双 Q 与熵正则分工 |
| 双向比较 | 对比 | [[论文图谱/节点/PPO — 受约束的策略更新\|PPO — 受约束的策略更新]] | 数据复用与更新约束 |
| 双向比较 | 对比 | [[论文图谱/节点/Q-chunking — 动作块空间中的 RL\|Q-chunking — 动作块空间中的 RL]] | 单步与动作块价值定义 |
| 本节点 → | 应用 | [[论文图谱/节点/HIL-SERL — 人类干预与样本高效 RL\|HIL-SERL — 人类干预与样本高效 RL]] | 在线学习与示教混合 |
| 本节点 → | 诊断 | [[论文图谱/节点/SAC 奖励诊断 — 假设与修正\|SAC 奖励诊断 — 假设与修正]] | 检验尺度、熵与动作梯度 |
| 本节点 → | 基础 | [[论文图谱/节点/RL Token — 轻量强化学习接口\|RL Token — 轻量强化学习接口]] | 轻量 actor-critic 的价值学习 |

## 原始笔记

- [[RobotRL/SAC/SAC - (actor-Critic 架构)|SAC - (actor-Critic 架构)]]
- [[RobotRL/SAC/SAC - `RunningMeanStd` 观测归一化工具|SAC - `RunningMeanStd` 观测归一化工具]]
- [[RobotRL/SAC/SAC - load or save model|SAC - load or save model]]
- [[RobotRL/SAC/SAC - train and learn|SAC - train and learn]]
- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/常见QA|常见QA]]
- [[RobotRL/Hil-Serl/lerobot - Hil-Serl pipeline说明|lerobot - Hil-Serl pipeline说明]]
- [[RobotRL/Hil-Serl/☆ - SAC recipe experiments log 对比 baseline PPO|☆ - SAC recipe experiments log 对比 baseline PPO]]

## 下一步阅读或验证

把 reward 诊断、归一化和 ACT-SAC 改造放在独立实验节点，避免改写 SAC 的一般结论。

返回 [[论文图谱/主题/03 价值学习与强化学习|03 价值学习与强化学习]] · [[00-论文知识图谱|知识图谱首页]]
