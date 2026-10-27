---
type: paper
node_id: qchunk
topic: rl
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, rl, paper]
---

# Q-chunking — 动作块空间中的 RL

## 节点摘要

**论文 · 03 价值学习与强化学习**  
如何利用离线时序行为改善长程稀疏奖励任务中的探索与 TD 学习？

在 chunked action space 中进行强化学习，让价值估计对应一段动作，结合时间一致的离线行为与多步回报。

## 训练、推理与数据流

明确 Q(s,a_{t:t+k})、块内累计奖励、终止掩码和块末 bootstrap。原笔记中的联合 log-prob 推导是特定实现说明，应另行核对。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

只把逐步 log-prob 相加，或只换成 ACT actor，并不能证明实现了原论文。必须检查 critic 是否真正评价动作块及 replay 的采样语义。

**对应文献**：Reinforcement Learning with Action Chunking  
**作者**：Qiyang Li et al.  
**首次预印本年份**：2025（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2507.07969)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 基础 | [[论文图谱/节点/MDP 与 SMDP — 状态、回报与时间尺度\|MDP 与 SMDP — 状态、回报与时间尺度]] | 多步决策和块末 bootstrap |
| 双向比较 | 关联 | [[论文图谱/节点/ACT — 动作块模仿学习\|ACT — 动作块模仿学习]] | 共享动作块思想，训练目标不同 |
| 双向比较 | 对比 | [[论文图谱/节点/SAC — 最大熵 Actor-Critic\|SAC — 最大熵 Actor-Critic]] | 单步与动作块价值定义 |

## 原始笔记

- [[RobotRL/Q-chunk ACT actor/2025_08_29 - 改进说明|2025_08_29 - 改进说明]]
- [[RobotRL/Q-chunk ACT actor/Q-chunking 联合概率公式说明|Q-chunking 联合概率公式说明]]
- [[RobotRL/Q-chunk ACT actor/mlp actor 2 act actor - 架构升级说明|mlp actor 2 act actor - 架构升级说明]]
- [[RobotRL/Agent - explore - SMDP半马尔可夫决策过程|Agent - explore - SMDP半马尔可夫决策过程]]

## 下一步阅读或验证

对照本地 ACT-SAC 改造笔记列出差异，再测试末尾不足一个 chunk、提前终止和动作块执行。

返回 [[论文图谱/主题/03 价值学习与强化学习|03 价值学习与强化学习]] · [[00-论文知识图谱|知识图谱首页]]
