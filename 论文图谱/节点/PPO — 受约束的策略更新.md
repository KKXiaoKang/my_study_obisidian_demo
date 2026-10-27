---
type: paper
node_id: ppo
topic: rl
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, rl, paper]
---

# PPO — 受约束的策略更新

## 节点摘要

**论文 · 03 价值学习与强化学习**  
如何在复用一批交互样本时，减少策略更新过大的问题？

用新旧策略概率比和代理目标更新策略；常见 clipped PPO 限制有利方向上的过度变化，并配合价值估计和优势估计。

## 训练、推理与数据流

采集当前策略数据，计算优势，多轮 minibatch 更新；必须明确 log-prob 对应的是单步动作还是生成过程中的变量。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

clipping 不是性能改善的绝对保证，也不能仅凭某个任务结果声称 PPO 总优于 SAC 或 GRPO。

**对应文献**：Proximal Policy Optimization Algorithms  
**作者**：John Schulman et al.  
**首次预印本年份**：2017（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/1707.06347)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 双向比较 | 对比 | [[论文图谱/节点/SAC — 最大熵 Actor-Critic\|SAC — 最大熵 Actor-Critic]] | 数据复用与更新约束 |
| 双向比较 | 对比 | [[论文图谱/节点/SAC 奖励诊断 — 假设与修正\|SAC 奖励诊断 — 假设与修正]] | 同预算与同奖励下对照 |
| 本节点 → | 基础 | [[论文图谱/节点/DPPO — 扩散策略的策略梯度优化\|DPPO — 扩散策略的策略梯度优化]] | 概率比与优势估计 |

## 原始笔记

- [[RobotRL/Hil-Serl/☆ - SAC recipe experiments log 对比 baseline PPO|☆ - SAC recipe experiments log 对比 baseline PPO]]
- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/Reward设计 - ppo vs SAC 说明|Reward设计 - ppo vs SAC 说明]]
- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/model-free RL 和 model-base RL|model-free RL 和 model-base RL]]
- [[RobotRL/Diffusion-RL/DPPO - Diffusion Policy Policy Optimization|DPPO - Diffusion Policy Policy Optimization]]

## 下一步阅读或验证

先对齐相同奖励和交互预算，再读 DPPO 怎样为扩散生成过程定义策略梯度。

返回 [[论文图谱/主题/03 价值学习与强化学习|03 价值学习与强化学习]] · [[00-论文知识图谱|知识图谱首页]]
