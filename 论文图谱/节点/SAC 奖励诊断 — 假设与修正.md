---
type: experiment
node_id: rewarddiag
topic: rl
status: "历史实验记录；未重新复现"
updated: 2026-10-27
tags: [论文图谱, rl, experiment]
---

# SAC 奖励诊断 — 假设与修正

## 节点摘要

**实验 · 03 价值学习与强化学习**  
为什么某些 shaped reward 设置下策略进步缓慢，怎样定位原因？

原记录先提出熵与任务回报数量级的解释，随后在“计算对比说明”中修正：actor 更新更直接依赖 ∇aQ 与熵梯度，而非 Q 绝对值。

## 训练、推理与数据流

先校验 tanh 高斯的 log-prob/Jacobian，再检查 reward scale、shift、温度、终止条件和实际动作变化。将每个解释转为可观测指标。

## 证据与适用边界

**整理状态：历史实验记录；未重新复现。**

这些是待实验区分的机制，不能写成“SAC 无法处理多维正奖励”的普遍定论。旧数字和口头结论不算当前复现结果。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 诊断 | [[论文图谱/节点/SAC — 最大熵 Actor-Critic\|SAC — 最大熵 Actor-Critic]] | 检验尺度、熵与动作梯度 |
| 双向比较 | 对比 | [[论文图谱/节点/PPO — 受约束的策略更新\|PPO — 受约束的策略更新]] | 同预算与同奖励下对照 |

## 原始笔记

- [[RobotRL/Hil-Serl/☆ - SAC recipe experiments log 对比 baseline PPO|☆ - SAC recipe experiments log 对比 baseline PPO]]
- [[RobotRL/Reward设计/default joint reward - 设计思路|default joint reward - 设计思路]]
- [[RobotRL/Reward设计/manipulation reward 设计|manipulation reward 设计]]
- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/Reward设计 - ppo vs SAC 说明|Reward设计 - ppo vs SAC 说明]]
- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/Reward设计 - 计算对比说明|Reward设计 - 计算对比说明]]
- [[Robotmanipulation - opt_VA/YouKnow3M/AMP & locomotion/上楼梯奖励一览|上楼梯奖励一览]]

## 下一步阅读或验证

做单因素实验，固定种子与预算，同时记录回报、成功率、α、熵和动作梯度统计。

返回 [[论文图谱/主题/03 价值学习与强化学习|03 价值学习与强化学习]] · [[00-论文知识图谱|知识图谱首页]]
