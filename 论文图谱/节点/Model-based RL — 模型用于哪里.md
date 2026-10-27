---
type: concept
node_id: modelbased
topic: rl
status: "原笔记归纳"
updated: 2026-10-27
tags: [论文图谱, rl, concept]
---

# Model-based RL — 模型用于哪里

## 节点摘要

**概念 · 03 价值学习与强化学习**  
学习环境模型与直接学习策略／价值之间有什么边界？

按模型参与决策的位置区分：模型生成训练数据、潜空间想象、在线规划，或仅作为辅助表征。原笔记讨论了 MBPO、Dreamer 与 TD-MPC2。

## 训练、推理与数据流

检查是否预测转移和奖励、是否在模型中 rollout、是否对候选动作进行搜索；这些条件决定方法定位。

## 证据与适用边界

**整理状态：原笔记归纳。**

“带 world model 的网络”并不自动等于部署时 model-based planning。辅助预测损失也不能单独证明因果动力学已学会。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 基础 | [[论文图谱/节点/Bellman 与 DQN — 价值估计基础\|Bellman 与 DQN — 价值估计基础]] | 模型是否参与转移预测和规划 |
| 本节点 → | 应用 | [[论文图谱/节点/VLA-RFT — 世界模拟器中的强化微调\|VLA-RFT — 世界模拟器中的强化微调]] | 世界模型作为 rollout 环境 |
| 双向比较 | 辨析 | [[论文图谱/节点/BFM-Zero 与 FB-CPR — 待拆分阅读入口\|BFM-Zero 与 FB-CPR — 待拆分阅读入口]] | 先核对行为模型目标，不能按名称分类 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/model-free RL 和 model-base RL|model-free RL 和 model-base RL]]

## 下一步阅读或验证

与 WAM 节点对照模型输出、训练用途与部署用途。

返回 [[论文图谱/主题/03 价值学习与强化学习|03 价值学习与强化学习]] · [[00-论文知识图谱|知识图谱首页]]
