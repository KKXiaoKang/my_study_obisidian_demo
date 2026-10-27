---
type: method
node_id: hilserl
topic: rl
status: "原笔记归纳"
updated: 2026-10-27
tags: [论文图谱, rl, method]
---

# HIL-SERL — 人类干预与样本高效 RL

## 节点摘要

**方法 · 03 价值学习与强化学习**  
怎样把示教、人类接管和在线学习组成一个可运行闭环？

原笔记重点是 actor／learner 分工、在线与离线 replay 混合、连续动作与离散夹爪处理，以及 SAC 训练过程。

## 训练、推理与数据流

采集、干预、存储、采样与更新分别记录；在线／离线 buffer 比例和成功判定是重要实验变量。

## 证据与适用边界

**整理状态：原笔记归纳。**

框架流程说明不等于某台机器人已经完成物理任务；混合动作设计和温度更新应按实现版本核对。

原始来源：[论文／作者项目入口](https://github.com/rail-berkeley/hil-serl)。链接由已有笔记或书目线索整理，未逐项复核正文。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 应用 | [[论文图谱/节点/SAC — 最大熵 Actor-Critic\|SAC — 最大熵 Actor-Critic]] | 在线学习与示教混合 |
| 双向比较 | 关联 | [[论文图谱/节点/ConRFT — 一致性策略的强化微调\|ConRFT — 一致性策略的强化微调]] | 人类干预的在线学习闭环 |
| → 本节点 | 工程 | [[论文图谱/节点/LeRobot 与 RL 工程链路\|LeRobot 与 RL 工程链路]] | 采样、回放与 actor-learner |
| 双向比较 | 关联 | [[论文图谱/节点/ViTAL — 视觉与触觉残差学习\|ViTAL — 视觉与触觉残差学习]] | 残差和人类干预的阅读线索 |

## 原始笔记

- [[RobotRL/Hil-Serl/lerobot - Hil-Serl pipeline说明|lerobot - Hil-Serl pipeline说明]]
- [[RobotRL/Hil-Serl/☆ - SAC recipe experiments log 对比 baseline PPO|☆ - SAC recipe experiments log 对比 baseline PPO]]
- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/Reward设计 - ppo vs SAC 说明|Reward设计 - ppo vs SAC 说明]]
- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/Reward设计 - 计算对比说明|Reward设计 - 计算对比说明]]
- [[Robotmanipulation - opt_VA/YouKnow3M/Hil-Serl & SAC/常见QA|常见QA]]

## 下一步阅读或验证

把算法接口与实验验收拆开，比较干预次数、训练时长、成功率和失败恢复。

返回 [[论文图谱/主题/03 价值学习与强化学习|03 价值学习与强化学习]] · [[00-论文知识图谱|知识图谱首页]]
