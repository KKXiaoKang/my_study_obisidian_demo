---
type: paper
node_id: rlt
topic: posttrain
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, posttrain, paper]
---

# RL Token — 轻量强化学习接口

## 节点摘要

**论文 · 04 VLA 后训练与奖励**  
如何复用大 VLA 的知识，同时降低在线 RL 的模型更新成本？

让 VLA 暴露紧凑的任务相关 readout 表示，再训练小型 actor-critic head 改进动作，并保留与基础 VLA 的锚定关系。

## 训练、推理与数据流

将表示接口的适配与在线 actor-critic 优化分开记录。原笔记进一步讨论特征重建模块和熵项，需要对应论文版本核对。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

不能把 RLT 简化成“任意小 Transformer 重建所有 VLM 特征”。表示是否保留任务信息、如何约束偏离基础策略都很关键。

**对应文献**：RL Token: Bootstrapping Online RL with Vision-Language-Action Models  
**作者**：Charles Xu et al.  
**首次预印本年份**：2026（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2604.23073)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 基础 | [[论文图谱/节点/SAC — 最大熵 Actor-Critic\|SAC — 最大熵 Actor-Critic]] | 轻量 actor-critic 的价值学习 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/pi系列 -RL Token 单独建一个Transformer用于重建VLM的输出嵌入feature，之后RL训练用此模型作为基座，加快RL训练效率|pi系列 -RL Token 单独建一个Transformer用于重建VLM的输出嵌入feature，之后RL训练用此模型作为基座，加快RL训练效率]]

## 下一步阅读或验证

比较冻结特征、可适配 RL token 和端到端更新，在相同真实交互预算下测精度与速度。

返回 [[论文图谱/主题/04 VLA 后训练与奖励|04 VLA 后训练与奖励]] · [[00-论文知识图谱|知识图谱首页]]
