---
type: paper
node_id: vlarft
topic: posttrain
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, posttrain, paper]
---

# VLA-RFT — 世界模拟器中的强化微调

## 节点摘要

**论文 · 04 VLA 后训练与奖励**  
真实交互成本高时，能否用数据训练的世界模型提供策略练习环境？

学习动作条件视觉模拟器，在模型中进行 rollout，并利用任务目标参考形成轨迹奖励。

## 训练、推理与数据流

先检验模拟器对动作和未来观测的响应，再在其中优化策略，最后到独立环境验证。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

策略可能利用世界模型误差获得虚高奖励；模型内成功必须与真实环境或可信外部仿真验证分开。

**对应文献**：VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators  
**作者**：作者名单见原文  
**首次预印本年份**：2025（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2510.00406)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 应用 | [[论文图谱/节点/Model-based RL — 模型用于哪里\|Model-based RL — 模型用于哪里]] | 世界模型作为 rollout 环境 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/VLA need RL, not just SFT|VLA need RL, not just SFT]]

## 下一步阅读或验证

对比模型预测误差与策略偏离程度，测模型内外的收益一致性。

返回 [[论文图谱/主题/04 VLA 后训练与奖励|04 VLA 后训练与奖励]] · [[00-论文知识图谱|知识图谱首页]]
