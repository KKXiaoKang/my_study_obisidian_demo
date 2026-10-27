---
type: paper
node_id: conrft
topic: posttrain
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, posttrain, paper]
---

# ConRFT — 一致性策略的强化微调

## 节点摘要

**论文 · 04 VLA 后训练与奖励**  
有限且质量不一的示教怎样启动接触任务中的 VLA 在线改进？

将离线与在线微调结合在一致性策略训练框架中：离线融合 BC 与 Q-learning，在线结合人类干预进行改进。

## 训练、推理与数据流

先用示教稳定策略与价值，再进入在线阶段；比较时记录相同初始策略、干预策略和任务环境。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

一致性策略与普通 flow head 的接口并不完全相同；论文任务上的分钟数和成功率不能直接变成自己系统的预期。

**对应文献**：ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy  
**作者**：作者名单见原文  
**首次预印本年份**：2025（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2502.05450)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 双向比较 | 关联 | [[论文图谱/节点/HIL-SERL — 人类干预与样本高效 RL\|HIL-SERL — 人类干预与样本高效 RL]] | 人类干预的在线学习闭环 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/VLA need RL, not just SFT|VLA need RL, not just SFT]]

## 下一步阅读或验证

拆出可复用的离线初始化、人类干预和策略更新三部分，与 HIL-SERL 对齐。

返回 [[论文图谱/主题/04 VLA 后训练与奖励|04 VLA 后训练与奖励]] · [[00-论文知识图谱|知识图谱首页]]
