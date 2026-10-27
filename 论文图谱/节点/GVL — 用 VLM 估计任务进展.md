---
type: paper
node_id: gvl
topic: posttrain
status: "原笔记归纳；原文细节待核对"
updated: 2026-10-27
tags: [论文图谱, posttrain, paper]
---

# GVL — 用 VLM 估计任务进展

## 节点摘要

**论文 · 04 VLA 后训练与奖励**  
能否利用 VLM 的语义理解，在少量任务条件下评估状态的价值或进度？

原笔记介绍通过视觉语言模型对轨迹状态进行语义评估，以形成任务相关价值信号。

## 训练、推理与数据流

把任务描述、图像序列和评估提示作为输入；将输出评分的定义与策略训练如何使用该分数分开记录。

## 证据与适用边界

**整理状态：原笔记归纳；原文细节待核对。**

语义进度评分不天然等同于 Bellman 意义的折扣回报价值；提示敏感性与接触细节识别需要单独测量。

原始来源：[论文／作者项目入口](https://generative-value-learning.github.io/)。链接由已有笔记或书目线索整理，未逐项复核正文。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 双向比较 | 对比 | [[论文图谱/节点/SARM — 阶段感知的进度奖励\|SARM — 阶段感知的进度奖励]] | 通用语义评估与专用阶段进度 |

## 原始笔记

- [[Robotmanipulation - opt_VA/VLA/VLM学习价值函数 - Vision Language Models are In-Context Value Learners 谷歌deepmind|VLM学习价值函数 - Vision Language Models are In-Context Value Learners 谷歌deepmind]]

## 下一步阅读或验证

与 SARM 比较是否需训练专用奖励模型、时序监督来源和 rollout 上的一致性。

返回 [[论文图谱/主题/04 VLA 后训练与奖励|04 VLA 后训练与奖励]] · [[00-论文知识图谱|知识图谱首页]]
