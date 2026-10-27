---
type: paper
node_id: transformer
topic: foundation
status: "原笔记归纳；原文细节待核对"
updated: 2026-10-27
tags: [论文图谱, foundation, paper]
---

# Transformer — 注意力与信息通路

## 节点摘要

**论文 · 01 表征与生成基础**  
序列中不同位置怎样交换信息，并让解码器只使用允许看到的上下文？

用 Q、K、V 计算注意力；位置编码表达顺序，残差、归一化和前馈层组成 block。原始 encoder-decoder 结构是理解后续 cross-attention 和 mask 的入口。

## 训练、推理与数据流

训练时并行处理 token，并通过 mask 限制信息流；自回归推理逐步生成。机器人动作专家可以复用这些模块，但输出不必是词表 token。

## 证据与适用边界

**整理状态：原笔记归纳；原文细节待核对。**

“都是 Transformer”不等于相同条件路径。要分别标出谁提供 Q/K/V、哪些参数共享、哪些位置允许互看。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/1706.03762)。链接由已有笔记或书目线索整理，未逐项复核正文。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 本节点 → | 基础 | [[论文图谱/节点/ACT — 动作块模仿学习\|ACT — 动作块模仿学习]] | 动作块模型使用注意力结构 |
| 本节点 → | 基础 | [[论文图谱/节点/π0.5 — 异构协同训练与开放环境泛化\|π0.5 — 异构协同训练与开放环境泛化]] | 用 QKV 与 mask 理解通路 |

## 原始笔记

- [[Robotmanipulation - opt_VA/Imitation learning/Transformer/01 - Transformer - Transformer 输入|01 - Transformer - Transformer 输入]]
- [[Robotmanipulation - opt_VA/Imitation learning/Transformer/02 - Transformer - Self-Attention（自注意力机制）|02 - Transformer - Self-Attention（自注意力机制）]]
- [[Robotmanipulation - opt_VA/Imitation learning/Transformer/03 - Transformer - Encoder结构|03 - Transformer - Encoder结构]]
- [[Robotmanipulation - opt_VA/Imitation learning/Transformer/04 - Transformer - Decoder结构|04 - Transformer - Decoder结构]]
- [[Robotmanipulation - opt_VA/Imitation learning/Transformer/Attention Is All You Need -  Transformer 开山之作 - pipeline|Attention Is All You Need -  Transformer 开山之作 - pipeline]]

## 下一步阅读或验证

拿 π0.5 与 GR00T 的原笔记画同一尺度的数据流图，逐条标注 mask 和参数归属。

返回 [[论文图谱/主题/01 表征与生成基础|01 表征与生成基础]] · [[00-论文知识图谱|知识图谱首页]]
