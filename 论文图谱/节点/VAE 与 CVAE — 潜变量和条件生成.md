---
type: concept
node_id: vae
topic: foundation
status: "原笔记归纳"
updated: 2026-10-27
tags: [论文图谱, foundation, concept]
---

# VAE 与 CVAE — 潜变量和条件生成

## 节点摘要

**概念 · 01 表征与生成基础**  
同一观测对应多种合理动作时，怎样保留多样性并在推理时可控生成？

AE 学确定性编码；VAE 用近似后验、重参数化和先验正则学习潜变量；CVAE 把条件 c 引入生成过程。CVAE 的 C 是 Conditional，中文应为“条件”。

## 训练、推理与数据流

训练：q(z|x,c) 产生 latent，decoder 重建 x；推理：从先验 p(z|c) 采样或取指定代表值，再用 decoder 生成。常见目标是重构项加 KL 项。

## 证据与适用边界

**整理状态：原笔记归纳。**

KL 不是 MSE 的替代项；两者约束不同对象。先验也不必在所有模型中都是标准高斯，具体依模型定义。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 本节点 → | 基础 | [[论文图谱/节点/ACT — 动作块模仿学习\|ACT — 动作块模仿学习]] | CVAE 潜变量与重构正则 |
| 本节点 → | 基础 | [[论文图谱/节点/GenDexGrasp — 接触图驱动灵巧抓取\|GenDexGrasp — 接触图驱动灵巧抓取]] | 条件接触图生成 |

## 原始笔记

- [[Robotmanipulation - opt_VA/Imitation learning/编码器/AE - 自动编码器|AE - 自动编码器]]
- [[Robotmanipulation - opt_VA/Imitation learning/编码器/CVAE - 差分变分自动编码器|CVAE - 差分变分自动编码器]]
- [[Robotmanipulation - opt_VA/Imitation learning/编码器/MLP - 多层感知机|MLP - 多层感知机]]
- [[Robotmanipulation - opt_VA/Imitation learning/编码器/VAE - 变分自动编码器|VAE - 变分自动编码器]]
- [[Robotmanipulation - opt_VA/KL散度对比MSE计算loss|KL散度对比MSE计算loss]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VA & CVAE/CVAE & VAE 关于半月图之争|CVAE & VAE 关于半月图之争]]
- [[Robotmanipulation - opt_VA/YouKnow3M/cumotion & Gendexgrasp/Gendexgrasp网络 - 后验概率如何拟合先验|Gendexgrasp网络 - 后验概率如何拟合先验]]

## 下一步阅读或验证

对照 ACT 与 GenDexGrasp，写出各自 x、c、z 的含义，并区分训练专用 encoder 与部署 decoder。

返回 [[论文图谱/主题/01 表征与生成基础|01 表征与生成基础]] · [[00-论文知识图谱|知识图谱首页]]
