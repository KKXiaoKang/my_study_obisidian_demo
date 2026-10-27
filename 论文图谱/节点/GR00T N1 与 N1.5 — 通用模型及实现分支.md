---
type: paper
node_id: groot
topic: vla
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, vla, paper]
---

# GR00T N1 与 N1.5 — 通用模型及实现分支

## 节点摘要

**论文 · 02 模仿学习与 VLA**  
如何将视觉语言理解与连续动作生成耦合，用于人形机器人的多任务操控？

N1 论文采用视觉语言模块与 diffusion transformer 动作模块的双系统设计，并混合真实机器人、人类视频与合成数据。

## 训练、推理与数据流

N1.5 相关原笔记主要是 action head 与本地改造分析；应单独记录其版本、embodiment、维度和训练目标，不能直接套用 N1 的全部配置。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

多头解码、双臂注意力、优势 token 等本地改造属于工程方案，不能写成官方论文原有贡献。代码注释也不能证明已获真机收益。

**对应文献**：GR00T N1: An Open Foundation Model for Generalist Humanoid Robots  
**作者**：NVIDIA et al.  
**首次预印本年份**：2025（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2503.14734)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 基础 | [[论文图谱/节点/Flow Matching — 条件速度场\|Flow Matching — 条件速度场]] | 理解笔记中的 flow action head |
| 双向比较 | 对比 | [[论文图谱/节点/π0.5 — 异构协同训练与开放环境泛化\|π0.5 — 异构协同训练与开放环境泛化]] | joint attention 与动作模块条件路径 |
| 本节点 → | 方案 | [[论文图谱/节点/GR00T 优势条件化 — 后训练方案\|GR00T 优势条件化 — 后训练方案]] | 向本地 flow head 注入优势条件 |
| 本节点 → | 方案 | [[论文图谱/节点/WAM 融合方案 — 从设想到可检验假设\|WAM 融合方案 — 从设想到可检验假设]] | 动作专家与多视觉流融合 |
| 本节点 → | 方案 | [[论文图谱/节点/动作头分解 — 尺度、共享与双臂协调\|动作头分解 — 尺度、共享与双臂协调]] | 按动作语义拆解输出头 |

## 原始笔记

- [[RobotVLA/多模态对齐  - 特征融合部分|多模态对齐  - 特征融合部分]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/GROOT N1.5 action expert改动一览|GROOT N1.5 action expert改动一览]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/pi05 vs GROOT|pi05 vs GROOT]]
- [[RobotRL/VLA-RL/GROOT N1.5 flow-matching head 如何修改带advantage优势值条件生成|GROOT N1.5 flow-matching head 如何修改带advantage优势值条件生成]]

## 下一步阅读或验证

先建立未修改基线，再逐项消融多头、损失权重、双臂协调与优势条件。

返回 [[论文图谱/主题/02 模仿学习与 VLA|02 模仿学习与 VLA]] · [[00-论文知识图谱|知识图谱首页]]
