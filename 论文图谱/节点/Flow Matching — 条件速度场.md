---
type: paper
node_id: flow
topic: foundation
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, foundation, paper]
---

# Flow Matching — 条件速度场

## 节点摘要

**论文 · 01 表征与生成基础**  
能否通过回归速度场，把简单噪声分布输运成数据分布？

为噪声与数据构造条件概率路径，学习该路径的向量场。常见线性路径 x_t=(1-t)ε+t·a 的目标速度为 a−ε；这是常用实例，不是全部 FM。

## 训练、推理与数据流

训练抽取 t 和样本对，回归速度；推理从噪声出发数值积分 ODE。观测、语言和状态可作为条件。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

不能用“FM 非马尔可夫、DP 马尔可夫”作为根本分类。确定性 ODE 在给定初始噪声时确定，但多次采样仍可产生多样动作。

**对应文献**：Flow Matching for Generative Modeling  
**作者**：Yaron Lipman et al.  
**首次预印本年份**：2022（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2210.02747)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 本节点 → | 基础 | [[论文图谱/节点/π0 — 视觉语言动作流模型\|π0 — 视觉语言动作流模型]] | 连续动作的向量场生成 |
| 本节点 → | 基础 | [[论文图谱/节点/GR00T N1 与 N1.5 — 通用模型及实现分支\|GR00T N1 与 N1.5 — 通用模型及实现分支]] | 理解笔记中的 flow action head |
| 双向比较 | 对比 | [[论文图谱/节点/Diffusion Policy — 扩散动作策略\|Diffusion Policy — 扩散动作策略]] | 路径、目标与采样器 |
| 本节点 → | 扩展 | [[论文图谱/节点/ReinFlow — Flow Policy 的在线 RL\|ReinFlow — Flow Policy 的在线 RL]] | 为 flow 路径引入训练期随机性 |
| 本节点 → | 基础 | [[论文图谱/节点/DiT4DiT — 视频与动作的级联建模\|DiT4DiT — 视频与动作的级联建模]] | 视频与动作双 flow 目标 |
| 双向比较 | 对比 | [[论文图谱/节点/JiT 与 Drifting — 生成参数化阅读线索\|JiT 与 Drifting — 生成参数化阅读线索]] | 输出参数化与损失参数化 |

## 原始笔记

- [[Robotmanipulation - opt_VA/Imitation learning/Diffusion Policy - Flow Matching 对比|Diffusion Policy - Flow Matching 对比]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/FM vs DP noise denoise流程分析|FM vs DP noise denoise流程分析]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/模型loss差异|模型loss差异]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/生成模型新范式 - JiT or Drifting Model|生成模型新范式 - JiT or Drifting Model]]

## 下一步阅读或验证

把路径、网络输出、loss 和求解器分成四列比较 DP、FM 与 JiT。

返回 [[论文图谱/主题/01 表征与生成基础|01 表征与生成基础]] · [[00-论文知识图谱|知识图谱首页]]
