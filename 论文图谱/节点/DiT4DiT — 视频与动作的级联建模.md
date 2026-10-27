---
type: paper
node_id: dit4dit
topic: wam
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, wam, paper]
---

# DiT4DiT — 视频与动作的级联建模

## 节点摘要

**论文 · 05 世界模型与长程能力**  
怎样使用视频模型中的时空信息来改善动作生成？

级联视频 DiT 与动作 DiT，读取视频去噪中间特征作为动作条件，并用双 flow matching 目标联合建模视频动态和动作。

## 训练、推理与数据流

分开记录视频预测、hidden 提取和动作推理的时间步与噪声尺度；部署不应笼统描述成“先完整生成未来视频再控制”。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

原笔记提到的单步推理、detach 和未启用 future token 是特定 motionWAM 实现快照，不等于 DiT4DiT 论文的全部定义。

**对应文献**：DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control  
**作者**：Teli Ma et al.  
**首次预印本年份**：2026（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2603.10448)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 基础 | [[论文图谱/节点/Flow Matching — 条件速度场\|Flow Matching — 条件速度场]] | 视频与动作双 flow 目标 |
| 双向比较 | 对比 | [[论文图谱/节点/JEPA-WAM — 联合嵌入世界动作模型\|JEPA-WAM — 联合嵌入世界动作模型]] | 去噪特征条件与联合嵌入目标 |
| → 本节点 | 评测 | [[论文图谱/节点/Latent 容量与探针 — 怎样评价表征\|Latent 容量与探针 — 怎样评价表征]] | token 数、压缩与读出信息 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/JEPA-WAM 精读 - latent WAM 四类范式与 motionWAM 的归类|JEPA-WAM 精读 - latent WAM 四类范式与 motionWAM 的归类]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VA & CVAE/latent space dim的思考|latent space dim的思考]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/未来发展 - pi05 GROOT future 范式下视觉-动作联立建模下的流如何融合?|未来发展 - pi05 GROOT future 范式下视觉-动作联立建模下的流如何融合?]]

## 下一步阅读或验证

与 JEPA-WAM 比较预测空间、动作条件来源和实际端到端成本。

返回 [[论文图谱/主题/05 世界模型与长程能力|05 世界模型与长程能力]] · [[00-论文知识图谱|知识图谱首页]]
