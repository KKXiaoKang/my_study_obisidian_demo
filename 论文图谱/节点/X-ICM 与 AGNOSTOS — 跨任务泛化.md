---
type: paper
node_id: xicm
topic: vla
status: "原笔记归纳；原文细节待核对"
updated: 2026-10-27
tags: [论文图谱, vla, paper]
---

# X-ICM 与 AGNOSTOS — 跨任务泛化

## 节点摘要

**论文 · 02 模仿学习与 VLA**  
已有任务示例如何帮助模型在未见任务中执行操作？

原笔记介绍 AGNOSTOS 基准和 Cross-Task In-Context Manipulation，重点比较检索／示例条件化与参数微调路线。

## 训练、推理与数据流

把视频相关模型、示例选择、提示构造和动作输出分开，明确哪些模块训练、哪些保持固定。

## 证据与适用边界

**整理状态：原笔记归纳；原文细节待核对。**

不能把“加 K 条示例”和 SFT 概括为唯一差别；是否训练模型、检索空间和任务划分都会改变比较。具体实现依原文核对。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 双向比较 | 对比 | [[论文图谱/节点/VIMA — 多模态提示操控\|VIMA — 多模态提示操控]] | 提示中的任务示例如何进入策略 |
| 双向比较 | 对比 | [[论文图谱/节点/Human-to-Robot Transfer — 人类示教迁移\|Human-to-Robot Transfer — 人类示教迁移]] | 人类示范与任务上下文泛化 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/Exploring the Limits of Vision-Language-ActionManipulation in Cross-task Generalization|Exploring the Limits of Vision-Language-ActionManipulation in Cross-task Generalization]]

## 下一步阅读或验证

固定 seen／unseen 任务边界，比较无示例、随机示例、检索示例和微调基线。

返回 [[论文图谱/主题/02 模仿学习与 VLA|02 模仿学习与 VLA]] · [[00-论文知识图谱|知识图谱首页]]
