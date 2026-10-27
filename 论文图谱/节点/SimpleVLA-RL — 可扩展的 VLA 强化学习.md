---
type: paper
node_id: simplevla
topic: posttrain
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, posttrain, paper]
---

# SimpleVLA-RL — 可扩展的 VLA 强化学习

## 节点摘要

**论文 · 04 VLA 后训练与奖励**  
如何提高 VLA 强化学习中的采样、渲染与优化吞吐？

基于 veRL 构建 VLA 训练框架，处理轨迹采样、多环境渲染、并行和损失计算，并在 OpenVLA-OFT 上研究 RL 改进。

## 训练、推理与数据流

区分基础策略、环境交互和分布式训练接口，再测相同资源下的训练效率与任务效果。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

框架支持或某基线成功不意味着可直接替换任意 GR00T action head；动作概率定义和任务接口需适配。

**对应文献**：SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning  
**作者**：作者名单见原文  
**首次预印本年份**：2025（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2509.09674)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 双向比较 | 关联 | [[论文图谱/节点/OpenVLA — 开源视觉语言动作模型\|OpenVLA — 开源视觉语言动作模型]] | OpenVLA-OFT 为后续变体，核对动作 head |
| 双向比较 | 关联 | [[论文图谱/节点/RIPT-VLA、HIL-DAFT 与 RLVLA — 扩展阅读\|RIPT-VLA、HIL-DAFT 与 RLVLA — 扩展阅读]] | 交互后训练与泛化评测对照 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/VLA need RL, not just SFT|VLA need RL, not just SFT]]

## 下一步阅读或验证

与 DPPO／ReinFlow 比较策略参数化，再评估可复用的分布式采样部分。

返回 [[论文图谱/主题/04 VLA 后训练与奖励|04 VLA 后训练与奖励]] · [[00-论文知识图谱|知识图谱首页]]
