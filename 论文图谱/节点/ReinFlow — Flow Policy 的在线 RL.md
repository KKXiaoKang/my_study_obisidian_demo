---
type: paper
node_id: reinflow
topic: posttrain
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, posttrain, paper]
---

# ReinFlow — Flow Policy 的在线 RL

## 节点摘要

**论文 · 04 VLA 后训练与奖励**  
如何为确定性 flow 生成路径构造便于 RL 的随机策略和概率计算？

在 flow 路径中注入可学习噪声，形成离散时间随机过程，使训练中的 likelihood 与探索可处理。

## 训练、推理与数据流

在已有 flow policy 上执行在线策略优化；必须明确每步噪声、生成轨迹概率和环境奖励如何连接。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

需要 likelihood 的是选定 RL 估计器，不是所有 RL 方法都必须直接计算完整生成模型密度；跨机器人迁移仍需验证。

**对应文献**：ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning  
**作者**：作者名单见原文  
**首次预印本年份**：2025（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2505.22094)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 扩展 | [[论文图谱/节点/Flow Matching — 条件速度场\|Flow Matching — 条件速度场]] | 为 flow 路径引入训练期随机性 |
| 双向比较 | 对比 | [[论文图谱/节点/DPPO — 扩散策略的策略梯度优化\|DPPO — 扩散策略的策略梯度优化]] | 扩散与 flow 的 RL 接口 |
| 双向比较 | 对比 | [[论文图谱/节点/RECAP 与 π*0.6 — 从经验和纠正学习\|RECAP 与 π*0.6 — 从经验和纠正学习]] | 策略梯度与优势条件化 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/VLA need RL, not just SFT|VLA need RL, not just SFT]]
- [[RobotRL/VLA-RL/GROOT N1.5 flow-matching head 如何修改带advantage优势值条件生成|GROOT N1.5 flow-matching head 如何修改带advantage优势值条件生成]]

## 下一步阅读或验证

与 RECAP 的优势条件化对比，确定需要修改采样器还是只改变监督与条件输入。

返回 [[论文图谱/主题/04 VLA 后训练与奖励|04 VLA 后训练与奖励]] · [[00-论文知识图谱|知识图谱首页]]
