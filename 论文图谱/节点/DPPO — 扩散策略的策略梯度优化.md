---
type: paper
node_id: dppo
topic: posttrain
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, posttrain, paper]
---

# DPPO — 扩散策略的策略梯度优化

## 节点摘要

**论文 · 04 VLA 后训练与奖励**  
如何对已经用示教训练好的扩散动作策略做在线强化学习？

把扩散策略的生成结构纳入策略梯度优化框架，利用其结构化探索，同时处理去噪过程与环境交互两个时间尺度。

## 训练、推理与数据流

从预训练策略出发采集 rollout，构造优势和相应概率比，更新生成策略；去噪步、环境步和动作块必须区别。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

原笔记的通用 PPO 公式不能替代完整 DPPO 实现。额外 BC loss 或熵项是否使用，应核对具体配置。

**对应文献**：Diffusion Policy Policy Optimization  
**作者**：Allen Z. Ren et al.  
**首次预印本年份**：2024（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2409.00588)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 扩展 | [[论文图谱/节点/Diffusion Policy — 扩散动作策略\|Diffusion Policy — 扩散动作策略]] | 用策略梯度改进扩散策略 |
| → 本节点 | 基础 | [[论文图谱/节点/PPO — 受约束的策略更新\|PPO — 受约束的策略更新]] | 概率比与优势估计 |
| 双向比较 | 对比 | [[论文图谱/节点/DSRL — 在扩散噪声空间中学习引导\|DSRL — 在扩散噪声空间中学习引导]] | 更新策略参数或噪声引导器 |
| 双向比较 | 对比 | [[论文图谱/节点/ReinFlow — Flow Policy 的在线 RL\|ReinFlow — Flow Policy 的在线 RL]] | 扩散与 flow 的 RL 接口 |

## 原始笔记

- [[RobotRL/Diffusion-RL/DPPO - Diffusion Policy Policy Optimization|DPPO - Diffusion Policy Policy Optimization]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/VLA need RL, not just SFT|VLA need RL, not just SFT]]

## 下一步阅读或验证

与冻结基础策略的 DSRL、面向 flow 的 ReinFlow 比较更新对象和采样开销。

返回 [[论文图谱/主题/04 VLA 后训练与奖励|04 VLA 后训练与奖励]] · [[00-论文知识图谱|知识图谱首页]]
