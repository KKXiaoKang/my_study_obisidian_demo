---
type: paper
node_id: dp
topic: vla
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, vla, paper]
---

# Diffusion Policy — 扩散动作策略

## 节点摘要

**论文 · 02 模仿学习与 VLA**  
怎样避免单一动作回归平均掉多种可行策略，并生成连贯的动作序列？

把视觉条件下的动作分布表示为去噪生成过程，结合视觉条件、动作序列预测和滚动时域执行。

## 训练、推理与数据流

常见训练对动作加噪并预测噪声；部署迭代去噪，执行部分动作后重新观测。网络参数化和 DDPM／DDIM 等采样选择需单独记录。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

score、噪声预测网络与显式标量能量函数不是同一个对象。生成步数和真实机器人控制频率也不是同一指标。

**对应文献**：Diffusion Policy: Visuomotor Policy Learning via Action Diffusion  
**作者**：Cheng Chi et al.  
**首次预印本年份**：2023（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2303.04137)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 实例 | [[论文图谱/节点/行为克隆 — 策略表示与监督基线\|行为克隆 — 策略表示与监督基线]] | 扩散动作分布 |
| 双向比较 | 对比 | [[论文图谱/节点/ACT — 动作块模仿学习\|ACT — 动作块模仿学习]] | 动作生成方式、时序和部署成本 |
| 双向比较 | 对比 | [[论文图谱/节点/Flow Matching — 条件速度场\|Flow Matching — 条件速度场]] | 路径、目标与采样器 |
| 本节点 → | 扩展 | [[论文图谱/节点/GenDP — 三维语义场与扩散策略\|GenDP — 三维语义场与扩散策略]] | 用三维语义条件增强泛化 |
| 本节点 → | 扩展 | [[论文图谱/节点/DPPO — 扩散策略的策略梯度优化\|DPPO — 扩散策略的策略梯度优化]] | 用策略梯度改进扩散策略 |
| 本节点 → | 扩展 | [[论文图谱/节点/DSRL — 在扩散噪声空间中学习引导\|DSRL — 在扩散噪声空间中学习引导]] | 冻结生成器，优化噪声选择 |

## 原始笔记

- [[Robotmanipulation - opt_VA/Imitation learning/※ Diffusion Policy - 扩散去噪动作策略模型|※ Diffusion Policy - 扩散去噪动作策略模型]]
- [[Robotmanipulation - opt_VA/Imitation learning/Diffusion Policy - Flow Matching 对比|Diffusion Policy - Flow Matching 对比]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/FM vs DP noise denoise流程分析|FM vs DP noise denoise流程分析]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/模型loss差异|模型loss差异]]

## 下一步阅读或验证

与 ACT 比较多模态行为和时延，再沿 GenDP 看表征增强，沿 DPPO／DSRL 看后训练。

返回 [[论文图谱/主题/02 模仿学习与 VLA|02 模仿学习与 VLA]] · [[00-论文知识图谱|知识图谱首页]]
