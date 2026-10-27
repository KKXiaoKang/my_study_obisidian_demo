---
type: paper
node_id: act
topic: vla
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, vla, paper]
---

# ACT — 动作块模仿学习

## 节点摘要

**论文 · 02 模仿学习与 VLA**  
低成本双臂硬件上，怎样从少量示教学习精细且时序一致的动作？

Action Chunking with Transformers 用 CVAE 建模动作序列，并以时间集成融合重叠预测。chunk 让策略一次预测一段动作，latent 用于建模示教差异。

## 训练、推理与数据流

训练用真实动作与状态编码 latent，并优化动作重构与 KL；原始 ACT 推理使用先验均值 z=0。原笔记的损失对照明确指出所讨论实现用 L1 重构。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

推理取 z=0 是原方法的设计选择，不能仅凭这一点称为漏洞。把 actor 换成 ACT 也不自动得到论文中的 Q-chunking。

**对应文献**：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware  
**作者**：Tony Z. Zhao et al.  
**首次预印本年份**：2023（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2304.13705)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 基础 | [[论文图谱/节点/Transformer — 注意力与信息通路\|Transformer — 注意力与信息通路]] | 动作块模型使用注意力结构 |
| → 本节点 | 基础 | [[论文图谱/节点/VAE 与 CVAE — 潜变量和条件生成\|VAE 与 CVAE — 潜变量和条件生成]] | CVAE 潜变量与重构正则 |
| → 本节点 | 实例 | [[论文图谱/节点/行为克隆 — 策略表示与监督基线\|行为克隆 — 策略表示与监督基线]] | CVAE 动作块 |
| 双向比较 | 对比 | [[论文图谱/节点/Diffusion Policy — 扩散动作策略\|Diffusion Policy — 扩散动作策略]] | 动作生成方式、时序和部署成本 |
| 双向比较 | 关联 | [[论文图谱/节点/Q-chunking — 动作块空间中的 RL\|Q-chunking — 动作块空间中的 RL]] | 共享动作块思想，训练目标不同 |
| 本节点 → | 方案 | [[论文图谱/节点/Deep Reactive Policy — 全局意图与局部避障\|Deep Reactive Policy — 全局意图与局部避障]] | 学习的全局意图与局部反应接口 |
| → 本节点 | 工程 | [[论文图谱/节点/LeRobot 与 RL 工程链路\|LeRobot 与 RL 工程链路]] | 数据归一化与动作接口 |
| 本节点 → | 扩展 | [[论文图谱/节点/rewACT — 动作策略上的奖励辅助头\|rewACT — 动作策略上的奖励辅助头]] | 本地奖励辅助头实验 |

## 原始笔记

- [[Robotmanipulation - opt_VA/Imitation learning/※ ACT -  Transformer CVAE动作生成器|※ ACT -  Transformer CVAE动作生成器]]
- [[RobotRL/Q-chunk ACT actor/ACT - latenX 推理时设置为None存在的漏洞|ACT - latenX 推理时设置为None存在的漏洞]]
- [[RobotRL/Q-chunk ACT actor/mlp actor 2 act actor - 架构升级说明|mlp actor 2 act actor - 架构升级说明]]
- [[Robotmanipulation - opt_VA/Imitation learning/rewACT - 将MLP结构的reward model融合在ACT训练当中|rewACT - 将MLP结构的reward model融合在ACT训练当中]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/模型loss差异|模型loss差异]]

## 下一步阅读或验证

分开比较 chunk 长度、时间集成、latent 使用方式；记录连续动作质量和闭环成功率。

返回 [[论文图谱/主题/02 模仿学习与 VLA|02 模仿学习与 VLA]] · [[00-论文知识图谱|知识图谱首页]]
