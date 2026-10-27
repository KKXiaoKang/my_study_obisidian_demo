---
type: method
node_id: rabc
topic: posttrain
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, posttrain, method]
---

# RA-BC — 用进度信号重加权示教

## 节点摘要

**方法 · 04 VLA 后训练与奖励**  
如何从包含低效或失败片段的示教中突出更有价值的动作？

用奖励模型估计一段动作后的进度变化，把结果转换为样本权重，训练加权行为克隆策略。与 SARM 属于同一论文中的下游策略方法。

## 训练、推理与数据流

先拟合进度模型，再计算样本权重并训练 BC；应记录时间间隔、权重校准、截断规则及覆盖率。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

加权 BC 不等于在线策略梯度 RL。若奖励模型把恢复动作判成倒退，过滤可能损失关键数据。

**对应文献**：SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation  
**作者**：Qianzhong Chen et al.  
**首次预印本年份**：2025（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2509.25358)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 应用 | [[论文图谱/节点/SARM — 阶段感知的进度奖励\|SARM — 阶段感知的进度奖励]] | 用进度模型筛选与加权 BC |
| 双向比较 | 对比 | [[论文图谱/节点/RECAP 与 π*0.6 — 从经验和纠正学习\|RECAP 与 π*0.6 — 从经验和纠正学习]] | 样本重加权与优势条件化 |
| 双向比较 | 对比 | [[论文图谱/节点/rewACT — 动作策略上的奖励辅助头\|rewACT — 动作策略上的奖励辅助头]] | 辅助预测与样本重加权 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/RA-BC reward align behavior cloning|RA-BC reward align behavior cloning]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/IMG/sarm_reward_model_architecture|sarm_reward_model_architecture]]
- [[Robotmanipulation - opt_VA/Imitation learning/rewACT - 将MLP结构的reward model融合在ACT训练当中|rewACT - 将MLP结构的reward model融合在ACT训练当中]]

## 下一步阅读或验证

同时报告权重分布、保留数据比例、任务成功率，并以未加权 BC 为基线。

返回 [[论文图谱/主题/04 VLA 后训练与奖励|04 VLA 后训练与奖励]] · [[00-论文知识图谱|知识图谱首页]]
