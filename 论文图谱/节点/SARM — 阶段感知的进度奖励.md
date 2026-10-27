---
type: paper
node_id: sarm
topic: posttrain
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, posttrain, paper]
---

# SARM — 阶段感知的进度奖励

## 节点摘要

**论文 · 04 VLA 后训练与奖励**  
不同长度、质量和动作顺序的示教，怎样获得可比较的任务进度？

根据语言子任务标注学习阶段与阶段内进度，减轻直接把帧索引当进度产生的标签偏差。

## 训练、推理与数据流

训练阶段预测与进度估计；下游把进度变化用于筛选或加权示教。原架构笔记还包含本地双塔实现与配置。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

阶段顺序、回退与失败会影响进度意义。论文定义、本地归一化函数和具体层数要分开核对，不能把本地配置直接标为论文配置。

**对应文献**：SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation  
**作者**：Qianzhong Chen et al.  
**首次预印本年份**：2025（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2509.25358)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 本节点 → | 应用 | [[论文图谱/节点/RA-BC — 用进度信号重加权示教\|RA-BC — 用进度信号重加权示教]] | 用进度模型筛选与加权 BC |
| 双向比较 | 对比 | [[论文图谱/节点/GVL — 用 VLM 估计任务进展\|GVL — 用 VLM 估计任务进展]] | 通用语义评估与专用阶段进度 |
| 双向比较 | 对比 | [[论文图谱/节点/χ0（kai0）— 阶段优势与训练部署对齐\|χ0（kai0）— 阶段优势与训练部署对齐]] | 阶段信号的定义与使用 |
| 双向比较 | 对比 | [[论文图谱/节点/GR-RL — 从通用策略到精细专家\|GR-RL — 从通用策略到精细专家]] | 进度模型与 Q 价值筛选 |
| 双向比较 | 对比 | [[论文图谱/节点/rewACT — 动作策略上的奖励辅助头\|rewACT — 动作策略上的奖励辅助头]] | 独立进度模型与共享 reward head |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/IMG/sarm_reward_model_architecture|sarm_reward_model_architecture]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/RA-BC reward align behavior cloning|RA-BC reward align behavior cloning]]

## 下一步阅读或验证

在未见轨迹和策略 rollout 上验证进度，再检查奖励误判是否导致 RA-BC 错误筛选。

返回 [[论文图谱/主题/04 VLA 后训练与奖励|04 VLA 后训练与奖励]] · [[00-论文知识图谱|知识图谱首页]]
