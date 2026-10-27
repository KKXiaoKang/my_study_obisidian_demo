---
type: engineering
node_id: ocs2
topic: planning
status: "原笔记归纳"
updated: 2026-10-27
tags: [论文图谱, planning, engineering]
---

# OCS2 与 KMPC — 滚动优化和实时执行

## 节点摘要

**工程 · 07 运动规划与控制**  
高频遥操目标怎样通过受约束优化变成稳定控制输入？

原笔记描述状态空间模型、SQP 子问题、代价与约束，以及 MPC 求解和 MRT 执行异步分工。

## 训练、推理与数据流

从 reference 到状态／输入轨迹，再由执行循环读取策略；明确求解频率、执行频率、延迟和约束是否满足。

## 证据与适用边界

**整理状态：原笔记归纳。**

记录中的 SQP、DDP、100Hz 和模型类型属于特定配置；不能推出 SQP 在所有任务上一定优于 DDP。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 应用 | [[论文图谱/节点/ESDF 与碰撞表示 — 从感知到约束\|ESDF 与碰撞表示 — 从感知到约束]] | 作为约束或代价输入 |
| 双向比较 | 对比 | [[论文图谱/节点/cuRobo — GPU 并行运动生成\|cuRobo — GPU 并行运动生成]] | 并行运动生成与滚动最优控制 |
| 双向比较 | 对比 | [[论文图谱/节点/REMANI-Planner — 移动操作约束规划\|REMANI-Planner — 移动操作约束规划]] | 移动操作模型与任务约束 |
| → 本节点 | 应用 | [[论文图谱/节点/VR Retarget — 人体到机器人目标映射\|VR Retarget — 人体到机器人目标映射]] | 遥操目标进入 MPC |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/kmpc/MPC + MRT 说明 & 常见QA|MPC + MRT 说明 & 常见QA]]
- [[Robotmanipulation - opt_VA/YouKnow3M/kmpc/QP问题，ocs2根据reference pose进行线性插值得到一系列input|QP问题，ocs2根据reference pose进行线性插值得到一系列input]]
- [[Robotmanipulation - opt_VA/YouKnow3M/kmpc/SSM state space model 详解 ocs2|SSM state space model 详解 ocs2]]
- [[Robotmanipulation - opt_VA/YouKnow3M/kmpc/kmpc改动一览|kmpc改动一览]]
- [[Robotmanipulation - opt_VA/YouKnow3M/cumotion & Gendexgrasp/ocs2 CPU vs curobo GPU 优化架构对比|ocs2 CPU vs curobo GPU 优化架构对比]]

## 下一步阅读或验证

用时间戳检查目标更新、求解完成和实际应用时刻，避免只以求解器耗时评价闭环。

返回 [[论文图谱/主题/07 运动规划与控制|07 运动规划与控制]] · [[00-论文知识图谱|知识图谱首页]]
