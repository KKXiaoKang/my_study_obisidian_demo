---
type: paper
node_id: curobo
topic: planning
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, planning, paper]
---

# cuRobo — GPU 并行运动生成

## 节点摘要

**论文 · 07 运动规划与控制**  
怎样在短时间内求解机械臂无碰撞、平滑的运动轨迹？

利用 GPU 并行多种子优化，组合碰撞检测、IK、几何规划与轨迹优化；并非仅随机生成很多轨迹后选最小代价。

## 训练、推理与数据流

输入机器人模型、起终状态和障碍表示，输出轨迹；工程层需对齐 ESDF 语义、插值、执行频率和规划失败处理。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

论文的平均耗时与原笔记的本地耗时不能混用；机械臂运动生成也不自动包含全身动力学平衡。

**对应文献**：cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation  
**作者**：Balakumar Sundaralingam et al.  
**首次预印本年份**：2023（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2310.17274)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 双向比较 | 关联 | [[论文图谱/节点/CHOMP 与 GPMP — 轨迹优化的两种视角\|CHOMP 与 GPMP — 轨迹优化的两种视角]] | 轨迹代价与多种子优化 |
| → 本节点 | 基础 | [[论文图谱/节点/ESDF 与碰撞表示 — 从感知到约束\|ESDF 与碰撞表示 — 从感知到约束]] | 障碍距离语义对齐 |
| 双向比较 | 对比 | [[论文图谱/节点/OCS2 与 KMPC — 滚动优化和实时执行\|OCS2 与 KMPC — 滚动优化和实时执行]] | 并行运动生成与滚动最优控制 |
| → 本节点 | 应用 | [[论文图谱/节点/GenDexGrasp — 接触图驱动灵巧抓取\|GenDexGrasp — 接触图驱动灵巧抓取]] | 抓取目标到无碰撞到达轨迹 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/cumotion & Gendexgrasp/cumotion & Gendexgrasp 改动一览|cumotion & Gendexgrasp 改动一览]]
- [[Robotmanipulation - opt_VA/YouKnow3M/cumotion & Gendexgrasp/ocs2 CPU vs curobo GPU 优化架构对比|ocs2 CPU vs curobo GPU 优化架构对比]]
- [[Robotmanipulation - opt_VA/YouKnow3M/cumotion & Gendexgrasp/技术讨论|技术讨论]]
- [[Robotmanipulation - opt_VA/YouKnow3M/demo|demo]]
- [[Robot工具/curobo convert urdf 2 usd|curobo convert urdf 2 usd]]

## 下一步阅读或验证

与 OCS2 比较求解目标和并行方式，记录完整感知—规划—执行延迟。

返回 [[论文图谱/主题/07 运动规划与控制|07 运动规划与控制]] · [[00-论文知识图谱|知识图谱首页]]
