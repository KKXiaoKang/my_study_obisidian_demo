---
type: concept
node_id: esdf
topic: planning
status: "原笔记归纳"
updated: 2026-10-27
tags: [论文图谱, planning, concept]
---

# ESDF 与碰撞表示 — 从感知到约束

## 节点摘要

**概念 · 07 运动规划与控制**  
怎样把场景几何变成规划器可查询的距离与梯度？

ESDF 表示到障碍表面的有符号距离，多球模型近似机器人几何；通过距离和球半径构造避障代价或约束。

## 训练、推理与数据流

对齐坐标系、距离符号、体素分辨率与未知区域含义，再检查梯度和碰撞查询。

## 证据与适用边界

**整理状态：原笔记归纳。**

漂亮点云与准确距离场是不同输出；未知区域不能无条件当自由空间。多球近似还会引入保守性或漏检。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 应用 | [[论文图谱/节点/仿真资产与工具 — 从重建到实验环境\|仿真资产与工具 — 从重建到实验环境]] | 场景几何到距离约束 |
| 本节点 → | 基础 | [[论文图谱/节点/cuRobo — GPU 并行运动生成\|cuRobo — GPU 并行运动生成]] | 障碍距离语义对齐 |
| 本节点 → | 应用 | [[论文图谱/节点/OCS2 与 KMPC — 滚动优化和实时执行\|OCS2 与 KMPC — 滚动优化和实时执行]] | 作为约束或代价输入 |
| 双向比较 | 关联 | [[论文图谱/节点/Deep Reactive Policy — 全局意图与局部避障\|Deep Reactive Policy — 全局意图与局部避障]] | 局部几何信息用于避障 |

## 原始笔记

- [[Robotmanipulation - opt_VA/优化规划/ESDF - 多球模型|ESDF - 多球模型]]
- [[RobotVision/高精度地图测绘/高精度测绘|高精度测绘]]
- [[Robotmanipulation - opt_VA/YouKnow3M/cumotion & Gendexgrasp/cumotion & Gendexgrasp 改动一览|cumotion & Gendexgrasp 改动一览]]
- [[Robotmanipulation - opt_VA/YouKnow3M/cumotion & Gendexgrasp/技术讨论|技术讨论]]
- [[RobotSim/isaac-sim碰撞对比/碰撞对比表格|碰撞对比表格]]

## 下一步阅读或验证

用已知几何测距离误差，再把误差传播到 cuRobo／MPC 的安全距离设置。

返回 [[论文图谱/主题/07 运动规划与控制|07 运动规划与控制]] · [[00-论文知识图谱|知识图谱首页]]
