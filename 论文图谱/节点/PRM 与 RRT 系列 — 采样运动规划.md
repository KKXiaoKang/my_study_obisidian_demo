---
type: method
node_id: sampling
topic: planning
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, planning, method]
---

# PRM 与 RRT 系列 — 采样运动规划

## 节点摘要

**方法 · 07 运动规划与控制**  
高维配置空间中怎样快速找到可行路径，并逐渐改善路径质量？

PRM 构建可复用路网，RRT 扩展探索树，RRT-Connect 双向连接；PRM*／RRT* 研究渐近最优性，Informed RRT* 聚焦可能改善当前路径的区域。

## 训练、推理与数据流

采样、邻域搜索、局部连接、碰撞检查与路径提取分别计时；最优变体还需要合适的连接和重连规则。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

原综述把 Sampling-based Algorithms for Optimal Motion Planning 标成 2001，原文是 2011。Informed RRT* 的起终点焦点固定，收缩的是可改善解的采样区域。

**对应文献**：Sampling-based Algorithms for Optimal Motion Planning  
**作者**：Sertac Karaman and Emilio Frazzoli  
**首次预印本年份**：2011（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/1105.1186)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 双向比较 | 对比 | [[论文图谱/节点/CHOMP 与 GPMP — 轨迹优化的两种视角\|CHOMP 与 GPMP — 轨迹优化的两种视角]] | 搜索路径与局部连续优化 |

## 原始笔记

- [[Robotmanipulation - opt_VA/运动规划方法一览|运动规划方法一览]]

## 下一步阅读或验证

分开比较首次可行解时间与固定预算后的解质量，再与轨迹优化方法比较。

返回 [[论文图谱/主题/07 运动规划与控制|07 运动规划与控制]] · [[00-论文知识图谱|知识图谱首页]]
