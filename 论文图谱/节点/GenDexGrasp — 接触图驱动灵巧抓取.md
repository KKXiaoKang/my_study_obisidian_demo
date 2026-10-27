---
type: paper
node_id: gendex
topic: planning
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, planning, paper]
---

# GenDexGrasp — 接触图驱动灵巧抓取

## 节点摘要

**论文 · 07 运动规划与控制**  
怎样生成多样抓取，并在不同灵巧手之间共享表示？

用接触图作为相对独立于手型的中间表示，结合多手数据学习与后端优化，生成可行的灵巧抓取。

## 训练、推理与数据流

原笔记具体展开 PointNetCVAE 的编码、解码和接触图损失，再通过优化得到手部姿态／关节。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

预测接触图不等于直接输出已验证可执行的手关节；静态抓取可行性、到达路径避障和动态接触稳定性需分别验证。

**对应文献**：GenDexGrasp: Generalizable Dexterous Grasping  
**作者**：作者名单见原文  
**首次预印本年份**：2022（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2210.00722)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 基础 | [[论文图谱/节点/VAE 与 CVAE — 潜变量和条件生成\|VAE 与 CVAE — 潜变量和条件生成]] | 条件接触图生成 |
| 本节点 → | 应用 | [[论文图谱/节点/cuRobo — GPU 并行运动生成\|cuRobo — GPU 并行运动生成]] | 抓取目标到无碰撞到达轨迹 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/cumotion & Gendexgrasp/Gendexgrasp网络 - 后验概率如何拟合先验|Gendexgrasp网络 - 后验概率如何拟合先验]]
- [[Robotmanipulation - opt_VA/YouKnow3M/cumotion & Gendexgrasp/Gendexgrasp网络 - 网络架构 - 训练流程讲解|Gendexgrasp网络 - 网络架构 - 训练流程讲解]]
- [[Robotmanipulation - opt_VA/YouKnow3M/cumotion & Gendexgrasp/Gendexgrasp网络 - 详解PointNetCVAE流程|Gendexgrasp网络 - 详解PointNetCVAE流程]]
- [[Robotmanipulation - opt_VA/YouKnow3M/cumotion & Gendexgrasp/cumotion & Gendexgrasp 改动一览|cumotion & Gendexgrasp 改动一览]]

## 下一步阅读或验证

将 CVAE 数据流与 cuRobo 的到达轨迹连接，明确谁负责接触、谁负责路径。

返回 [[论文图谱/主题/07 运动规划与控制|07 运动规划与控制]] · [[00-论文知识图谱|知识图谱首页]]
