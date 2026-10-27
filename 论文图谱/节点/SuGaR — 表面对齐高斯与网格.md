---
type: paper
node_id: sugar
topic: vision
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, vision, paper]
---

# SuGaR — 表面对齐高斯与网格

## 节点摘要

**论文 · 06 三维感知与重建**  
怎样从无序高斯中提取可编辑网格？

加入表面对齐正则，利用高斯结构提取网格，并可将高斯绑定到网格继续优化渲染。

## 训练、推理与数据流

先训练／调整高斯，再提取表面，按需要联合细化。原笔记补充了本地导出参数和软件环境。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

笔记中的 CUDA 依赖是历史环境记录，不是方法永远不变的要求；网格几何、颜色、UV 和碰撞表示需分别验收。

**对应文献**：SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering  
**作者**：Antoine Guédon and Vincent Lepetit  
**首次预印本年份**：2023（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2311.12775)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 扩展 | [[论文图谱/节点/3DGS — 可微高斯场景渲染\|3DGS — 可微高斯场景渲染]] | 表面对齐后提取网格 |
| 双向比较 | 对比 | [[论文图谱/节点/Gaussian Opacity Fields — 从高斯提取表面\|Gaussian Opacity Fields — 从高斯提取表面]] | 表面定义与提取方式 |
| 双向比较 | 对比 | [[论文图谱/节点/AGS-Mesh 与 DN-Splatter — 几何监督重建\|AGS-Mesh 与 DN-Splatter — 几何监督重建]] | 几何监督与网格提取 |
| 本节点 → | 应用 | [[论文图谱/节点/仿真资产与工具 — 从重建到实验环境\|仿真资产与工具 — 从重建到实验环境]] | 网格导入和物理验收 |

## 原始笔记

- [[RobotVision/三维重建/3D_Gaussian/SuGaR - GS To Mesh|SuGaR - GS To Mesh]]
- [[RobotVision/工具流讲解/PLY 带颜色点云文件 如何 转换为obj+纹理材质|PLY 带颜色点云文件 如何 转换为obj+纹理材质]]

## 下一步阅读或验证

与 GOF 比较表面定义和提取方式，再检查在仿真中的尺度与碰撞效果。

返回 [[论文图谱/主题/06 三维感知与重建|06 三维感知与重建]] · [[00-论文知识图谱|知识图谱首页]]
