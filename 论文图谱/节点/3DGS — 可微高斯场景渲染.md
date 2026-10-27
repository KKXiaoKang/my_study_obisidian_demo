---
type: paper
node_id: gs
topic: vision
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, vision, paper]
---

# 3DGS — 可微高斯场景渲染

## 节点摘要

**论文 · 06 三维感知与重建**  
怎样从多视图图像得到可快速渲染的新视角场景表示？

用带位置、协方差、不透明度和颜色参数的三维高斯表示场景，结合可见性处理的 splatting 渲染和密度控制优化。

## 训练、推理与数据流

利用相机位姿和稀疏点等初始化，依据图像重建误差更新高斯；原笔记补充视频抽帧、COLMAP、训练与可视化流程。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

渲染高质量不代表表面水密、尺度准确或具备碰撞能力；导入 Omniverse 用作背景也不能替代物理资产。

**对应文献**：3D Gaussian Splatting for Real-Time Radiance Field Rendering  
**作者**：Bernhard Kerbl et al.  
**首次预印本年份**：2023（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2308.04079)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 本节点 → | 扩展 | [[论文图谱/节点/SuGaR — 表面对齐高斯与网格\|SuGaR — 表面对齐高斯与网格]] | 表面对齐后提取网格 |
| 本节点 → | 扩展 | [[论文图谱/节点/Gaussian Opacity Fields — 从高斯提取表面\|Gaussian Opacity Fields — 从高斯提取表面]] | 从 opacity field 定义表面 |
| 本节点 → | 应用 | [[论文图谱/节点/HI-SLAM2 — 几何感知高斯 SLAM\|HI-SLAM2 — 几何感知高斯 SLAM]] | 高斯作为 SLAM 地图表示 |

## 原始笔记

- [[RobotVision/三维重建/3D_Gaussian/3D Gaussian Splatting 原理|3D Gaussian Splatting 原理]]
- [[RobotVision/三维重建/3D_Gaussian/3D GS 训练过程 - 3DGS散点ply训练|3D GS 训练过程 - 3DGS散点ply训练]]
- [[RobotVision/三维重建/3D_Gaussian/3D GS 导入场景 Omniverse - 仅用于背景像素渲染|3D GS 导入场景 Omniverse - 仅用于背景像素渲染]]
- [[RobotVision/三维重建/3D_Gaussian/splatviz - 高斯离线本地可视化渲染器|splatviz - 高斯离线本地可视化渲染器]]

## 下一步阅读或验证

沿 SuGaR／GOF 看表面提取，再到仿真与 ESDF 节点核验可用几何。

返回 [[论文图谱/主题/06 三维感知与重建|06 三维感知与重建]] · [[00-论文知识图谱|知识图谱首页]]
