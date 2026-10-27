---
type: paper
node_id: openvla
topic: vla
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, vla, paper]
---

# OpenVLA — 开源视觉语言动作模型

## 节点摘要

**论文 · 02 模仿学习与 VLA**  
如何将通用视觉语言预训练能力转成可微调的机器人策略？

原版 OpenVLA 基于 Llama 2，融合 DINOv2 与 SigLIP 视觉特征，学习动作 token 预测；提供大规模机器人数据训练与高效微调路径。

## 训练、推理与数据流

训练以机器人动作监督适配视觉语言模型；部署把预测动作表示转换回控制空间。不同后续版本的 action head 应分别记录。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

原笔记中“CLIP＋独立目标定位模块”等内容不能当作原版 OpenVLA 的标准结构；不要把通用 VLA 方案混入论文贡献。

**对应文献**：OpenVLA: An Open-Source Vision-Language-Action Model  
**作者**：Moo Jin Kim et al.  
**首次预印本年份**：2024（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2406.09246)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| → 本节点 | 实例 | [[论文图谱/节点/行为克隆 — 策略表示与监督基线\|行为克隆 — 策略表示与监督基线]] | 视觉语言动作监督 |
| 双向比较 | 对比 | [[论文图谱/节点/RT-2 — 把动作纳入语言模型输出\|RT-2 — 把动作纳入语言模型输出]] | 视觉语言知识到动作 token |
| 双向比较 | 对比 | [[论文图谱/节点/π0 — 视觉语言动作流模型\|π0 — 视觉语言动作流模型]] | 离散 token 与连续动作流 |
| 双向比较 | 关联 | [[论文图谱/节点/SimpleVLA-RL — 可扩展的 VLA 强化学习\|SimpleVLA-RL — 可扩展的 VLA 强化学习]] | OpenVLA-OFT 为后续变体，核对动作 head |

## 原始笔记

- [[Robotmanipulation - opt_VA/Imitation learning/※ OpenVLA - Vision-Language-Action Model|※ OpenVLA - Vision-Language-Action Model]]
- [[RobotVLA/VIMA vs RT2|VIMA vs RT2]]

## 下一步阅读或验证

按视觉编码器、语言底座、动作表示、训练数据和微调方式与 RT-2、Octo、π0.5 对齐。

返回 [[论文图谱/主题/02 模仿学习与 VLA|02 模仿学习与 VLA]] · [[00-论文知识图谱|知识图谱首页]]
