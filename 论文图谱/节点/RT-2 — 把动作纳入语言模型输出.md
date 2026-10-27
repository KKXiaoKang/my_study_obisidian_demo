---
type: paper
node_id: rt2
topic: vla
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, vla, paper]
---

# RT-2 — 把动作纳入语言模型输出

## 节点摘要

**论文 · 02 模仿学习与 VLA**  
互联网视觉语言知识能否迁移到机器人控制？

把机器人动作表达为文本 token，将机器人轨迹与视觉语言任务共同微调，使同一模型同时处理语言与控制输出。

## 训练、推理与数据流

图像和指令输入 VLM，模型预测动作 token，再解码为机器人动作。评估需要区分新对象、语义指令和新物理技能。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

语义推理能力不代表无需机器人数据就能掌握任意新技能；token 化也不等于连续动作专家的去噪过程。

**对应文献**：RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control  
**作者**：Anthony Brohan et al.  
**首次预印本年份**：2023（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2307.15818)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 双向比较 | 对比 | [[论文图谱/节点/VIMA — 多模态提示操控\|VIMA — 多模态提示操控]] | 文本动作与多模态提示 |
| 双向比较 | 对比 | [[论文图谱/节点/OpenVLA — 开源视觉语言动作模型\|OpenVLA — 开源视觉语言动作模型]] | 视觉语言知识到动作 token |

## 原始笔记

- [[RobotVLA/VIMA vs RT2|VIMA vs RT2]]
- [[Robotmanipulation - opt_VA/Imitation learning/※ OpenVLA - Vision-Language-Action Model|※ OpenVLA - Vision-Language-Action Model]]

## 下一步阅读或验证

与 VIMA 比较提示形式，与 OpenVLA 比较开放实现，与 π0 比较连续动作输出。

返回 [[论文图谱/主题/02 模仿学习与 VLA|02 模仿学习与 VLA]] · [[00-论文知识图谱|知识图谱首页]]
