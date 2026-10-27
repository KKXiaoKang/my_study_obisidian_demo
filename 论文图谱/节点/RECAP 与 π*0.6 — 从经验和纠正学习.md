---
type: paper
node_id: recap
topic: posttrain
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, posttrain, paper]
---

# RECAP 与 π*0.6 — 从经验和纠正学习

## 节点摘要

**论文 · 04 VLA 后训练与奖励**  
示教、自主尝试和专家纠正怎样共同改进通用 VLA？

RECAP 以 advantage-conditioned policy 组织经验学习，利用异构数据训练和改进 VLA；π*0.6 是对应模型实例。

## 训练、推理与数据流

构建价值／优势信号，训练优势条件策略，并通过机器人交互继续专业化。条件生成与样本加权应分别记录。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

π*0.6、π0.6 和记忆扩展 MEM 不能因为名称相近就合并。添加一个 advantage token 也不等于完整复现 RECAP。

**对应文献**：π*0.6: a VLA That Learns From Experience  
**作者**：作者名单见原文  
**首次预印本年份**：2025（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2511.14759)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 双向比较 | 关联 | [[论文图谱/节点/π0.5 — 异构协同训练与开放环境泛化\|π0.5 — 异构协同训练与开放环境泛化]] | VLA 从经验中改进的阅读顺序 |
| 双向比较 | 对比 | [[论文图谱/节点/ReinFlow — Flow Policy 的在线 RL\|ReinFlow — Flow Policy 的在线 RL]] | 策略梯度与优势条件化 |
| 本节点 → | 借鉴 | [[论文图谱/节点/GR00T 优势条件化 — 后训练方案\|GR00T 优势条件化 — 后训练方案]] | 优势标签和推理条件 |
| 双向比较 | 对比 | [[论文图谱/节点/RA-BC — 用进度信号重加权示教\|RA-BC — 用进度信号重加权示教]] | 样本重加权与优势条件化 |
| 双向比较 | 辨析 | [[论文图谱/节点/MEM 与 π0.7 — 长程记忆和可引导性待核对\|MEM 与 π0.7 — 长程记忆和可引导性待核对]] | π*0.6 与记忆版本命名分开 |

## 原始笔记

- [[RobotRL/VLA-RL/路线总览|路线总览]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/VLA need RL, not just SFT|VLA need RL, not just SFT]]
- [[RobotRL/VLA-RL/GROOT N1.5 flow-matching head 如何修改带advantage优势值条件生成|GROOT N1.5 flow-matching head 如何修改带advantage优势值条件生成]]

## 下一步阅读或验证

核验优势标签来源、推理条件和数据迭代闭环，再对照 GR00T advantage-token 方案。

返回 [[论文图谱/主题/04 VLA 后训练与奖励|04 VLA 后训练与奖励]] · [[00-论文知识图谱|知识图谱首页]]
