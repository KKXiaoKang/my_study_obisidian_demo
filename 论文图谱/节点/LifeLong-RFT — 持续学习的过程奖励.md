---
type: paper
node_id: lifelong
topic: posttrain
status: "原始摘要已核对；细节见原笔记"
updated: 2026-10-27
tags: [论文图谱, posttrain, paper]
---

# LifeLong-RFT — 持续学习的过程奖励

## 节点摘要

**论文 · 04 VLA 后训练与奖励**  
如何在持续适配新任务时减少示教需求和旧能力遗忘？

以动作块为粒度，结合离散动作一致性、连续轨迹对齐和格式合规三个过程奖励，不依赖在线环境反馈。

## 训练、推理与数据流

生成动作块，与参考轨迹／输出格式比较形成奖励，再做强化微调；持续任务序列应单独评测。

## 证据与适用边界

**整理状态：原始摘要已核对；细节见原笔记。**

不使用在线环境回报不等于完全没有监督来源；参考动作仍提供关键训练信号。

**对应文献**：Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning  
**作者**：作者名单见原文  
**首次预印本年份**：2026（不等同于会议发表年份）。

原始来源：[论文／作者项目入口](https://arxiv.org/abs/2602.10503)。本次核对书目身份和摘要级主张，未宣称完成全文复核。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 双向比较 | 对比 | [[论文图谱/节点/iRe-VLA — RL 与监督学习交替\|iRe-VLA — RL 与监督学习交替]] | 交互回放与参考过程奖励 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/VLA need RL, not just SFT|VLA need RL, not just SFT]]

## 下一步阅读或验证

对比新任务学习速度、旧任务遗忘和各过程奖励的独立贡献。

返回 [[论文图谱/主题/04 VLA 后训练与奖励|04 VLA 后训练与奖励]] · [[00-论文知识图谱|知识图谱首页]]
