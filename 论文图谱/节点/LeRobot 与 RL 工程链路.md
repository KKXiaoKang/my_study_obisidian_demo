---
type: engineering
node_id: lerobot
topic: practice
status: "原笔记归纳"
updated: 2026-10-27
tags: [论文图谱, practice, engineering]
---

# LeRobot 与 RL 工程链路

## 节点摘要

**工程 · 08 工程实验与评测**  
数据、归一化、训练循环与执行接口怎样保持一致？

将数据集／视频读取、观测和动作处理、replay、actor／learner、checkpoint 与部署构成一条可追踪链路。

## 训练、推理与数据流

原材料包括空白 Pipeline 入口、SAC 保存恢复、SB3 接口和调试日志。按“输入 → 处理 → 模型 → 动作 → 反馈”补齐索引。

## 证据与适用边界

**整理状态：原笔记归纳。**

原日志里的 tensor 和旧环境输出不是当前运行状态；off-policy 也不等于只能离线训练。版本安装问题应绑定具体依赖。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 本节点 → | 工程 | [[论文图谱/节点/HIL-SERL — 人类干预与样本高效 RL\|HIL-SERL — 人类干预与样本高效 RL]] | 采样、回放与 actor-learner |
| 本节点 → | 工程 | [[论文图谱/节点/ACT — 动作块模仿学习\|ACT — 动作块模仿学习]] | 数据归一化与动作接口 |
| → 本节点 | 工程 | [[论文图谱/节点/工具与项目记录 — 环境及任务背景\|工具与项目记录 — 环境及任务背景]] | 依赖与运行环境 |

## 原始笔记

- [[Robotmanipulation - opt_VA/LeRobot/LeRobot - PipeLine|LeRobot - PipeLine]]
- [[RobotRL/Hil-Serl/lerobot - Hil-Serl pipeline说明|lerobot - Hil-Serl pipeline说明]]
- [[RobotRL/SAC/SAC - `RunningMeanStd` 观测归一化工具|SAC - `RunningMeanStd` 观测归一化工具]]
- [[RobotRL/SAC/SAC - load or save model|SAC - load or save model]]
- [[RobotRL/SAC/SAC - train and learn|SAC - train and learn]]
- [[RobotRL/stable-baselines3 框架解析/common.base_class.BaseAlgorithm|common.base_class.BaseAlgorithm]]
- [[RobotRL/stable-baselines3 框架解析/common.off_policy_algorithm.OffPolicyAlgorithm|common.off_policy_algorithm.OffPolicyAlgorithm]]
- [[RobotRL/0801 - DEBUG LOG - Actor|0801 - DEBUG LOG - Actor]]
- [[RobotRL/0801 - DEBUG LOG - Learner|0801 - DEBUG LOG - Learner]]
- [[RobotRL/0801 - KuavoEnv Debug log|0801 - KuavoEnv Debug log]]
- [[Robot工具/关于Lerobot 中 torchcodec 视频编码Tensor工具库难装的问题|关于Lerobot 中 torchcodec 视频编码Tensor工具库难装的问题]]

## 下一步阅读或验证

用一条真实样本追踪 shape、dtype、时间戳、归一化与动作单位，再检查恢复后的行为一致性。

返回 [[论文图谱/主题/08 工程实验与评测|08 工程实验与评测]] · [[00-论文知识图谱|知识图谱首页]]
