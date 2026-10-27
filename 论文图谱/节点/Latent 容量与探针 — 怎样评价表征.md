---
type: concept
node_id: latent
topic: wam
status: "原笔记归纳"
updated: 2026-10-27
tags: [论文图谱, wam, concept]
---

# Latent 容量与探针 — 怎样评价表征

## 节点摘要

**概念 · 05 世界模型与长程能力**  
latent 多大才有用，怎样判断里面保留了什么信息？

把 token 数、每 token 维度、时空压缩率与有效信息分开；线性探针测试信息是否容易读出，消融测试某组件是否帮助最终任务。

## 训练、推理与数据流

冻结表征，训练受控探针；调参用验证集，最终报告测试集。再做策略任务消融，避免把探针分数当控制成功率。

## 证据与适用边界

**整理状态：原笔记归纳。**

线性探针失败不证明信息完全不存在；更大 latent 也不保证更强控制。历史模型形状需要随具体版本核验。

## 有理由的关联

| 方向 | 关系 | 节点 | 为什么连接 |
|---|---|---|---|
| 本节点 → | 评测 | [[论文图谱/节点/JEPA-WAM — 联合嵌入世界动作模型\|JEPA-WAM — 联合嵌入世界动作模型]] | 探针与组件消融分工 |
| 本节点 → | 评测 | [[论文图谱/节点/DiT4DiT — 视频与动作的级联建模\|DiT4DiT — 视频与动作的级联建模]] | token 数、压缩与读出信息 |

## 原始笔记

- [[Robotmanipulation - opt_VA/YouKnow3M/VA & CVAE/latent space dim的思考|latent space dim的思考]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/JEPA-WAM 补充 - 消融实验清单 与「探针 probe」到底是什么|JEPA-WAM 补充 - 消融实验清单 与「探针 probe」到底是什么]]
- [[RobotVLA/Vision - Encoder对比|Vision - Encoder对比]]

## 下一步阅读或验证

固定输入分辨率和计算预算，对 token 数、维度、时间间隔各做单因素比较。

返回 [[论文图谱/主题/05 世界模型与长程能力|05 世界模型与长程能力]] · [[00-论文知识图谱|知识图谱首页]]
