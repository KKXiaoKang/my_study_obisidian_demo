---
type: topic
tags: [论文图谱, posttrain]
---

# 04 VLA 后训练与奖励

> 怎样利用失败、进度和在线经验改进已有策略？

先看监督信号，再看优化接口：样本重加权、优势条件化、噪声空间引导、策略梯度与轻量 actor-critic。

打开 [[论文图谱/分图/04 VLA 后训练与奖励.canvas|本主题节点图]] · 返回 [[00-论文知识图谱|总览]]。

## 论文与方法节点

| 节点 | 类型 | 核心问题 | 证据层级 |
|---|---|---|---|
| [[论文图谱/节点/DPPO — 扩散策略的策略梯度优化\|DPPO — 扩散策略的策略梯度优化]] | 论文 | 如何对已经用示教训练好的扩散动作策略做在线强化学习？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/DSRL — 在扩散噪声空间中学习引导\|DSRL — 在扩散噪声空间中学习引导]] | 论文 | 已有扩散策略表现不足时，能否不修改其参数而用 RL 改进行为？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/ReinFlow — Flow Policy 的在线 RL\|ReinFlow — Flow Policy 的在线 RL]] | 论文 | 如何为确定性 flow 生成路径构造便于 RL 的随机策略和概率计算？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/RECAP 与 π*0.6 — 从经验和纠正学习\|RECAP 与 π*0.6 — 从经验和纠正学习]] | 论文 | 示教、自主尝试和专家纠正怎样共同改进通用 VLA？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/ConRFT — 一致性策略的强化微调\|ConRFT — 一致性策略的强化微调]] | 论文 | 有限且质量不一的示教怎样启动接触任务中的 VLA 在线改进？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/RL Token — 轻量强化学习接口\|RL Token — 轻量强化学习接口]] | 论文 | 如何复用大 VLA 的知识，同时降低在线 RL 的模型更新成本？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/SARM — 阶段感知的进度奖励\|SARM — 阶段感知的进度奖励]] | 论文 | 不同长度、质量和动作顺序的示教，怎样获得可比较的任务进度？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/RA-BC — 用进度信号重加权示教\|RA-BC — 用进度信号重加权示教]] | 方法 | 如何从包含低效或失败片段的示教中突出更有价值的动作？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/GVL — 用 VLM 估计任务进展\|GVL — 用 VLM 估计任务进展]] | 论文 | 能否利用 VLM 的语义理解，在少量任务条件下评估状态的价值或进度？ | 原笔记归纳；原文细节待核对 |
| [[论文图谱/节点/SimpleVLA-RL — 可扩展的 VLA 强化学习\|SimpleVLA-RL — 可扩展的 VLA 强化学习]] | 论文 | 如何提高 VLA 强化学习中的采样、渲染与优化吞吐？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/iRe-VLA — RL 与监督学习交替\|iRe-VLA — RL 与监督学习交替]] | 论文 | 在线 RL 的探索收益怎样与大 VLA 训练稳定性兼顾？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/VLA-RFT — 世界模拟器中的强化微调\|VLA-RFT — 世界模拟器中的强化微调]] | 论文 | 真实交互成本高时，能否用数据训练的世界模型提供策略练习环境？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/LifeLong-RFT — 持续学习的过程奖励\|LifeLong-RFT — 持续学习的过程奖励]] | 论文 | 如何在持续适配新任务时减少示教需求和旧能力遗忘？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/χ0（kai0）— 阶段优势与训练部署对齐\|χ0（kai0）— 阶段优势与训练部署对齐]] | 论文 | 长程操作的示教分布、模型偏置和部署状态怎样对齐？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/GR-RL — 从通用策略到精细专家\|GR-RL — 从通用策略到精细专家]] | 论文 | 如何从带噪示教中学习长程、精细的灵巧操作？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/RIPT-VLA、HIL-DAFT 与 RLVLA — 扩展阅读\|RIPT-VLA、HIL-DAFT 与 RLVLA — 扩展阅读]] | 阅读线索 | 交互后训练、人类干预和泛化评测还有哪些对照路线？ | 待补原文与独立精读 |

## 跨主题桥梁

- [[论文图谱/节点/Model-based RL — 模型用于哪里|Model-based RL — 模型用于哪里]] → **应用** → [[论文图谱/节点/VLA-RFT — 世界模拟器中的强化微调|VLA-RFT — 世界模拟器中的强化微调]]：世界模型作为 rollout 环境。
- [[论文图谱/节点/Diffusion Policy — 扩散动作策略|Diffusion Policy — 扩散动作策略]] → **扩展** → [[论文图谱/节点/DPPO — 扩散策略的策略梯度优化|DPPO — 扩散策略的策略梯度优化]]：用策略梯度改进扩散策略。
- [[论文图谱/节点/PPO — 受约束的策略更新|PPO — 受约束的策略更新]] → **基础** → [[论文图谱/节点/DPPO — 扩散策略的策略梯度优化|DPPO — 扩散策略的策略梯度优化]]：概率比与优势估计。
- [[论文图谱/节点/Diffusion Policy — 扩散动作策略|Diffusion Policy — 扩散动作策略]] → **扩展** → [[论文图谱/节点/DSRL — 在扩散噪声空间中学习引导|DSRL — 在扩散噪声空间中学习引导]]：冻结生成器，优化噪声选择。
- [[论文图谱/节点/Flow Matching — 条件速度场|Flow Matching — 条件速度场]] → **扩展** → [[论文图谱/节点/ReinFlow — Flow Policy 的在线 RL|ReinFlow — Flow Policy 的在线 RL]]：为 flow 路径引入训练期随机性。
- [[论文图谱/节点/π0.5 — 异构协同训练与开放环境泛化|π0.5 — 异构协同训练与开放环境泛化]] → **关联** → [[论文图谱/节点/RECAP 与 π*0.6 — 从经验和纠正学习|RECAP 与 π*0.6 — 从经验和纠正学习]]：VLA 从经验中改进的阅读顺序。
- [[论文图谱/节点/HIL-SERL — 人类干预与样本高效 RL|HIL-SERL — 人类干预与样本高效 RL]] → **关联** → [[论文图谱/节点/ConRFT — 一致性策略的强化微调|ConRFT — 一致性策略的强化微调]]：人类干预的在线学习闭环。
- [[论文图谱/节点/SAC — 最大熵 Actor-Critic|SAC — 最大熵 Actor-Critic]] → **基础** → [[论文图谱/节点/RL Token — 轻量强化学习接口|RL Token — 轻量强化学习接口]]：轻量 actor-critic 的价值学习。
- [[论文图谱/节点/RECAP 与 π*0.6 — 从经验和纠正学习|RECAP 与 π*0.6 — 从经验和纠正学习]] → **借鉴** → [[论文图谱/节点/GR00T 优势条件化 — 后训练方案|GR00T 优势条件化 — 后训练方案]]：优势标签和推理条件。
- [[论文图谱/节点/χ0（kai0）— 阶段优势与训练部署对齐|χ0（kai0）— 阶段优势与训练部署对齐]] → **借鉴** → [[论文图谱/节点/GR00T 优势条件化 — 后训练方案|GR00T 优势条件化 — 后训练方案]]：阶段信号与训练部署对齐。
- [[论文图谱/节点/OpenVLA — 开源视觉语言动作模型|OpenVLA — 开源视觉语言动作模型]] → **关联** → [[论文图谱/节点/SimpleVLA-RL — 可扩展的 VLA 强化学习|SimpleVLA-RL — 可扩展的 VLA 强化学习]]：OpenVLA-OFT 为后续变体，核对动作 head。
- [[论文图谱/节点/行为克隆 — 策略表示与监督基线|行为克隆 — 策略表示与监督基线]] → **扩展** → [[论文图谱/节点/iRe-VLA — RL 与监督学习交替|iRe-VLA — RL 与监督学习交替]]：将监督阶段与 RL 交替。
- [[论文图谱/节点/RECAP 与 π*0.6 — 从经验和纠正学习|RECAP 与 π*0.6 — 从经验和纠正学习]] → **辨析** → [[论文图谱/节点/MEM 与 π0.7 — 长程记忆和可引导性待核对|MEM 与 π0.7 — 长程记忆和可引导性待核对]]：π*0.6 与记忆版本命名分开。
- [[论文图谱/节点/rewACT — 动作策略上的奖励辅助头|rewACT — 动作策略上的奖励辅助头]] → **对比** → [[论文图谱/节点/RA-BC — 用进度信号重加权示教|RA-BC — 用进度信号重加权示教]]：辅助预测与样本重加权。
- [[论文图谱/节点/SARM — 阶段感知的进度奖励|SARM — 阶段感知的进度奖励]] → **对比** → [[论文图谱/节点/rewACT — 动作策略上的奖励辅助头|rewACT — 动作策略上的奖励辅助头]]：独立进度模型与共享 reward head。

## 本主题的原始材料

- [[RobotRL/DSRL/DSRL - 基于Diffusion策略的RL微调|DSRL - 基于Diffusion策略的RL微调]]
- [[RobotRL/Diffusion-RL/DPPO - Diffusion Policy Policy Optimization|DPPO - Diffusion Policy Policy Optimization]]
- [[RobotRL/VLA-RL/GROOT N1.5 flow-matching head 如何修改带advantage优势值条件生成|GROOT N1.5 flow-matching head 如何修改带advantage优势值条件生成]]
- [[RobotRL/VLA-RL/路线总览|路线总览]]
- [[Robotmanipulation - opt_VA/Imitation learning/rewACT - 将MLP结构的reward model融合在ACT训练当中|rewACT - 将MLP结构的reward model融合在ACT训练当中]]
- [[Robotmanipulation - opt_VA/VLA/VLM学习价值函数 - Vision Language Models are In-Context Value Learners 谷歌deepmind|VLM学习价值函数 - Vision Language Models are In-Context Value Learners 谷歌deepmind]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/IMG/sarm_reward_model_architecture|sarm_reward_model_architecture]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/RA-BC reward align behavior cloning|RA-BC reward align behavior cloning]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/VLA need RL, not just SFT|VLA need RL, not just SFT]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/pi系列 -RL Token 单独建一个Transformer用于重建VLM的输出嵌入feature，之后RL训练用此模型作为基座，加快RL训练效率|pi系列 -RL Token 单独建一个Transformer用于重建VLM的输出嵌入feature，之后RL训练用此模型作为基座，加快RL训练效率]]
