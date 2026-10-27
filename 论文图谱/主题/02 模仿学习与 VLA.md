---
type: topic
tags: [论文图谱, vla]
---

# 02 模仿学习与 VLA

> 如何从示教、图像与语言生成机器人动作？

按动作表示比较 ACT、Diffusion Policy、离散 token VLA 和连续流模型；不要只按模型规模排列。

打开 [[论文图谱/分图/02 模仿学习与 VLA.canvas|本主题节点图]] · 返回 [[00-论文知识图谱|总览]]。

## 论文与方法节点

| 节点 | 类型 | 核心问题 | 证据层级 |
|---|---|---|---|
| [[论文图谱/节点/行为克隆 — 策略表示与监督基线\|行为克隆 — 策略表示与监督基线]] | 概念 | 如何从专家的观测—动作对直接学习控制策略？ | 原笔记归纳 |
| [[论文图谱/节点/ACT — 动作块模仿学习\|ACT — 动作块模仿学习]] | 论文 | 低成本双臂硬件上，怎样从少量示教学习精细且时序一致的动作？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/Diffusion Policy — 扩散动作策略\|Diffusion Policy — 扩散动作策略]] | 论文 | 怎样避免单一动作回归平均掉多种可行策略，并生成连贯的动作序列？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/GenDP — 三维语义场与扩散策略\|GenDP — 三维语义场与扩散策略]] | 论文 | 如何把几何和语义对应关系用于新物体实例上的操作泛化？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/OpenVLA — 开源视觉语言动作模型\|OpenVLA — 开源视觉语言动作模型]] | 论文 | 如何将通用视觉语言预训练能力转成可微调的机器人策略？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/RT-2 — 把动作纳入语言模型输出\|RT-2 — 把动作纳入语言模型输出]] | 论文 | 互联网视觉语言知识能否迁移到机器人控制？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/VIMA — 多模态提示操控\|VIMA — 多模态提示操控]] | 论文 | 语言、目标图片和示范等任务描述能否统一成一个提示接口？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/Octo — 可适配的通用机器人策略\|Octo — 可适配的通用机器人策略]] | 论文 | 通用策略如何适配不同传感器、机器人动作空间和新任务？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/π0 — 视觉语言动作流模型\|π0 — 视觉语言动作流模型]] | 论文 | 怎样结合 VLM 的知识与连续高维动作的生成能力？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/π0.5 — 异构协同训练与开放环境泛化\|π0.5 — 异构协同训练与开放环境泛化]] | 论文 | 如何把多机器人、网页知识和高层子任务信息用于新环境中的长程操作？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/GR00T N1 与 N1.5 — 通用模型及实现分支\|GR00T N1 与 N1.5 — 通用模型及实现分支]] | 论文 | 如何将视觉语言理解与连续动作生成耦合，用于人形机器人的多任务操控？ | 原始摘要已核对；细节见原笔记 |
| [[论文图谱/节点/Alpamayo-R1 — 推理与动作预测\|Alpamayo-R1 — 推理与动作预测]] | 论文 | 显式推理信息怎样与可执行的轨迹预测结合？ | 原笔记归纳；原文细节待核对 |
| [[论文图谱/节点/Human-to-Robot Transfer — 人类示教迁移\|Human-to-Robot Transfer — 人类示教迁移]] | 论文 | 人类视频能否帮助机器人完成新的语义组合和任务结构？ | 原笔记归纳；原文细节待核对 |
| [[论文图谱/节点/X-ICM 与 AGNOSTOS — 跨任务泛化\|X-ICM 与 AGNOSTOS — 跨任务泛化]] | 论文 | 已有任务示例如何帮助模型在未见任务中执行操作？ | 原笔记归纳；原文细节待核对 |

## 跨主题桥梁

- [[论文图谱/节点/Transformer — 注意力与信息通路|Transformer — 注意力与信息通路]] → **基础** → [[论文图谱/节点/ACT — 动作块模仿学习|ACT — 动作块模仿学习]]：动作块模型使用注意力结构。
- [[论文图谱/节点/VAE 与 CVAE — 潜变量和条件生成|VAE 与 CVAE — 潜变量和条件生成]] → **基础** → [[论文图谱/节点/ACT — 动作块模仿学习|ACT — 动作块模仿学习]]：CVAE 潜变量与重构正则。
- [[论文图谱/节点/Transformer — 注意力与信息通路|Transformer — 注意力与信息通路]] → **基础** → [[论文图谱/节点/π0.5 — 异构协同训练与开放环境泛化|π0.5 — 异构协同训练与开放环境泛化]]：用 QKV 与 mask 理解通路。
- [[论文图谱/节点/Flow Matching — 条件速度场|Flow Matching — 条件速度场]] → **基础** → [[论文图谱/节点/π0 — 视觉语言动作流模型|π0 — 视觉语言动作流模型]]：连续动作的向量场生成。
- [[论文图谱/节点/Flow Matching — 条件速度场|Flow Matching — 条件速度场]] → **基础** → [[论文图谱/节点/GR00T N1 与 N1.5 — 通用模型及实现分支|GR00T N1 与 N1.5 — 通用模型及实现分支]]：理解笔记中的 flow action head。
- [[论文图谱/节点/Diffusion Policy — 扩散动作策略|Diffusion Policy — 扩散动作策略]] → **对比** → [[论文图谱/节点/Flow Matching — 条件速度场|Flow Matching — 条件速度场]]：路径、目标与采样器。
- [[论文图谱/节点/ACT — 动作块模仿学习|ACT — 动作块模仿学习]] → **关联** → [[论文图谱/节点/Q-chunking — 动作块空间中的 RL|Q-chunking — 动作块空间中的 RL]]：共享动作块思想，训练目标不同。
- [[论文图谱/节点/Diffusion Policy — 扩散动作策略|Diffusion Policy — 扩散动作策略]] → **扩展** → [[论文图谱/节点/DPPO — 扩散策略的策略梯度优化|DPPO — 扩散策略的策略梯度优化]]：用策略梯度改进扩散策略。
- [[论文图谱/节点/Diffusion Policy — 扩散动作策略|Diffusion Policy — 扩散动作策略]] → **扩展** → [[论文图谱/节点/DSRL — 在扩散噪声空间中学习引导|DSRL — 在扩散噪声空间中学习引导]]：冻结生成器，优化噪声选择。
- [[论文图谱/节点/π0.5 — 异构协同训练与开放环境泛化|π0.5 — 异构协同训练与开放环境泛化]] → **关联** → [[论文图谱/节点/RECAP 与 π*0.6 — 从经验和纠正学习|RECAP 与 π*0.6 — 从经验和纠正学习]]：VLA 从经验中改进的阅读顺序。
- [[论文图谱/节点/GR00T N1 与 N1.5 — 通用模型及实现分支|GR00T N1 与 N1.5 — 通用模型及实现分支]] → **方案** → [[论文图谱/节点/GR00T 优势条件化 — 后训练方案|GR00T 优势条件化 — 后训练方案]]：向本地 flow head 注入优势条件。
- [[论文图谱/节点/OpenVLA — 开源视觉语言动作模型|OpenVLA — 开源视觉语言动作模型]] → **关联** → [[论文图谱/节点/SimpleVLA-RL — 可扩展的 VLA 强化学习|SimpleVLA-RL — 可扩展的 VLA 强化学习]]：OpenVLA-OFT 为后续变体，核对动作 head。
- [[论文图谱/节点/行为克隆 — 策略表示与监督基线|行为克隆 — 策略表示与监督基线]] → **扩展** → [[论文图谱/节点/iRe-VLA — RL 与监督学习交替|iRe-VLA — RL 与监督学习交替]]：将监督阶段与 RL 交替。
- [[论文图谱/节点/π0.5 — 异构协同训练与开放环境泛化|π0.5 — 异构协同训练与开放环境泛化]] → **扩展** → [[论文图谱/节点/JEPA-WAM — 联合嵌入世界动作模型|JEPA-WAM — 联合嵌入世界动作模型]]：预测性辅助监督的迁移实例。
- [[论文图谱/节点/GR00T N1 与 N1.5 — 通用模型及实现分支|GR00T N1 与 N1.5 — 通用模型及实现分支]] → **方案** → [[论文图谱/节点/WAM 融合方案 — 从设想到可检验假设|WAM 融合方案 — 从设想到可检验假设]]：动作专家与多视觉流融合。
- [[论文图谱/节点/π0.5 — 异构协同训练与开放环境泛化|π0.5 — 异构协同训练与开放环境泛化]] → **关联** → [[论文图谱/节点/MEM 与 π0.7 — 长程记忆和可引导性待核对|MEM 与 π0.7 — 长程记忆和可引导性待核对]]：高层子任务与长程能力的分解。
- [[论文图谱/节点/ACT — 动作块模仿学习|ACT — 动作块模仿学习]] → **方案** → [[论文图谱/节点/Deep Reactive Policy — 全局意图与局部避障|Deep Reactive Policy — 全局意图与局部避障]]：学习的全局意图与局部反应接口。
- [[论文图谱/节点/GR00T N1 与 N1.5 — 通用模型及实现分支|GR00T N1 与 N1.5 — 通用模型及实现分支]] → **方案** → [[论文图谱/节点/动作头分解 — 尺度、共享与双臂协调|动作头分解 — 尺度、共享与双臂协调]]：按动作语义拆解输出头。
- [[论文图谱/节点/LeRobot 与 RL 工程链路|LeRobot 与 RL 工程链路]] → **工程** → [[论文图谱/节点/ACT — 动作块模仿学习|ACT — 动作块模仿学习]]：数据归一化与动作接口。
- [[论文图谱/节点/ACT — 动作块模仿学习|ACT — 动作块模仿学习]] → **扩展** → [[论文图谱/节点/rewACT — 动作策略上的奖励辅助头|rewACT — 动作策略上的奖励辅助头]]：本地奖励辅助头实验。

## 本主题的原始材料

- [[RobotRL/Q-chunk ACT actor/ACT - latenX 推理时设置为None存在的漏洞|ACT - latenX 推理时设置为None存在的漏洞]]
- [[RobotRL/Q-chunk ACT actor/mlp actor 2 act actor - 架构升级说明|mlp actor 2 act actor - 架构升级说明]]
- [[RobotRL/VLA-RL/GROOT N1.5 flow-matching head 如何修改带advantage优势值条件生成|GROOT N1.5 flow-matching head 如何修改带advantage优势值条件生成]]
- [[RobotVLA/NVIDIA/Alpamayo-R1 ：Bridging Reasoning and Action Prediction|Alpamayo-R1 ：Bridging Reasoning and Action Prediction]]
- [[RobotVLA/VIMA vs RT2|VIMA vs RT2]]
- [[RobotVLA/Vision - Encoder对比|Vision - Encoder对比]]
- [[RobotVLA/physicalintelligence/Emergence of Human to Robot Transfer in VLAs|Emergence of Human to Robot Transfer in VLAs]]
- [[RobotVLA/多模态对齐  - 特征融合部分|多模态对齐  - 特征融合部分]]
- [[Robotmanipulation - opt_VA/Imitation learning/Diffusion Policy - Flow Matching 对比|Diffusion Policy - Flow Matching 对比]]
- [[Robotmanipulation - opt_VA/Imitation learning/rewACT - 将MLP结构的reward model融合在ACT训练当中|rewACT - 将MLP结构的reward model融合在ACT训练当中]]
- [[Robotmanipulation - opt_VA/Imitation learning/※ ACT -  Transformer CVAE动作生成器|※ ACT -  Transformer CVAE动作生成器]]
- [[Robotmanipulation - opt_VA/Imitation learning/※ Diffusion Policy - 扩散去噪动作策略模型|※ Diffusion Policy - 扩散去噪动作策略模型]]
- [[Robotmanipulation - opt_VA/Imitation learning/※ GenDP - 机器人操作语义几何表示|※ GenDP - 机器人操作语义几何表示]]
- [[Robotmanipulation - opt_VA/Imitation learning/※ OpenVLA - Vision-Language-Action Model|※ OpenVLA - Vision-Language-Action Model]]
- [[Robotmanipulation - opt_VA/Imitation learning/传统BC方法一览|传统BC方法一览]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/Exploring the Limits of Vision-Language-ActionManipulation in Cross-task Generalization|Exploring the Limits of Vision-Language-ActionManipulation in Cross-task Generalization]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/FM vs DP noise denoise流程分析|FM vs DP noise denoise流程分析]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/GROOT N1.5 action expert改动一览|GROOT N1.5 action expert改动一览]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/Octo vs pi05|Octo vs pi05]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/pi05 - 详细解析joint attn怎么一步+mask实现类似于交叉attn效果|pi05 - 详细解析joint attn怎么一步+mask实现类似于交叉attn效果]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/pi05 vs GROOT|pi05 vs GROOT]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/未来发展 - pi05 GROOT future 范式下视觉-动作联立建模下的流如何融合?|未来发展 - pi05 GROOT future 范式下视觉-动作联立建模下的流如何融合?]]
- [[Robotmanipulation - opt_VA/YouKnow3M/VLA/模型loss差异|模型loss差异]]
- [[Robotmanipulation - opt_VA/显式策略 - 隐式策略 - 扩散策略|显式策略 - 隐式策略 - 扩散策略]]
