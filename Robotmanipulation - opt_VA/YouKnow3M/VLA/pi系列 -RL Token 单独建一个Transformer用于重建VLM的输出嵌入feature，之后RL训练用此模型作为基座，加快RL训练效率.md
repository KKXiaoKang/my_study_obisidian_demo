![[RL Token Bootstrapping Online RL with.pdf]]

RL Token（RLT）是一项旨在通过在线强化学习（RL）微调预训练Vision-Language-Action (VLA) 模型以实现高精度机器人操作的轻量级方法。尽管VLA模型能够学习多样化的操作技能，但在实际任务中，它们往往在精度和速度方面遇到挑战，尤其是在“最后一毫米”的执行阶段。传统的RL方法对于大型基础模型而言计算和样本效率低下，而数据高效的实时RL方法则通常训练较小的模型，牺牲了VLA的泛化能力。RLT旨在弥合这一差距，它在几小时甚至几分钟的真实机器人实践中，实现了对VLA模型的高效在线RL微调，从而在不牺牲VLA泛化能力的前提下提升其性能。

RLT的核心方法包括两个主要组成部分：
1.  **RL token的暴露（Exposure of RL token）**：RLT通过在预训练VLA模型中引入一个“RL token”来解决状态表示问题。VLA的内部特征通常是高维的，不适合直接用于样本高效的RL。RLT为此设计了一个编码器-解码器 Transformer。该 Transformer 将VLA模型最终层产生的 Token 嵌入 $z = f(s, \ell; \theta_{vla})$（其中 $s$ 为状态，$ℓ$ 为语言指令，$\theta_{vla}$ 为VLA参数）作为输入。它将一个学习到的嵌入 $e_{rl} = e_\phi(\text{<rl>})$ 附加到 Token 序列中，并通过一个轻量级编码器 Transformer $g_\phi$ 处理增强后的序列。编码器在特殊 Token 位置的输出，即 $z_{rl} = g_\phi([z_{1:M}, e_{rl}])_{M+1}$，便成为紧凑的RL token。接着，一个解码器 Transformer $d_\phi$ 和一个线性输出投影 $h_\phi$ 被训练来从 $z_{rl}$ 自回归地重建原始的VLA嵌入。训练目标是最小化重建损失 $L_{ro} = E_{\mathcal{D}}\left[\sum_{i=1}^M \left\|h_\phi d_\phi([z_{rl}, \bar{z}_{1:i-1}])_i - \bar{z}_i\right\|^2\right]$，其中 $\bar{z}_i = \text{sg}(z_i)$ 表示对VLA嵌入应用停止梯度操作。RL token作为瓶颈，旨在保留任务相关的预训练知识，并作为在线RL的有效接口。一旦训练完成，VLA模型参数 $\theta_{vla}$ 和RL token的参数 $\phi$ 都将被冻结。

2.  **轻量级在线RL训练（Lightweight Online RL Training）**：在RL token生成模块冻结后，RLT在线训练轻量级的actor（$\pi_\theta$）和critic（$Q_\psi$）网络。
    *   **输入（Input）**：actor和critic的输入 $x$ 结合了RL token $z_{rl}$ 和额外的本体感受状态信息 $s_p$（例如关节位置、末端执行器姿态）。
    *   **分块动作（Chunked Actions）**：与单步动作方法不同，RLT在动作分块（action chunks）上操作，这与VLA模型的原生动作接口对齐。机器人控制频率为50Hz，一个H=50步的动作分块对应1秒的控制。RLT的RL分块长度 $C$ 小于VLA的预测长度 $H$（例如 $C=10$），这使得策略更具反应性。
    *   **Critic训练（Critic Training）**：Critic网络 $Q_\psi(x, a_{1:C})$ 估计状态-动作分块的价值。它使用标准的异策略时序差分学习（temporal-difference learning）进行训练，利用从经验回放缓冲区 $\mathcal{B}$ 中采样的转换。损失函数为 $L_Q = E_{(x,a_{1:C},x')\sim \mathcal{B}}\left[\left\|\hat{Q} - Q_\psi(x, a_{1:C})\right\|^2\right]$，其中目标价值 $\hat{Q} = \sum_{t'=1}^C \gamma^{t'-1}r_{t'} + \gamma^C E_{a'\sim\pi_\theta}[Q_{\psi'}(x', a')]$。这里，$x'$ 是下一个输入状态，$a'$ 是从RL策略中采样得到的动作分块，$\gamma$ 是折扣因子。实践中，遵循TD3算法，使用目标网络 $\psi'$。
    *   **Actor训练（Actor Training）**：Actor网络 $\pi_\theta(\cdot|x, \tilde{a}_{1:C})$ 生成动作分块的高斯分布，即 $\pi_\theta(a_{1:C} | x, \tilde{a}_{1:C}) = \mathcal{N}(\mu_\theta(x, \tilde{a}_{1:C}), \sigma^2I)$。关键在于，actor以VLA模型提出的参考动作分块 $\tilde{a}_{1:C}$ 为条件进行训练，这意味着在线RL是在一个强大的初始提议基础上进行细化，而不是从零开始学习。为了进一步稳定学习，actor被正则化以保持与参考动作分块的接近度，类似于KL正则化RL方法。Actor的优化目标是最大化critic价值，同时约束其接近VLA参考动作：$L_\pi(\theta) = E_{s\sim\mathcal{B}, a_{1:C}\sim\pi_\theta}[-Q_\psi(x, a_{1:C}) + \beta \|a_{1:C} - \tilde{a}_{1:C}\|^2_2]$，其中 $\tilde{a}_{1:C} \sim \pi_{vla}(\cdot | s, \ell)$，系数 $\beta$ 控制正则化强度。
    *   **参考动作丢弃（Reference Action Dropout）**：为了防止actor仅仅复制参考动作而不主动改进，训练批次中的随机一部分转换会丢弃参考动作（替换为零）。这迫使actor在没有VLA提议时也能独立生成动作。

RLT的训练流程包括：首先进行热身阶段，使用基础VLA策略收集经验填充回放缓冲区。然后，训练在收集机器人经验和从回放缓冲区进行异策略actor-critic更新之间交替进行。回放缓冲区聚合了VLA热身数据、在线RL运行数据和可选的人工干预数据。人工操作员提供稀疏的成功/失败标签。为了提高样本效率，RLT在训练时会子采样动作分块，例如每2个控制步存储一个分块。RLT特别强调针对任务中的关键高精度阶段进行改进，通过允许操作员选择何时将控制权从基础VLA切换到RL策略，从而将数据收集和信用分配集中在最需要在线适应的行为部分。

实验评估在四个需要高精度和灵巧控制的真实世界任务上进行：螺丝安装（screw installation）、扎带紧固（zip tie fastening）、以太网插入（Ethernet insertion）和充电器插入（charger insertion）。
*   **性能提升**：RLT显著提高了所有任务的成功率和执行速度，在关键阶段的速度提升高达3倍。即使对于VLA模型已经相当熟练的任务，RLT也能进一步提升吞吐量。
*   **与基线对比**：与HIL-SERL和PLD等单步在线RL方法相比，RLT表现出显著优势，因为单步方法在长任务中难以有效传播稀疏奖励信号。DAgger和DSRL在成功率上与RLT接近，但在速度上远不及RLT，因为DAgger受限于演示速度，而DSRL则强加了过于严格的策略约束。
*   **组件贡献**：RL token、分块动作、BC正则化器和参考动作传递（pass-through）对RLT的性能都至关重要。移除任何一个组件都会导致性能下降，其中移除BC正则化器导致性能下降最大。
*   **新兴策略**：RLT不仅模仿演示，还能发现更高效的新策略。例如，在以太网任务中，RLT学习了一种更流畅、更果断的插入动作，甚至超过了人类专家操作员的速度，而基础VLA模型则表现出犹豫不决的“探测”行为。

RLT的局限性在于它仍需要人工干预来提供奖励信号、修正操作以及在RL策略和基础策略之间进行切换。未来的工作方向包括开发更自主的RL改进流程，例如通过奖励模型和进度预测实现自动化。RLT为机器人系统提供了一个实用路径，使其不仅能从演示中学习，还能在工作中直接改进，从而最终通过RL发现最成功和高性能的策略。