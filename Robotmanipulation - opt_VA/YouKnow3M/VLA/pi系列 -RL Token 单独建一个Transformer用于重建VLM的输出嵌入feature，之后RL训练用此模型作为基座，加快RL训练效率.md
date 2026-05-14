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


## QA环节
1、因为本身最开始动作不稳定，所以SAC的loss方程里面TD Target包含了一个熵项，鼓励策略动作自由探索，但是到了RL Token这里，把熵项去掉了，在我看来其实本质上就是不再鼓励动作出现偏差，而是在现有动作的情况下不断拟合Q值最大的动作分布（相当于最优动作裁剪），我不知道我本身的这个理解对不对

2、然后还有一点就是关于在训练VLA这里，新增了一个模型为RL Token，本质上是transformer encoder 和 transformer decoder + MLP线性投影，它的作用就是学习到如何重建原始的VLA嵌入特征，就是VLA嵌入到action expert之间的中间特征（这说法对吗？其实我有点看不懂这句话它将一个学习到的嵌入附加到 Token 序列中，并通过一个轻量级编码器 Transformer g_\phigϕg_\phigϕ​ 处理增强后的序列），然后相当于就是把VLM给压缩了，在后面在线强化学习训练的时候，作为actor的encoder输入（输入状态s，输出重建后的VLM特征），或者是critic encoder的输入（输入状态s，输出重建后的VLM特征）

我的说法和理解对吗？请你检查一下

我先把你引用的 SAC（HIL-SERL）实现里的关键损失看了一遍，再结合 RLT 论文的描述来逐条点评。

## 一、关于"去掉熵项"的理解

你的方向是对的，但把它说成"最优动作裁剪"其实并不准确。我帮你把这里的逻辑捋一下。

### HIL-SERL（你贴的 modeling_sac.py）里熵项的位置

在你这份 SAC 里，熵项一共出现在**两个**地方：

1. TD target 里（叫 backup entropy）：

```398:403:lerobot/lerobot/common/policies/sac/modeling_sac.py
        min_q, _ = q_targets.min(dim=0)
        
        # 6. Q-chunking核心：使用联合对数概率进行熵正则化
        if self.config.use_backup_entropy:
            # 注意：这里使用的是整个动作序列的联合概率，而不是单个动作的概率
            min_q = min_q - (self.temperature * next_log_probs_joint)
```

2. Actor loss 里（最大熵 SAC 的标准写法）：

```686:687:lerobot/lerobot/common/policies/sac/modeling_sac.py
        # 计算标准SAC Actor损失
        sac_actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()
```

也就是说 `L_π = E[α·log π(a|s) - Q(s,a)]`。这里熵的作用本质上是**"让策略尽量随机一些，去碰一碰那些 Q 估计还没覆盖到的动作"**——它是从零探索的支柱。

### RLT 为什么把这一项去掉

RLT 的 actor loss 写成 

$$L_\pi(\theta) = \mathbb{E}\big[-Q_\psi(x, a_{1:C}) + \beta \|a_{1:C} - \tilde a_{1:C}\|^2\big]$$

注意它**只是把"熵正则"替换成了"BC 正则"**，并不是单纯把熵删掉。我觉得更准确的说法是：

- HIL-SERL 没有任何先验，只能用最大熵框架来鼓励探索；
- RLT 已经有了一个相当能用的 VLA 策略 $\pi_{vla}$ 作为"先验/参考策略"，再去鼓励熵就会把动作从这个好用的先验上**推开**，反而有害；
- 因此 RLT 用 $\beta\|a - \tilde a\|^2$ 把策略**拉回到 VLA 提议附近**，本质是 **KL-regularized / behavior-regularized RL**（和 TD3+BC、AWAC、BRAC、DPG+KL 一类方法是一个套路）。

所以更精确地讲：

- 不是"动作已经稳定所以不再鼓励偏差"，而是"我已经有一个高质量先验 $\pi_{vla}$，我希望在它的邻域里去做局部改进"；
- 不是"最优动作裁剪"。"裁剪"暗示一种硬约束（像 PPO clip 那样），而 RLT 用的是**软约束**（二次惩罚），允许策略偏离参考动作但偏得越多惩罚越大；
- 探索其实仍然存在：策略仍然是 $\mathcal N(\mu_\theta(x, \tilde a), \sigma^2 I)$，σ 给了局部噪声，只是探索范围被锁定在 $\tilde a$ 邻域而已。

顺便指出：RLT 论文里还提到 **Reference Action Dropout**（训练时按概率把 $\tilde a$ 置零），这一步就是防止 actor 退化成"无脑拷贝 VLA"。如果你光看 BC 正则项会觉得"那它不就是模仿 VLA 吗"，加上 dropout 之后逻辑就闭环了：有 $\tilde a$ 时贴着它做精修，没 $\tilde a$ 时也得自己能产出动作。

## 二、关于 RL Token 模块的理解

这部分你的整体架构图基本对，但**最后一句话理解错了**，我逐条说。

### 1. "transformer encoder + transformer decoder + MLP" —— 对

$g_\phi$ 是编码器，$d_\phi$ 是解码器，$h_\phi$ 是线性投影头。没问题。

### 2. "学习重建 VLA 嵌入" —— 对，但要明确重建的是什么

你说的"VLA 嵌入到 action expert 之间的中间特征"基本对。更精确地说：

- $z = f(s, \ell; \theta_{vla})$ 是 **VLA 主干（通常是 VLM）最后一层输出的一串 token 嵌入** $z_{1:M}$；
- 在标准 VLA 里，这串 $z$ 会喂给后面的 action expert / action head 来出动作；
- RLT 把这串 $z$ 当作目标，希望用一个**信息瓶颈** $z_{rl}$ 把它重建出来。

所以"VLM 主干输出 → action expert 输入"之间的那一份 token 嵌入就是它要保护的信息。

### 3. 关于 $e_{rl} = e_\phi(<rl>)$ 这句你没看懂的地方

这其实是 **BERT [CLS] 风格的"汇总 token"**。具体来说：

- $<rl>$ 是一个特殊符号；
- $e_\phi(\cdot)$ 是一个**可学习的 embedding 表**，把这个特殊符号查表得到一个向量 $e_{rl}$（就一个向量，单 token）；
- 把这个向量**拼接到 $z_{1:M}$ 后面**，变成长度 $M+1$ 的序列；
- 送进编码器 $g_\phi$，取**第 $M+1$ 个位置的输出**当作 $z_{rl}$：

$$z_{rl} = g_\phi([z_{1:M}, e_{rl}])_{M+1}$$

可以类比成：CLIP / BERT / DINO 里那种放一个 `[CLS]` token，让它通过 self-attention 把整段序列的信息"吸"进来，作为整段序列的紧凑摘要。这里 $z_{rl}$ 就是"对当前观测 + 语言指令的紧凑摘要向量"。

然后 decoder + MLP 用 $z_{rl}$ **自回归地**还原 $z_{1:M}$（注意 $\bar z_i = \text{sg}(z_i)$ 是停止梯度，**梯度不回传到 VLA**，VLA 一直是冻结的）。重建只是"逼着 $z_{rl}$ 尽可能多地保留 $z_{1:M}$ 的任务相关信息"的**预训练辅助目标**而已。

### 4. ❌ "actor/critic 输入是重建后的 VLM 特征" —— 这里说错了

这是关键的纠正点。一旦 RL token 模块训练好之后：

- $\theta_{vla}$ 冻结，$\phi$（含 encoder/decoder/MLP）也冻结；
- **online RL 阶段只用 encoder $g_\phi$ 这条路**，输出 $z_{rl}$；
- **decoder 和 MLP 不会再被调用**，它们只是训练 token 时的"陪练"；
- actor / critic 的输入是 $x = [\,z_{rl},\, s_p\,]$，即 **压缩后的瓶颈向量 + 本体感受**；
- actor / critic **不会看到"重建出来的 $\hat z_{1:M}$"**。重建只是训练阶段用来约束 $z_{rl}$ 信息量的代理任务。

换句话说，整条数据流是：

```
原始观测 s, 语言 ℓ
        │ (VLA 冻结)
        ▼
   z_{1:M}  ←─ 原始 VLM 特征 (高维, 长序列)
        │ + 可学习 token e_rl
        ▼ (encoder g_φ, 冻结)
       z_rl  ←─ 紧凑摘要向量 ★ 这就是 RL 的状态输入
        │
        ├──► [z_rl, s_p] ──► actor / critic
        │
        └──► (训练时) decoder + MLP → 重建 z_{1:M}（仅训练用）
```

所以你那句

> "作为 actor 的 encoder 输入（输入状态 s，输出重建后的 VLM 特征）"

应当改成：

> **"actor / critic 直接以 RL token 模块 encoder 的输出 $z_{rl}$（加上本体感受 $s_p$）作为输入。decoder 和 MLP 只在 RL token 预训练阶段使用，online RL 时不参与前向。"**


---
## 总结一下

| 你的理解 | 评价 | 修正 |
|---|---|---|
| 去掉熵 = 不再鼓励偏差，对当前动作做最优拟合 | 方向对，但表述不准 | 是把"熵正则"换成了"对 VLA 参考动作的 BC 正则"，属于 KL/behavior-regularized RL；探索仍由 Gaussian σ 提供，只是被局限在 VLA 邻域 |
| RL token 模块 = encoder + decoder + MLP，重建 VLA 嵌入 | 对 | 重建的是 VLM 主干输出、喂给 action expert 之前的那一份 token 嵌入 $z_{1:M}$ |
| `<rl>` token 是什么 | 没看懂 | 就是一个 BERT/CLS 风格的可学习汇总 token，单个向量，拼到 token 序列末尾，让 self-attention 把整段序列信息聚合进它对应的位置 |
| actor/critic 输入是"重建后的 VLM 特征" | ❌ 错 | 输入是 encoder 的瓶颈输出 $z_{rl}$（再拼上 proprioception $s_p$），decoder 训练完就不再使用 |

如果你之后要在 lerobot 里把 HIL-SERL 改造成 RLT 风格，关键改动其实就是三件事：把 actor loss 里的 `temperature * log_probs` 项替换成 `β * ||a - ã||²`、关掉 `use_backup_entropy`、再把 VLA 输出的 `z_{1:M}` 经过一个冻结的 encoder 压成 `z_rl` 作为状态特征输入到 actor/critic（替换/补充现在的 `self.encoder`）。