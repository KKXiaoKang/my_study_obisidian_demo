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
- actor / critic 的输入是 $x = [\,z_{rl},\, s_p\,]$，即 **压缩后的瓶颈向量 + 本体感受**（latent space）；
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

---
# Q：你的decoder是不是摆设来的？
这是个非常好的问题，触到了**表征学习（representation learning）/ 自编码器（autoencoder）这一类方法的核心**。我分几层来回答你。

## 短答

**decoder + MLP 不是"装样子"，它们是 encoder 唯一的训练信号源。** 没有它们，你根本没办法把 encoder 训出来。一旦训完，它们才被丢掉。

## 长答：为什么必须有 decoder？

你可以想象一下：如果只有 encoder，那么 encoder 的训练目标是什么？

$$z_{rl} = g_\phi([z_{1:M}, e_{rl}])_{M+1}$$

这个 encoder 把一串 token 压成一个向量。问题是——**怎么知道压得好不好？怎么定义"好的 $z_{rl}$"？**

可能的几种选项：

| 方案 | 问题 |
|---|---|
| (A) 用 RL 信号梯度直接训 encoder | 不行。RL 信号稀疏、噪声大、样本效率低，而 encoder 又是个 transformer，参数很多。这样训会需要海量交互数据——这正是 RLT 要避免的事 |
| (B) 不训 encoder，随机初始化 | 那它就是个随机投影，$z_{rl}$ 完全不能保证含有任务信息 |
| (C) 监督学习（标签是什么？） | VLA 嵌入 $z_{1:M}$ 没有现成标签 |
| (D) 自监督——让 $z_{rl}$ 能还原 $z_{1:M}$ | ✅ 这就是 RLT 选的方案 |

方案 (D) 就是经典 autoencoder / 信息瓶颈思路：

```
            ┌─────────────────── 训练阶段才用 ─────────────────┐
            │                                                  │
z_{1:M} ──► encoder g_φ ──► z_rl ──► decoder d_φ ──► h_φ ──► ẑ_{1:M}
                              │                                  │
                              │       Loss = ‖ẑ_{1:M} - z_{1:M}‖²
                              │       (停止梯度只让信号回流到 encoder/decoder/MLP)
                              │
                              └─► [z_rl, s_p] ──► actor / critic   （推理才用）
```

**reconstruction loss 是 encoder 的"导师"**。它的逻辑是：

> 如果 $z_{rl}$ 这个向量足够把整串 $z_{1:M}$ 重建回来，那它就保留了 $z_{1:M}$ 里"几乎所有"的信息。

decoder + MLP 就是用来**评判 $z_{rl}$ 是否真的保留了信息**的工具。如果你不让 decoder 把它解开看看，你就没法判断 encoder 是不是只学了一堆毫无意义的常数向量。

## 一个更直观的类比

这其实就是经典 autoencoder：

```python
# 训练阶段
x ──► encoder ──► z ──► decoder ──► x_hat
                  │       ↑
                  │       └── 训练目标: ||x_hat - x||²
                  │
# 推理阶段
x ──► encoder ──► z   ←── 只用 encoder 当 feature extractor，decoder 丢掉
```

你用过的几乎所有"先预训练再用 encoder"的模型都长这样：

- **Vanilla Autoencoder / VAE**：训练时要 decoder，下游任务只要 encoder
- **MAE (Masked Autoencoder)** for vision：训练时 decoder 重建被 mask 掉的 patch，下游分类只用 encoder
- **Word2Vec 的 CBOW/Skip-gram**：训练时要 softmax 头预测周围词，最后只留 embedding 表
- **BERT 的 MLM 头**：预训练完，下游任务砍掉 MLM 头，只用 backbone

> 这是表征学习里一个反复出现的范式：**"创造一个代理任务（pretext task）逼模型学到好的特征，等特征学好了就把代理任务扔掉。"**

RL Token 里的 decoder + MLP 就是这个"代理任务"的执行者。

## 为什么 RLT 偏偏选"自回归重建"作为代理任务？

论文用的是**自回归重建**：

$$L_{ro} = \mathbb{E}\Big[\sum_{i=1}^M \big\|h_\phi d_\phi([z_{rl}, \bar z_{1:i-1}])_i - \bar z_i\big\|^2\Big]$$

这个公式说的是：给 decoder 看 $z_{rl}$（瓶颈）外加已经重建出来的 $\bar z_{1:i-1}$，让它预测下一个 $\bar z_i$。换句话说，decoder 是一个**条件在 $z_{rl}$ 上的"小 GPT"**。

为什么要这么做，而不是用 decoder 一次性输出所有 $z_{1:M}$？

1. **token 序列本来就是序列结构**，自回归解码天然契合；
2. **强迫 $z_{rl}$ 承载全局信息**：因为 decoder 每一步只能依赖 $z_{rl}$ + 之前的部分，要想把整段都准确还原，$z_{rl}$ 就必须包含足以推断后续每一步的高层语义；
3. 和 GPT-style 预训练同一个思路，已经被证明能学出非常好的表征。

## 为什么不直接拿 VLA 已有的 token 当 RL 输入？

你可能会接着问：既然要重建 $z_{1:M}$，那干嘛绕一圈，直接拿 $z_{1:M}$ 当 actor/critic 的输入不就好了？

这正是 RLT 要解决的核心痛点。论文里也强调了：

- $z_{1:M}$ 是**高维、长序列**（一串 token，每个都是几百到几千维），直接喂给 actor/critic 会让小模型样本效率极差；
- VLM 的 token 包含很多和当前任务无关的通用知识（语言、视觉常识等等）；
- 直接用 $z_{1:M}$ 训练 actor/critic 的话，要么 actor/critic 变得很大（违背"轻量级"初衷），要么训不出来。

所以需要一个**任务相关、紧凑的瓶颈**——这就是 $z_{rl}$ 的角色，decoder 只是用来**确保它真的紧凑且任务相关**的训练装置。

## 一句话总结

> decoder + MLP 不是 online RL 的一部分，而是 **online RL 之前那一阶段预训练 encoder 的工具**。它们的存在让 reconstruction loss 有意义；reconstruction loss 反过来强迫 encoder 把 $z_{1:M}$ 压成一个保留任务信息的小瓶颈 $z_{rl}$。训练结束后，"脚手架"（decoder + MLP）拆掉，留下"骨架"（encoder + $z_{rl}$）给后续 RL 用。

这是 autoencoder / MAE / BERT 这一脉表征学习方法的标准做法，不是 RLT 独有的设计。


# Q：decoder重建计算KL散度的时候，为什么不是encoder过后的latent feature和原输入做重建Loss，而是decoder重建之后的feature和原输入做重建loss？
先纠正一个小用词：这一步叫 **reconstruction loss（重建损失，$L_{ro}$）**，不叫 BC loss。BC loss 是 online RL 阶段 actor 那条目标函数里的 $\beta\|a-\tilde a\|^2$ 项，两者完全不是一回事，不要混。

回到你的问题——答案是：

**是 $\hat z_{1:M}$（解码器输出）和 $z_{1:M}$（原始 VLA 特征）做 MSE，不是 $z_{rl}$ 和 $z_{1:M}$。**

直接看论文里的公式：

$$
L_{ro} = \mathbb{E}_{\mathcal{D}}\Big[\sum_{i=1}^{M}\underbrace{\big\| \underbrace{h_\phi\, d_\phi([z_{rl}, \bar z_{1:i-1}])_i}_{=\ \hat z_i\ (\text{重建出来的第}\,i\,\text{个 token})} \;-\; \underbrace{\bar z_i}_{=\ \text{sg}(z_i)\ (\text{原始 VLA 特征})} \big\|^2}_{\text{逐 token 的 MSE}}\Big]
$$

逐项拆开：

| 符号 | 是什么 | 形状 |
|---|---|---|
| $z_{1:M}$ | VLA 主干输出的原始 token 序列 | $M \times d$ |
| $\bar z_i = \text{sg}(z_i)$ | 同上，只是停掉梯度（防止反传进 VLA） | $d$ |
| $z_{rl}$ | encoder 压出来的瓶颈向量 | $d_{rl}$（单个向量） |
| $d_\phi(\cdots)_i$ | decoder 在第 $i$ 个位置的输出 | $d_{dec}$ |
| $h_\phi d_\phi(\cdots)_i = \hat z_i$ | 经线性投影后重建出的第 $i$ 个 VLA token | $d$（要和 $\bar z_i$ 同维度） |

注意几个关键点：

1. **维度上就决定了不可能是 $z_{rl}$ 和 $z_{1:M}$ 做 MSE**：$z_{rl}$ 是一个向量，$z_{1:M}$ 是 $M$ 个向量。维度都对不上。

2. **MSE 是逐 token 求和的**：对 $i = 1, 2, \dots, M$ 每个位置，都让 decoder 重建一个 $\hat z_i$，再和对应的 $\bar z_i$ 算平方误差，最后加起来。

3. **梯度怎么走**：
   - $\bar z_i = \text{sg}(z_i)$ 这个 stop-gradient 保证 loss **不会** 回传到 VLA 主干 $\theta_{vla}$；
   - 但梯度会回传到 $\phi$ 里所有可训练参数：
     - $h_\phi$（投影头）
     - $d_\phi$（decoder）
     - $z_{rl}$ 这条路径上的 $g_\phi$（encoder）和 $e_\phi$（`<rl>` token embedding）

   也就是说，**$z_{rl}$ 是中间变量，梯度透过 decoder 流过它，再继续流回到 encoder**——这才是 encoder 被训出来的方式。

4. **自回归条件**：decoder 在预测第 $i$ 个位置时，输入是 $[z_{rl}, \bar z_{1:i-1}]$，即"瓶颈向量 + 已经知道的前 $i-1$ 个真值 token"（典型的 teacher forcing 训练范式，和 GPT 训练时一样）。

把数据流再画一遍就很清楚了：

```
z_{1:M} ──► encoder g_φ ──► z_rl
                              │
              ┌───────────────┘
              ▼
       [z_rl, z̄_{1:i-1}] ──► decoder d_φ ──► h_φ ──► ẑ_i
                                                      │
                                                      ▼
                                        MSE( ẑ_i ,  z̄_i = sg(z_i) )
                                                      ▲
                                                      │
                                              原始 VLA 特征
```

所以你只要记一句话：

> **重建损失是"解码器重建出来的 token"和"原始 VLA token"逐位置做 MSE；$z_{rl}$ 只是中间瓶颈，梯度穿过它来训 encoder。**