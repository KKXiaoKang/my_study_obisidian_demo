* Diffusion基于RL的微调，使用近端ppo进行拟合策略
* ![[Diffusion Policy Policy Optimization.pdf]]
## backward | forward 伪代码
![[Pasted image 20250707180604.png]]

### 核心构建 - param参数说明
* `denoising_steps = 20`
	* 含义：扩散模型在生成动作块时，反向去噪的总步数（K）
	* 作用：决定了扩散过程的“深度”，即每次从高斯噪声到最终动作序列要经过多少步去噪。步数越多，理论上生成能力越强，但计算量也越大
*  `ft_denoising_steps = 10`
	* 含义：fine-tune（微调）阶段实际用到的去噪步数。
	* 作用：在微调/强化学习阶段，为了加速训练和提升稳定性，通常会用比预训练更少的去噪步数（如10），只优化最后的若干步
* `cond_steps = 1 `
	* 每次生成动作时，输入给扩散模型的观测（observation）历史步数
	* 作用：决定了模型能“看到”多少历史观测信息（如过去1帧、4帧等），用于条件生成动作块
	* 典型值：如1，表示只用当前观测
* `horizon_steps=4`
	* 含义：每次扩散采样生成的动作块长度（Ta）。
	* 作用：即action chunk size。每次扩散模型输出的是一段动作序列（如4步），而不是单步动作。这有助于生成更平滑、连贯的控制信号
	* 典型值：如4
* `act_steps = 4`
	* 含义：每次与环境交互时，实际执行的动作块长度
	* 作用：通常与horizon_steps一致，表示每次step会把动作块里的所有动作都送进环境，环境连续执行这几步
	* 典型值：如4
* `max_episode_steps=1000`
	* 含义：每个episode（回合）的最大步数
	* 作用：用于限制每个episode的长度，防止无限循环。到达这个步数后，环境会自动reset
	* 典型值：如1000
* `multi_step（wrapper/multi_step.py）`
	* 含义：一个环境wrapper，允许每次step传入并执行一个动作块（长度为n_action_steps），并返回聚合后的观测、奖励等
	* 作用：实现“高层step = 多个原子step”的机制，支持action chunking和观测堆叠
	* `n_obs_steps : ${cond_steps}`
		* 含义：每次返回的观测历史长度
		* 作用：决定了每次reset/step后，返回的观测是最近n_obs_steps帧的堆叠（如stacked frames），用于给模型更多上下文
		* 典型值：如1（只用当前观测）
	* `n_action_steps: ${act_steps}`
		* 含义：每次step实际执行的动作块长度
		* 作用：与act_steps一致，决定了每次step会连续执行多少个动作
		* 典型值：如4
---
## loss function 构建
好的，下面我将结合你给的`PPODiffusion.loss`函数源码，从**数学公式**和**原理**两个层面，详细讲解DPPO的损失函数设计。

---
## 1. PPO的基本思想

PPO（Proximal Policy Optimization）是一种常用的策略梯度方法，其核心思想是：  
**在保证策略更新“不过大”的前提下，最大化期望优势（Advantage）加权的概率比。**

PPO的标准损失函数为：
$$
L^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$
其中：
- $$ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} $$ 是新旧策略的概率比
- $A_t$是优势函数
- $\epsilon$ 是clip系数

---

## 2. Diffusion Policy中的PPO损失

在DPPO中，策略不是直接输出单步动作，而是通过扩散模型生成**动作块**（action chunk），并且每个chunk是通过多步去噪（denoising）得到的。

### 2.1. log概率的处理

- 对于每个batch、每个去噪步、每个动作chunk，都会计算新旧策略的log概率（对高斯分布的对数似然）。
- 代码中：
  ```python
  newlogprobs = newlogprobs.mean(dim=(-1, -2)).view(-1)
  oldlogprobs = oldlogprobs.mean(dim=(-1, -2)).view(-1)
  ```
  这表示**对所有去噪步和动作chunk的log概率做均值**，即每个样本的log概率是所有去噪步和chunk的平均。

### 2.2. 概率比与优势

- 概率比：
  $$
  r = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})
  $$
- 优势经过归一化和分位数裁剪，并且乘以一个**去噪折扣因子**（credit assignment，见下文）。

### 2.3. PPO的clip损失

- 计算两个版本的policy loss：
  $$
  L_1 = -A \cdot r
  $$
  $$
  L_2 = -A \cdot \text{clip}(r, 1-\epsilon, 1+\epsilon)
  $$
- 取两者的最大值（即PPO的“clipped surrogate objective”）：
  $$
  L^{\text{PPO}} = \mathbb{E}[\max(L_1, L_2)]
  $$

### 2.4. Value loss

- 价值网络的损失是MSE（可选clip）：
  $$
  L^{\text{value}} = \frac{1}{2} \mathbb{E}[(V(s) - R)^2]
  $$
  或clip版本。

### 2.5. 熵损失

- 熵损失用于鼓励策略多样性：
  $$
  L^{\text{entropy}} = -\mathbb{E}[\text{Entropy}(\pi_\theta)]
  $$

### 2.6. BC损失（可选）

- 如果启用行为克隆（BC）正则项，则加上：
  $$
  L^{\text{BC}} = -\mathbb{E}[\log \pi_\theta(a_{\text{teacher}}|s)]
  $$

---

## 3. 去噪折扣因子（Denoising Discount）

- 在diffusion policy中，每个动作块是通过多步去噪生成的。为了合理分配优势（credit assignment），对每个去噪步的优势乘以一个折扣因子：
  $$
  \text{discount}_i = \gamma_{\text{denoising}}^{K - i - 1}
  $$
  其中$K$是总去噪步数，$i$是当前去噪步。
- 这样，越靠近最终动作的去噪步，优势权重越大，越早的去噪步权重越小。

---

## 4. 总体损失函数

综合上述，DPPO的总损失为：
$$
L = L^{\text{PPO}} + \beta_{\text{entropy}} L^{\text{entropy}} + \beta_{\text{value}} L^{\text{value}} + \beta_{\text{BC}} L^{\text{BC}}
$$
其中各项权重可调。

---

## 5. 直观理解

- **PPO部分**：鼓励新策略在不偏离旧策略太多的前提下，提升优势高的动作概率。
- **Diffusion部分**：每个动作块的所有动作、所有去噪步都被平均考虑，credit assignment通过去噪折扣实现。
- **Value/熵/BC部分**：分别用于价值估计、探索和行为克隆正则。

---

## 6. 代码与公式一一对应

- `logratio = newlogprobs - oldlogprobs` → \(\log \pi_\theta - \log \pi_{\theta_{\text{old}}}\)
- `ratio = logratio.exp()` → \(r\)
- `pg_loss1 = -advantages * ratio`，`pg_loss2 = -advantages * torch.clamp(ratio, ...)`，`pg_loss = torch.max(pg_loss1, pg_loss2).mean()` → PPO clipped loss
- `v_loss = ...` → value loss
- `entropy_loss = -eta.mean()` → 熵损失
- `advantages *= discount` → 去噪折扣

---

## 总结

**DPPO的损失函数本质上是PPO的clipped surrogate loss在diffusion policy（动作块+多步去噪）上的扩展，结合了优势归一化、credit assignment、熵正则和可选的行为克隆正则。每个动作块的所有动作和所有去噪步都被平均考虑，credit assignment通过去噪折扣实现。**

如需更具体的公式推导或某一部分的详细解释，欢迎继续追问！