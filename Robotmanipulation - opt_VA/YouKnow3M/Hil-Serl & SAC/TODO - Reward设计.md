我已经收集到了足够的理论依据，现在给你一个系统、明确的分析。

# 为什么 SAC 在多维度 shaped reward 下无法收敛 —— 完整分析

## 一、核心问题：你已经触碰到了 SAC 的"熵奖励通胀（Reward Inflation）"陷阱

### 1.1 你的两种 reward 在数值层面的根本差异

| Reward 类型 | 公式 | 取值范围 | 在 14 维 random action 下数量级 |
|---|---|---|---|
| **Shaped reward** | $r = \frac{1}{1+\\|a-a^*\\|}$ | $(0, 1]$ | 约 0.2–0.4（每步都是**正值**） |
| **MSE reward** | $r = -\frac{1}{D}\sum (a_i - a^*_i)^2$ | $(-\infty, 0]$ | 约 -0.3 ～ -1（**负值，随机时magnitude大**） |

这两种 reward 看起来都"指向同一个最优 $a^* $"，但对 SAC 而言**完全不等价**，原因要回到下面这条公式。

### 1.2 SAC 的 Critic 目标里"嵌入"了熵奖励——这是关键

看 `stable_baselines3/sac/sac.py` 第 253–255 行：

```253:255:stable_baselines3/sac/sac.py
next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
# td error + entropy term
target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
```

也就是说 SAC 的 Q 学习目标是：

$$Q(s,a)\;\leftarrow\; r + \gamma \Big( Q'(s',a') \;\boxed{-\alpha\log\pi(a'|s')}\Big)$$

这就把"熵奖励" $-\alpha \log\pi$ **递归地累加到 Q 值里**。这是 SAC 论文（Haarnoja 2018）的核心，但也是它在 shaped reward 下崩掉的根因。

而 PPO 的 critic 学的是纯粹的 task return：

```243:244:stable_baselines3/ppo/ppo.py
# Value loss using the TD(gae_lambda) target
value_loss = F.mse_loss(rollout_data.returns, values_pred)
```

`returns` 由 GAE 从**原始 reward** 算出来，**不含熵项**。熵只是在 actor loss 里作为一个小正则（默认 `ent_coef=0.0`）：

```247:256:stable_baselines3/ppo/ppo.py
if entropy is None:
    entropy_loss = -th.mean(-log_prob)
else:
    entropy_loss = -th.mean(entropy)

entropy_losses.append(entropy_loss.item())

loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
```

---

## 二、为什么 shaped reward（正奖励）会让 SAC 卡在随机策略上

### 2.1 数量级对比：熵 vs 任务奖励

SAC 的 actor loss 是：

$$L_\pi = \mathbb{E}\big[\alpha \log\pi(a|s) - Q(s,a)\big]$$

策略要让 $Q(s,a)-\alpha\log\pi$ 尽可能大。对一个 14 维 Squashed Gaussian（tanh 后），高斯本身的熵约 $0.5\cdot D\cdot \log(2\pi e) \approx 1.42D$，再减去 tanh 修正后量级仍然 $\sim D$。

在 SBL3 的 `_setup_model` 第 168 行：

```168:168:stable_baselines3/sac/sac.py
self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
```

`target_entropy = -dim(A) = -14`。这是**自动调温**的 target，意味着 $\alpha$ 会被调整到使 $\log\pi$ 大约稳定在 -14 附近。

也就是说，每一步的"熵奖励" $-\alpha\log\pi(a|s) \approx 14\alpha$。

来比较一下两种 reward 下 Q 值大致的稳态量级（用 $\gamma=0.99$，无穷视界）：

| 维度 D | reward 类型 | 每步 task r | 每步熵奖励 (α≈1 初期) | 稳态 Q (任务部分) | 稳态 Q (熵部分) |
|---|---|---|---|---|---|
| 2 | shaped | ~0.5 | ~2 | ~50 | ~200 |
| 6 | shaped | ~0.3 | ~6 | ~30 | ~600 |
| 14 | shaped | ~0.2 | ~14 | ~20 | ~1400 |
| 14 | MSE | -0.5 | ~14 | -50 | ~1400 |

注意 shaped reward 下：
- **任务部分 Q 值始终是有界小正数** (上界 $\frac{1}{1-\gamma}\approx 100$)
- **熵部分 Q 值随维度线性扩张**

当维度从 2 升到 14：
- task signal 的"梯度信号"被维度稀释（每个维度贡献 $\partial r/\partial a_i \to 0$）；
- 而熵奖励反而随 $D$ **线性增加**；
- 于是 actor 优化 $Q - \alpha\log\pi$ 时，**优化压力几乎全在熵那一项**，结果是 policy 维持高熵 = 高斯方差大 = 接近均匀随机 → 看起来 reward 卡死在某个低位震荡。

这就是为什么你看到：
- D=2 时 shaped reward 能收敛到 ≈200；
- D=6 时变差；
- D=14 时彻底崩盘。

### 2.2 进一步：Reward Inflation（熵奖励通胀效应）

ICML 2022 的论文 *"Do You Need the Entropy Reward (in Practice)?"*（arXiv 2201.12434）直接命名了这个现象——**reward inflation**。我引用一段它对 episodic MDP 的关键描述：

> "Entropy bonuses are added to normal time steps but not to those post termination. This could result in overly optimistic or pessimistic policies, depending on whether the entropy is positive or negative... When $\alpha$ is tunable and starts with 0.2, $\alpha H(\pi)$ can still be greater than the task reward... the agent's policy becomes **overly optimistic and never wants to reach the terminal state**."

翻译过来对应到你的实验：
1. 每个 episode 200 步、固定时间截断；
2. shaped reward 上界是 1，每步的熵奖励 $14\alpha$ 远大于 1；
3. SAC 实际在最大化的不是"接近 target"，而是"**保持高熵尽量久**"，task reward 完全被熵掩盖；
4. 这就是论文里说的 **policy overly optimistic, refuses to commit to the task**。

而你切到 MSE reward 后：
- reward 是**负数**且数量级随维度变大（14 维时随机动作每步 ~ -0.5 到 -1）；
- task Q 值是**强负的**（量级几十到几百），**支配了**熵那一项；
- $\alpha$ 在自动调温下迅速下调（因为只要稍微动一下 $a$ 朝 target 走，任务奖励的提升远大于损失的那点熵）；
- 于是 actor 被迫去最小化 MSE，正常收敛。

---

## 三、为什么 PPO 完全不在乎这个

PPO 是 on-policy + advantage normalization 的算法，有四道防火墙让它免疫这种问题：

**防火墙 1：Critic 不含熵项**  
见上文。PPO 的 $V(s)$ 学的是纯任务回报，没有 reward inflation。

**防火墙 2：Advantage 标准化**

```218:219:stable_baselines3/ppo/ppo.py
if self.normalize_advantage and len(advantages) > 1:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

每个 batch 把 advantage 归一化到零均值单位方差，**reward 加任意常数 / 任意正比例缩放都不会影响策略梯度**。所以 PPO 对 reward shift 和 reward scale 都是不敏感的（你也观察到了 scale 1 vs 1000 PPO 只是慢一点）。

**防火墙 3：固定的小熵系数 + 不通过 bootstrap 传播**  
默认 `ent_coef=0.0`，即使设了 0.01，也只是在 actor loss 里加一个微小正则，绝不会被 critic 累计放大到几百几千的量级。

**防火墙 4：clipped surrogate + on-policy**  
PPO 每个 batch 只能在 trust region 内移动，policy 不会突然发散；且数据来自当前 policy，不存在 off-policy 下 critic 把错误高估传染给 actor 的问题。

---

## 四、SAC 在 shaped reward 下到底"是不是熵的锅"？—— 是的，但不止熵

是的，**根本原因就是熵奖励的相对数量级**。可以拆成三个层面：

1. **熵 vs 任务奖励的相对量级失衡**（最主要）  
   shaped reward 被 $\frac{1}{1+\|\cdot\|}$ 上界限定为 1，而熵奖励量级随 action_dim 线性增长。维度越高越被压垮。

2. **`target_entropy = -dim(A)` 的自动调温不能自救**  
   自动调温保证的是 $\log\pi \approx -\dim(A)$，**它根本不"知道"任务奖励的数量级**。它只是按维度盲目地给一个熵 target。结果是：任务奖励小、维度高时，$\alpha$ 会自动调到一个能维持目标熵的水平，进一步把策略锁在高熵状态。

3. **Off-policy + Q bootstrap 把误差放大**  
   shaped reward 上界为 1，random policy 下 Q≈100（来自 $1/(1-\gamma)$），但熵部分能到 1400，**Q 网络要拟合主要是熵那部分**。这导致：
   - critic loss 看似收敛了（但收敛到了一个"熵驱动"的值函数）；
   - actor 跟着这个被熵污染的 Q 优化，离 task 越来越远。
   
   切到 MSE reward 后，任务奖励的数量级反过来超过熵，整个动力学就反转了。

---

## 五、回到你的工程问题：Kuavo VR 教学抓箱子

你的"抓但抬不起"是同一个病：
- VR 人工干预的 demonstration 在 buffer 里没问题；
- 但 reward shaping 如果用的是**正向小奖励**（比如"距离倒数"），在多自由度（双臂 + 手指 ≈ 十几到几十维）下，**熵奖励一定会把任务奖励淹没**，agent 只会在"看起来比平均稍微好一点点"的地方乱晃；
- 抓住箱子那一刻可能确实拿到了一个小正 reward，但比起持续高熵带来的"积分熵奖励"，没诱因继续往上抬。

**几个明确可执行的修复建议**：

1. **Reward 的设计原则**：让任务信号成为"惩罚 + 稀疏正奖励"的组合，而不是稠密的小正奖励。例如：
   - 主体用 negative shaping：$r = -\|a-a^*\|$ 或 $-\|\text{pos}_{\text{box}} - \text{pos}_{\text{target}}\|$；
   - 关键事件给稀疏大正奖励：抓起来 +10、抬到一定高度 +50、成功放下 +100。
2. **如果坚持用正向 shaped reward**，做以下任一处理：
   - 用 `SACLite` / `SACZero` 思路：把熵从 critic bootstrap 里去掉（只在 actor 里作为正则），或对熵奖励做 zero-mean normalization；
   - 把 `ent_coef` 改成手动小常数（如 0.01）+ 关闭 auto temperature；
   - 把 `target_entropy` 设小一些（更负），如 `-2*dim(A)`，强行压低 α。
3. **直接用 reward scale**：把 reward 乘个大的负偏置 `r = r - 1` 转成惩罚形态——你已经发现这招对 SBL3 SAC 很有效，理论上一致。
4. **加 gradient clipping**（你提到 RLPD-Lerobot 有，SBL3 SAC 默认没有），可以缓解 critic 爆炸但**治标不治本**。
5. **如果想要快速复现稳定**：先用 PPO 验证 reward 设计是否能学到正确行为（PPO 不会因为 reward 形态而崩），再切到 SAC 调超参，这样能把"reward 设计问题"和"算法超参问题"解耦排查。

---

## 六、一句话总结

> **SAC 失败的根因不是它学不动多维 action，而是它的 critic 把熵奖励内化进了 Q 值。当任务 reward 的数量级 ≤ 熵奖励的数量级（典型场景：正值有界 reward + 高维动作空间）时，policy 优化目标实际被熵主导，agent 选择"保持随机"而非"完成任务"。MSE reward 因为是负值且随维度增大 magnitude 变大，把任务信号重新顶到熵之上，所以能收敛。**
> 
> **PPO 不受这种现象影响：它的 critic 只学纯任务回报，且 advantage 做了 z-score 归一化，因此对 reward 形态、scale、shift 都鲁棒。**

参考资料：
- Yu et al., *Do You Need the Entropy Reward (in Practice)?* (arXiv [2201.12434](https://arxiv.org/abs/2201.12434))——"reward inflation" 概念的来源
- Haarnoja et al., *Soft Actor-Critic Algorithms and Applications* (arXiv [1812.05905](https://arxiv.org/abs/1812.05905))——自动调温与 `target_entropy = -|A|` 的来源
- Ahmed et al., *Understanding the Impact of Entropy on Policy Optimization* (ICML 2019)——熵与策略优化收敛性的系统研究