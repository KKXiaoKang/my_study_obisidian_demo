# SBL3 里的 TD 算法定位 + Model-free vs Model-based 完整对照

我把你赵世钰老师讲的概念框架（[一R]-[四R]）和 SBL3 代码一一对应起来。

---

## 一、回顾：TD 算法是怎么推出来的（赵老师的视角）

从 Bellman 期望方程出发：

$$v_\pi(s) = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]$$

TD 算法的核心思想就是：**用一次采样的"瞬时样本" $r + \gamma v(s')$ 去近似 Bellman 方程右边的期望，然后把当前估计 $v(s)$ 朝这个采样目标"轻轻挪一点点"**：

$$v(s) \leftarrow v(s) + \alpha\underbrace{[r + \gamma v(s') - v(s)]}_{\text{TD error } \delta}$$

这就是**TD(0)**——一步引导（one-step bootstrap）。这是所有 TD 算法的母版。

| TD 算法变体 | 目标 (target) | 是否需要下一步动作 $a'$？ |
|---|---|---|
| **TD(0)** (state value) | $r + \gamma v(s')$ | 否（只学 $V$） |
| **SARSA** | $r + \gamma Q(s', a')$，$a'$ 由当前策略 $\pi$ 采样 | 是，**on-policy** |
| **Q-learning** | $r + \gamma \max_{a'} Q(s', a')$ | 否（用 max），**off-policy** |
| **Expected SARSA** | $r + \gamma \mathbb{E}_{a'\sim\pi}[Q(s', a')]$ | 期望，可 on/off |
| **n-step TD** | $r_t + \gamma r_{t+1} + \dots + \gamma^n v(s_{t+n})$ | n 步引导 |
| **TD(λ)** | $\sum_n (1-\lambda)\lambda^{n-1} G^{(n)}_t$ | n-step 的几何加权平均 |
| **MC** | $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$ | 整条回合的回报 |

直观上：**MC 用真值（高方差、无偏）；TD(0) 用 bootstrap（低方差、有偏）；TD(λ) 在两者之间插值**。

---

## 二、SAC 用的是什么 TD？—— **Soft Q-learning（off-policy TD(0)）**

打开 `sac.py` 第 246–255 行：

```246:255:stable_baselines3/sac/sac.py
with th.no_grad():
    # Select action according to policy
    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
    # Compute the next Q values: min over all critics targets
    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
    # add entropy term
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    # td error + entropy term
    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
```

这正是一个 **Soft TD(0) target**：

$$y_{\text{SAC}} = r + \gamma \Big(\underbrace{\min_{i=1,2}Q_{\bar\theta_i}(s', a')}_{\text{Clipped Double Q (off-policy)}} - \alpha\log\pi(a'|s')\Big)$$

它属于哪个家族？我把它定位一下：

- **是 TD(0)**（只用一步引导，没有 n-step 或 GAE）
- **是 off-policy**：transition $(s,a,r,s')$ 来自 replay buffer（旧策略产生），但 target 用当前策略 $\pi$ 重新采样 $a'$ → behavior policy ≠ target policy
- **不是纯 Q-learning**：Q-learning 用 $\max_{a'} Q(s',a')$，SAC 用 $\mathbb{E}_{a'\sim\pi}[Q(s',a') - \alpha\log\pi(a'|s')]$ → 这是**Soft Expected SARSA**（带熵正则的 Expected SARSA）的连续版本！

> **结论**：SAC 在赵老师的分类里，最接近**Expected SARSA（off-policy 版）+ 熵正则**。它不是 Q-learning，因为没用 max；它是 off-policy 的，因为数据来自 replay buffer。

代码里另两个跟 off-policy TD 配套的细节：
- **target network**（`critic_target`）：缓解 bootstrap 自指带来的发散；
- **Clipped Double Q**（两个 critic 取 min）：缓解 max bias / overestimation。

---

## 三、PPO 用的是什么 TD？—— **GAE = TD(λ)**

PPO 自己的 `train()` 看不到 TD 计算，因为 returns 是在 rollout buffer 里**预计算**好的。打开 `buffers.py` 第 403–438 行：

```403:438:stable_baselines3/common/buffers.py
def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
    """
    Post-processing step: compute the lambda-return (TD(lambda) estimate)
    and GAE(lambda) advantage.

    Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
    ...
    The TD(lambda) estimator has also two special cases:
    - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
    - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))
    """
    ...
    delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
    last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
    self.advantages[step] = last_gae_lam
    # TD(lambda) estimator
    self.returns = self.advantages + self.values
```

这段就是 Schulman 2015 的 **GAE** 论文公式，等价于 **TD(λ) 估计器**：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad\text{(TD error)}$$
$$\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$
$$\hat R_t = \hat A_t + V(s_t) \quad\text{(TD(λ) return)}$$

PPO 默认 `gae_lambda = 0.95`，所以是介于 TD(0) (λ=0) 和 MC (λ=1) 之间的折衷。然后 critic loss 用这个 $\hat R$ 做监督：

```243:244:stable_baselines3/ppo/ppo.py
# Value loss using the TD(gae_lambda) target
value_loss = F.mse_loss(rollout_data.returns, values_pred)
```

定位：
- **是 TD(λ)**（n-step 的几何加权混合，λ=0.95 时大约等价于 ~20 step TD）
- **是 on-policy**：rollout buffer 一次性收集当前策略数据，训练完就丢弃 → behavior policy = target policy
- 策略侧不直接是某种 TD 控制算法，而是用 PG 配 clipped surrogate；critic 侧学 V(s) 用 TD(λ) target

> **结论**：PPO 的 critic 是 **on-policy TD(λ)（用 GAE 实现）**，policy 用策略梯度。整体可以看作"policy gradient + actor-critic + TD(λ) baseline"。

---

## 四、SAC vs PPO 在赵老师框架下的总对照表

| 维度 | SAC | PPO |
|---|---|---|
| TD 算法 | TD(0) | TD(λ)（λ=0.95，GAE 实现）|
| Value 函数 | Action-value $Q(s,a)$ | State-value $V(s)$ |
| 控制算法类型 | Soft Expected SARSA | Policy Gradient + Actor-Critic |
| Bootstrap 步数 | 1 步 | $\approx 20$ 步（几何加权）|
| Bias / Variance | 高 bias 低 variance | 中间值（λ 调控）|
| 数据使用 | Replay buffer（**off-policy**） | Rollout buffer 用完即扔（**on-policy**） |
| Target policy vs Behavior policy | 不同（off-policy） | 相同（on-policy）|
| 探索机制 | 熵正则 + 随机策略 | 熵小奖励（默认关）+ 随机策略 + clip |
| 防发散手段 | target net、double Q、clipped Q | clipped surrogate、trust region |
| 适合任务 | 连续控制、采样昂贵的任务 | 通用、并行环境多的任务 |

---

## 五、什么是 Model-free / Model-based？你的理解对吗？

### 5.1 定义

**Model-free RL**：智能体**不显式**构建/使用环境的状态转移模型 $P(s'|s,a)$ 或奖励模型 $R(s,a)$。它直接从经验 (s, a, r, s') 学习 value function 或 policy。

**Model-based RL**：智能体**显式**学习或使用一个环境模型，然后用这个模型来：
- 做"想象中的 rollout"，产生合成数据用于训练 policy/value（**Dyna 风格**）；
- 做规划（**MPC / 树搜索风格**），在每个决策点用模型 simulate 多条未来轨迹然后选最好的；
- 通过模型反向传播梯度（**Backprop through dynamics**），直接对策略求梯度。

### 5.2 你的理解修正

> "model-free RL 是不知道 Q 值状况下，和不知道状态转移方程下的 RL 训练方法"

**有一半正确，有一半要修正**：

- ✗ 不是"不知道 Q 值"——model-free 是**学** Q 值，不是"不知道"。SAC/PPO 都在显式拟合 Q 或 V。
- ✓ 是"不知道 / 不学状态转移方程 $P(s'|s,a)$"——这个是准确的。

更准确的说法：**Model-free = 不学/不用环境动态模型，直接从交互数据中学习决策**；**Model-based = 学/用环境动态模型来辅助决策**。

> "model-based RL 是不是很像把上面的 state space model 状态空间方程的表示进行描述建立的 RL 方法？"

✓ **这个直觉非常对**！你说的"state space model"在控制论里就是：
$$x_{t+1} = f(x_t, u_t) + w_t, \quad y_t = h(x_t) + v_t$$

Model-based RL 做的事在结构上完全平行：
$$s_{t+1} = \hat f_\phi(s_t, a_t) + \epsilon, \quad r_t = \hat r_\phi(s_t, a_t)$$

只是 model-based RL 把这个 $\hat f_\phi$ 用神经网络 / 高斯过程 / 集成模型来拟合，而经典控制用线性化、kalman、子空间辨识等。**所以现代 model-based RL 经常被描述为"learned-dynamics control"，本质上就是"先做系统辨识，再做最优控制"的 RL 化版本**。

---

## 六、Model-based RL 的具体方法谱系

我按 "怎么使用 model" 分四类，列出有代表性的方法：

### 类别 A：Dyna 风格（用模型生成合成数据）
- **Dyna-Q** (Sutton, 1991)：经典开山之作，用表格模型生成想象 transition，混合真数据训练 Q-learning
- **MBPO** (Janner et al., 2019)：训练 NN ensemble 作为模型，用模型短滚动 (k=1-5 步) 生成数据喂给 SAC。**这是把 model-based 嫁接到 SAC 的代表作**
- **AlgoVer / SLBO**：理论分析较强的 model-based RL

### 类别 B：规划/MPC 风格（用模型在决策时 simulate）
- **PILCO** (Deisenroth, 2011)：用高斯过程学动态模型，解析地计算策略梯度。**经典 sample-efficient 方法**
- **PETS** (Chua et al., 2018)：Probabilistic Ensembles + Trajectory Sampling，用集成模型做 CEM 优化的 MPC
- **iLQR / DDP + learned model**：把学到的 dynamics 接入 iterative LQR

### 类别 C：World Model 风格（在隐空间学动态 + 训 policy）
- **World Models** (Ha & Schmidhuber, 2018)：VAE 编码状态 + RNN 学 latent dynamics + CMA-ES 训 policy。开山之作
- **PlaNet** (Hafner et al., 2019)：RSSM 隐空间模型 + MPC 规划
- **Dreamer v1/v2/v3** (Hafner)：在 RSSM 隐空间用 actor-critic 训练 policy。**Dreamer v3 是 SOTA 之一**
- **TD-MPC / TD-MPC2** (Hansen et al.)：把 model 和 value function 联合训，决策时做小范围 trajectory optimization

### 类别 D：隐式模型 / 价值等价模型
- **MuZero** (Schrittwieser et al., 2020)：不学真实 $s'$，只学一个"价值等价"的隐空间模型，配合 MCTS。AlphaGo/AlphaZero 的下一代
- **EfficientZero**：MuZero 的 sample-efficient 加强版

### 简单对比表

| 方法 | 模型类型 | 怎么用模型 | sample 效率 | 适用场景 |
|---|---|---|---|---|
| Dyna-Q | 表格 | 合成数据 | 高 | 离散状态 |
| MBPO | NN ensemble | k 步合成数据→SAC | 高 | 连续控制 (MuJoCo) |
| PILCO | GP | 解析策略梯度 | 极高 | 低维任务 |
| PETS | NN ensemble | CEM-MPC | 高 | 中等维 robotics |
| Dreamer v3 | RSSM 隐空间 | 隐空间 actor-critic | 高 | 像素输入、复杂任务 |
| TD-MPC2 | Latent dynamics + value | 短规划 + value bootstrap | 极高 | 大规模连续控制 |
| MuZero | 价值等价隐模型 | MCTS | 中 | 离散动作、博弈类 |

---

## 七、回到你的实验：SAC 在 shaped reward 下崩，model-based 会更好吗？

值得一提的是：你做的"输出 target=0"是个**stateless**任务（环境状态与 action 无关）。在这种任务里 **model-based 反而没什么优势**——因为模型 $\hat f_\phi(s,a) = s$ 是恒等映射，没什么可学。

但在你最终目标 **VR 教学 Kuavo 搬箱子** 这种"动力学复杂、采样昂贵"的任务里，model-based RL 是非常值得考虑的方向：
- **MBPO 风格**：在 SAC 基础上加一个动力学模型，用模型短滚动扩充 replay buffer。可以用更少的真实交互数据；
- **Dreamer v3 / TD-MPC2 风格**：从像素 + 本体感知学一个 world model，在 imagination 里训 policy。这是目前 Hopkins、UCSD 一系做 humanoid 双足 / manipulation 的主流方向；
- **Residual Policy + MPC**：用 model-based MPC 做底层抓取，policy 学补偿/残差。这种方法在 manipulation 上结果通常更稳。

---

## 八、一句话总结串起整个回答

> SBL3 里的 **SAC = off-policy TD(0) 的 Soft Expected SARSA（带熵正则 + Clipped Double Q）**；**PPO = on-policy 策略梯度 + critic 用 TD(λ)（GAE）做监督**。两者都属于 **model-free**，区别只在 TD 步数和数据复用方式。
>
> **Model-based RL** 是把环境动力学 $P(s'|s,a)$ 显式学出来（你说的"state space model 化"完全对），然后通过 **生成合成数据 / 在线规划 / world model 隐空间训练 policy / 价值等价隐模型 + MCTS** 四种主要范式来加速学习。其中 **MBPO（model + SAC）** 和 **Dreamer v3**、**TD-MPC2** 是目前 robotics manipulation / 仿真控制场景里最值得关注的几条路线。

参考文献：
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) Ch. 6 (TD) Ch. 8 (model-based)
- Schulman et al., *High-dimensional continuous control using GAE* (arXiv [1506.02438](https://arxiv.org/abs/1506.02438))
- Janner et al., *MBPO* (NeurIPS 2019, arXiv [1906.08253](https://arxiv.org/abs/1906.08253))
- Hafner et al., *Dreamer v3* (arXiv [2301.04104](https://arxiv.org/abs/2301.04104))
- Hansen et al., *TD-MPC2* (arXiv [2310.16828](https://arxiv.org/abs/2310.16828))
- 赵世钰，《强化学习的数学原理》Ch. 7（TD 算法）