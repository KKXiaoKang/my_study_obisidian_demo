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

直观上：**MC 用真值（高方差、无偏）；TD(0) 用 bootstrap（低方差、有偏）；TD(λ) 在两者之间插值**
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

# Q : 详细的model-base RL 方程
# Model-based RL 算法详解：公式 + 架构 + 伪代码

我从最底层的"通用框架"开始，再用 5 个代表性算法把"如何使用 model"的不同范式一个一个讲清楚。每个算法都给你**公式 + 架构图 + 伪代码**。

---

## 零、通用框架：Model-based RL 学什么、怎么用

### 0.1 什么是"模型"

Model-based RL 就是要学一个**环境的近似器**，最一般的形式包含两部分：

$$\underbrace{\hat T_\phi(s_{t+1} \mid s_t, a_t)}_{\text{状态转移模型 / dynamics}} \quad\text{和}\quad \underbrace{\hat R_\phi(r_t \mid s_t, a_t)}_{\text{奖励模型}}$$

如果是**确定性模型**：
$$s_{t+1} = f_\phi(s_t, a_t), \quad r_t = R_\phi(s_t, a_t)$$

如果是**概率模型**（带不确定性）：
$$s_{t+1} \sim \mathcal{N}(\mu_\phi(s_t,a_t), \Sigma_\phi(s_t,a_t))$$

注意这跟你脑子里"state space model"完全一致：把控制理论里的 $x_{t+1}=f(x_t,u_t)+w_t$ 换成神经网络/高斯过程拟合，就是 RL 的 dynamics model。

### 0.2 模型的训练目标

收集真实 transition $(s_t, a_t, r_t, s_{t+1})$，做监督学习：

$$\mathcal{L}_{\text{model}}(\phi) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\Big[\underbrace{\|f_\phi(s,a) - s'\|^2}_{\text{下一步预测}} + \underbrace{\|R_\phi(s,a) - r\|^2}_{\text{奖励预测}}\Big]$$

或者最大化对数似然（概率模型）：

$$\mathcal{L}_{\text{model}}(\phi) = -\mathbb{E}_{\mathcal{D}}\big[\log p_\phi(s',r \mid s, a)\big]$$

### 0.3 模型可以怎么用——四大范式

|范式|做法|代表算法|
|---|---|---|
|**A. Dyna 风格**|用模型 simulate 出合成 transition → 喂给 model-free 算法|Dyna-Q, **MBPO**|
|**B. MPC 规划风格**|每个决策时刻，用模型 simulate 多条 trajectory，选最好的|**PETS**, MPC-NET|
|**C. 隐空间 world model**|在 latent space 学 dynamics + 用 actor-critic 训 policy|**Dreamer**, PlaNet|
|**D. 价值等价 / 短模型 + value**|短窗口用 model 展开，长视界用 value function 兜底|**TD-MPC**, MuZero|

下面 5 个算法分别覆盖这四种范式。

---

## 一、Dyna-Q（Sutton, 1991）—— 最小可理解的 model-based 范式

### 1.1 架构

```
                ┌────────────────────────────┐
                │   Real environment         │
                └─────────────┬──────────────┘
                              │  (s,a,r,s')
                              ▼
                ┌──────────────────────┐    ┌─────────────────┐
                │   Q-table  Q(s,a)    │ ◄─ │ Model 表 T̂(s,a) │
                │  (用 Q-learning 更新) │    │  = (s', r)       │
                └──────────┬───────────┘    └─────────┬───────┘
                           ▲                          │
                           │   k 次 imagined update    │
                           └──────────────────────────┘
```

### 1.2 公式（表格场景）

**模型更新**：
$$\hat T(s,a) \leftarrow (s', r) \quad\text{（直接记忆，因为是确定性环境）}$$

**Q-learning 更新**（用真实 OR imagined 数据都用同一个公式）：
$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big]$$

### 1.3 伪代码

```
初始化 Q(s,a) ← 0，  Model(s,a) ← ∅

repeat（每个真实 step）：
    1. 从当前 s 用 ε-greedy 选 a
    2. 与真环境交互得到 (r, s')
    3. Q-learning 更新（真实数据）
       Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    4. 模型更新
       Model(s,a) ← (r, s')
    5. for k 次:                            ← 这是 Dyna 的精髓
         随机采样过去访问过的 (s̃, ã)
         (r̃, s̃') = Model(s̃, ã)
         Q(s̃, ã) ← Q(s̃, ã) + α[r̃ + γ max Q(s̃', a') - Q(s̃, ã)]
```

**核心思想**：**每次真实交互后，用模型在脑子里"再玩 k 次"**，把 1 个真实样本变成 (k+1) 个学习样本。

---

## 二、PETS（Chua et al., 2018）—— NN Ensemble + MPC 规划

### 2.1 架构

```
                          ┌────────────────────────┐
                          │     Real environment   │
                          └────────────┬───────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────┐
                  │       Replay Buffer D               │
                  └────────────┬────────────────────────┘
                               │
                               ▼ (监督训练)
       ┌──────────────────────────────────────────────────────┐
       │   NN Ensemble:  {f_φ^1, f_φ^2, ..., f_φ^B}            │
       │   每个 f_φ^i(s,a) → (μ^i, Σ^i)，输出高斯分布           │
       │   B 个模型捕捉 epistemic uncertainty                   │
       └──────────────────────────────────────────────────────┘
                               │
                               │ 决策时刻 (推理)
                               ▼
                  ┌─────────────────────────────────┐
                  │       CEM Planner (MPC)          │
                  │  在 action sequence 空间搜索      │
                  └────────────┬────────────────────┘
                               │
                               ▼ a_t = a_seq[0]
                          回到真环境
```

**没有 policy 网络！** 决策完全靠 MPC。

### 2.2 公式

**每个 ensemble 成员输出概率分布**：
$$f_\phi^i(s_t, a_t) = \mathcal{N}\big(\mu_\phi^i(s,a),\ \Sigma_\phi^i(s,a)\big)$$

**训练 loss**（高斯负对数似然）：
$$\mathcal{L}^i = \sum_{(s,a,s')}\Big[(s' - \mu^i(s,a))^\top \Sigma^{i,-1}(s,a)(s' - \mu^i(s,a)) + \log\det\Sigma^i(s,a)\Big]$$

**决策时的 trajectory sampling**（每条 imaginary trajectory 在每步都**随机抽一个 ensemble 成员**，从而把不确定性传播进去）：

$$s_{t+h+1}^{(p)} \sim f_\phi^{i_h}(s_{t+h}^{(p)}, a_{t+h})\quad i_h \sim \text{Uniform}(\{1,\ldots,B\})$$

**CEM 目标**：选 action sequence $a_{t:t+H}$ 使期望累计奖励最大：

$$a_{t:t+H}^* = \arg\max_{a_{t:t+H}} \mathbb{E}_{p}\Big[\sum_{h=0}^{H-1} R_\phi(s_{t+h}^{(p)}, a_{t+h})\Big]$$

### 2.3 伪代码

```
初始化 ensemble {f_φ^1, ..., f_φ^B}，buffer D ← ∅

repeat（每个真实 episode）：
    s ← env.reset()
    for t = 1...T:
        # ===== 决策 (MPC + CEM) =====
        a_t = PLAN(s_t)               
        s_{t+1}, r_t = env.step(a_t)
        D ← D ∪ {(s_t, a_t, r_t, s_{t+1})}
    
    # ===== 训练 model =====
    for i = 1...B:
        从 D 采 bootstrap 子集 D_i
        梯度下降最小化 L^i 在 D_i 上


function PLAN(s_t):                    # CEM 迭代 N 轮
    μ, σ ← 初始高斯（H 步 action 序列）
    for iter = 1...N:
        # 采 K 条候选 action 序列
        a_seq ~ N(μ, σ)
        # 对每条做 P 次 trajectory rollout（用 ensemble 传播不确定性）
        for k = 1...K:
            for p = 1...P:
                s ← s_t
                G_k^p = 0
                for h = 0...H-1:
                    i ~ Uniform(1..B)        # 随机选一个 ensemble 成员
                    s ~ f_φ^i(s, a_seq_k[h])
                    G_k^p += R_φ(s, a_seq_k[h])
            G_k = mean_p(G_k^p)
        # 用 top-elite 的 a_seq 重新估计 μ, σ
        elite = top K' samples by G_k
        μ, σ ← mean(elite), std(elite)
    
    return μ[0]   # 只执行第一步（MPC receding horizon）
```

**核心思想**：**模型不必学得多好，只要能"短期可信"地评价候选 action sequence 就够了**。规划做"真正的最优化"，model 只是 evaluator。

---

## 三、MBPO（Janner et al., 2019）—— 把 model 装进 SAC

### 3.1 架构

```
                   ┌──────────────────────────────┐
                   │      Real environment        │
                   └────────────┬─────────────────┘
                                │
                                ▼
                ┌────────────────────────────────────┐
                │       D_env  (真实 buffer)          │
                └────┬────────────────────────┬──────┘
                     │                        │
                     ▼  训练 model            │
       ┌──────────────────────────────┐       │
       │ NN ensemble  f_φ (dynamics + │       │
       │ reward)                       │       │
       └────────┬─────────────────────┘       │
                │  k 步短滚动                  │
                ▼                              │
       ┌──────────────────────────────────┐   │
       │   D_model (合成 buffer)           │   │
       └────────────┬─────────────────────┘   │
                    │            ┌─────────────┘
                    ▼            ▼
                  ┌────────────────────────┐
                  │    SAC actor + critic   │
                  │   (从 D_env ∪ D_model    │
                  │    采样训练)             │
                  └────────────┬───────────┘
                               ▼
                          执行 policy
```

### 3.2 关键公式

**模型** 跟 PETS 一样是 NN ensemble，输出 $(s',r) \sim \mathcal{N}(\mu, \Sigma)$。

**短滚动生成 imaginary transitions**：从 $D_{\text{env}}$ 随机抽一个真实 $s_t$ 作为起点，用 policy $\pi$ 和 model 走 $k$ 步（$k$ 通常 1~5）：

$$\begin{aligned}
a_t &\sim \pi(\cdot | s_t)\\
(s_{t+1}, r_t) &\sim f_\phi^i(s_t, a_t),\quad i \sim \text{Uniform}\\
\text{加入 } & D_{\text{model}}
\end{aligned}$$

**SAC 更新**：从混合 buffer 采样，**比例 $\beta$** 来自真实 buffer，$1-\beta$ 来自合成 buffer（典型 $\beta=0.05$，95% 是 imaginary 数据）：

$$\text{batch} \sim \beta \cdot D_{\text{env}} + (1-\beta) \cdot D_{\text{model}}$$

然后用标准 SAC 的 Q-loss / actor loss / α-loss 更新。

### 3.3 伪代码

```
初始化 model φ，policy π_θ，Q_ψ，D_env ← ∅，D_model ← ∅

for epoch = 1...N:
    # 1) 真环境交互
    for E steps:
        a ~ π_θ(s)
        s', r = env.step(a)
        D_env ← D_env ∪ {(s,a,r,s')}
    
    # 2) 训练 model（在 D_env 上）
    训练 ensemble f_φ 几个 epoch
    
    # 3) 生成合成数据
    for M 次：
        从 D_env 随机抽一个 s
        for h = 0...k-1:
            a ~ π_θ(s)
            (s', r) ~ f_φ^i(s, a)        # i 随机选 ensemble
            D_model ← D_model ∪ {(s,a,r,s')}
            s ← s'
    
    # 4) 用混合 buffer 训练 SAC
    for G steps:
        batch ~ β·D_env + (1-β)·D_model
        SAC update (Q, π, α)
```

**核心思想**：
1. **短 horizon 控制 model 误差**：$k$ 只有 1-5，模型即使不完美也不会偏太远；
2. **数据放大**：每个真实样本被放大成 ~M·k/E 个合成样本，sample efficiency 大幅提升；
3. **算法侵入小**：现有 SAC 代码几乎不动，加一个 model 训练 + rollout 模块即可。

---

## 四、Dreamer V3（Hafner et al., 2023）—— 隐空间 world model + actor-critic

### 4.1 架构（RSSM）

```
真实图像 obs                                 
   │                                         
   ▼                                         
┌──────┐    z_t (stochastic)                
│Enc q │ ──────────────┐                     
└──────┘               ▼                     
                  ┌────────────────────────┐ 
                  │  RSSM dynamics         │ 
                  │ (h_t, z_t) →           │ 
                  │  (h_{t+1}, ẑ_{t+1})    │ 
                  │  h: deterministic GRU   │ 
                  │  z: discrete categorical│ 
                  └──┬──────────┬───┬───────┘
                     │          │   │       
              ┌──────▼──┐  ┌────▼─┐ ▼───────┐
              │Decoder  │  │Reward│ Continue│
              │ → obs   │  │ → r  │ → flag  │
              └─────────┘  └──────┘ └───────┘
                                             
        训完之后：在 latent 里 imagine rollout 
        让 actor 学习                         
                                              
┌─────────────────────────────────────────────┐
│  Actor π(a | h, z)                          │
│  Critic V(h, z)                             │
│  二者都只在 latent 里训练                     │
└─────────────────────────────────────────────┘
```

### 4.2 公式

**RSSM 的隐空间动态**（Recurrent State Space Model）：
$$\begin{aligned}
h_{t+1} &= \text{GRU}(h_t, z_t, a_t) \quad &\text{(deterministic part)}\\
\hat z_{t+1} &\sim p_\phi(\hat z_{t+1} | h_{t+1}) \quad &\text{(prior: 没看见 obs 时的预测)}\\
z_{t+1} &\sim q_\phi(z_{t+1} | h_{t+1}, o_{t+1}) \quad &\text{(posterior: 看见 obs 时的编码)}
\end{aligned}$$

**World model 训练 loss**（变分下界）：
$$\mathcal{L}_{\text{WM}} = \underbrace{\mathbb{E}[\log p(o | h,z)]}_{\text{重建}} + \underbrace{\mathbb{E}[\log p(r | h,z)]}_{\text{奖励}} + \underbrace{\mathbb{E}[\log p(c | h,z)]}_{\text{episode 终止}} - \underbrace{\beta\cdot\text{KL}[q(z|h,o)\|p(\hat z|h)]}_{\text{先验/后验对齐}}$$

**Actor-Critic 在 imagination 里训练**（从真实 buffer 抽 $h_0$，在 latent 里展开 $H$ 步）：

Actor loss（用 reinforce + entropy）：
$$\mathcal{L}_\pi = -\mathbb{E}\Big[\sum_t \log\pi(a_t | h_t,z_t) \cdot \text{sg}(V_\lambda(h_t,z_t) - V(h_t,z_t)) + \eta H(\pi)\Big]$$

Critic loss（$\lambda$-return target）：
$$V_\lambda(h_t,z_t) = r_t + \gamma\big[(1-\lambda)V(h_{t+1},z_{t+1}) + \lambda V_\lambda(h_{t+1},z_{t+1})\big]$$
$$\mathcal{L}_V = \mathbb{E}\big[(V(h_t,z_t) - \text{sg}(V_\lambda(h_t,z_t)))^2\big]$$

### 4.3 伪代码

```
初始化 world model φ，actor π_θ，critic V_ψ，buffer D ← ∅

for iteration = 1...N:
    # 1) 用 policy + 真环境收集数据
    s ← env.reset(); h ← 0
    for t = 1...T:
        z ~ q_φ(z | h, encode(o_t))     # 编码当前观测到 latent
        a ~ π_θ(a | h, z)
        o_{t+1}, r_t = env.step(a)
        D ← D ∪ {(o_t, a_t, r_t, o_{t+1})}
        h ← GRU(h, z, a)
    
    # 2) 训练 world model
    从 D 采若干 sequence，最大化 ELBO L_WM
    
    # 3) 在 imagination 里训 actor-critic
    从 D 抽起点 (h_0, z_0)
    for h_step = 0...H-1:           # H 通常 15-50
        a_h ~ π_θ(a | h_h, z_h)
        ẑ_{h+1} ~ p_φ(ẑ | h_{h+1})  # 用 prior 预测（imagination）
        r̂_h = R_φ(h_h, z_h)
        h_{h+1} = GRU(h_h, z_h, a_h)
    
    # 4) 用 imagined trajectory 更新 actor, critic
    更新 L_V 和 L_π
```

**核心思想**：
1. **观测 -> latent -> 决策**：actor 只在低维 latent 里工作，速度快；
2. **想象训练**：actor 完全用 imagined trajectory 训练，**几乎不消耗真实 sample**；
3. **scale 友好**：DreamerV3 一组超参跨 150+ 任务都能跑。

---

## 五、TD-MPC2（Hansen et al., 2024）—— Latent + 短期 MPC + Q bootstrap

### 5.1 架构

```
       真实 obs s
           │
           ▼
       ┌──────┐  z = h_θ(s)
       │ Enc  │ ──────────────────────────────────────────┐
       └──────┘                                            │
                                                           ▼
              ┌──────────────────────────┐    ┌──────────────────┐
              │ Latent dynamics          │    │ Reward predictor │
              │ z' = f_θ(z, a)           │    │ r̂ = R_θ(z, a)    │
              └────────────┬─────────────┘    └──────────────────┘
                           │
                           ▼
              ┌──────────────────────────┐    ┌──────────────────┐
              │ Q value  Q_θ(z, a)        │    │ Policy prior     │
              │ (TD 更新)                 │    │ π_θ(a|z)         │
              └──────────────────────────┘    └──────────────────┘

      决策时：在 latent 里 MPPI 搜 H 步 + 终端用 Q 兜底
```

### 5.2 公式

**联合训练 loss**（5 件套同时学）：
$$\mathcal{L}_{\text{TOLD}} = \sum_{i=0}^{H-1} \lambda^i \Big[ c_1\underbrace{\|R_\theta(z_i,a_i) - r_i\|_2^2}_{\text{奖励}} + c_2\underbrace{\|Q_\theta(z_i,a_i) - y_i\|_2^2}_{\text{TD value}} + c_3\underbrace{\|z_{i+1} - \text{sg}(h_\theta(s_{i+1}))\|_2^2}_{\text{latent consistency}}\Big]$$

其中 TD target：
$$y_i = r_i + \gamma \cdot \bar Q_{\theta^-}\big(\bar z_{i+1}, \pi(\bar z_{i+1})\big),\quad \bar z = \text{sg}(h_\theta(s))$$

**Policy prior**（DPG）：
$$\mathcal{L}_\pi = -\mathbb{E}_{z\sim\mathcal{D}}\big[Q_\theta(z, \pi_\theta(z))\big]$$

**MPPI 决策目标**：
$$G(a_{0:H-1}) = \sum_{i=0}^{H-1}\gamma^i R_\theta(z_i, a_i) + \gamma^H Q_\theta\big(z_H, \pi(z_H)\big)$$

### 5.3 伪代码

```
初始化 encoder h_θ，dynamics f_θ，reward R_θ，Q_θ，policy prior π_θ，buffer D

# ===== 训练循环 =====
for iter = 1...N:
    # 1) 用 MPPI 决策与真环境交互
    s ← env.reset()
    for t = 1...T:
        a_t = PLAN(s_t)
        s_{t+1}, r_t = env.step(a_t)
        D ← D ∪ {(s_t, a_t, r_t, s_{t+1})}
    
    # 2) 从 D 采长度 H+1 子轨迹，联合训练 5 件套
    采 batch of (s_0, a_0, r_0, ..., s_H)
    z_0 = h_θ(s_0)
    L = 0
    for i = 0...H-1:
        L += λ^i · (
            c1·||R_θ(z_i, a_i) - r_i||² 
          + c2·||Q_θ(z_i, a_i) - y_i||² 
          + c3·||z_{i+1} - sg(h_θ(s_{i+1}))||²
        )
        z_{i+1} = f_θ(z_i, a_i)             # 在 latent 里 unroll
    更新 θ 最小化 L
    
    # 3) Policy prior 更新（DPG）
    最大化 Q_θ(z, π_θ(z))


# ===== 决策（推理） =====
function PLAN(s_t):
    z = h_θ(s_t)
    # 初始化 H 步动作分布（部分用上一时刻 plan 热启动 + 部分用 π 采样）
    μ, σ = init_distribution()
    
    for iter = 1...K_iter:
        # 采 N 条 action 序列（N 通常 512）
        A ~ N(μ, σ)                          # shape [N, H, dim_a]
        # 用 policy prior 采另一部分（让搜索方向有 actor 提示）
        A[:N_π] ~ π_θ rollout from z
        
        # 在 latent 里 simulate
        for each a_seq in A:
            z' = z; G = 0
            for i = 0...H-1:
                G += γ^i · R_θ(z', a_seq[i])
                z' = f_θ(z', a_seq[i])
            G += γ^H · Q_θ(z', π_θ(z'))      # 终端 value bootstrap
        
        # MPPI 软更新分布
        w = softmax(τ · G)
        μ = Σ w_k · A_k
        σ = sqrt(Σ w_k · (A_k - μ)²)
    
    return μ[0] + 探索噪声
```

**核心思想**：
1. **短模型 + 长 value**：用模型走 5 步（可信），剩下用 $Q$ 兜底（学到的）；
2. **不重建 observation**：只学"reward + value 相关的 latent"，比 Dreamer 更精简；
3. **决策时用 MPPI 主动思考**：和 Dreamer 直接用 actor 输出不同，每个决策都跑一次小型 trajectory optimization。

---

## 六、五种范式速查对比表

| 算法 | 模型预测什么 | 用 model 的方式 | 决策方式 | 关键创新 |
|---|---|---|---|---|
| **Dyna-Q** | 表格 $(s',r)$ | 合成数据扩充 Q 学习 | $\epsilon$-greedy | 最早把 model 和 model-free 融合 |
| **PETS** | $p(s',r\|s,a)$ NN ensemble | 决策时 CEM-MPC 评估 | MPC | 不确定性传播 |
| **MBPO** | NN ensemble | 短 rollout 扩充 SAC buffer | SAC actor | 短 horizon 控制 model 误差 |
| **DreamerV3** | RSSM latent + decoder | latent 里 imagine 训 actor-critic | actor 直接输出 | 像素+长视界、统一超参 |
| **TD-MPC2** | latent + reward + Q | latent 里 MPPI + Q bootstrap | MPPI 规划 | "短模型 + 长 value" 最佳折衷 |

---

## 七、给你一个能直接上手的建议路线

**如果你完全没碰过 model-based RL，按下面的顺序去读代码和上手**：

1. **理解 Dyna-Q**：看 Sutton 书 Ch. 8，自己写一个 4 房间 grid world 跑通。理解"用 model 扩充经验"这件事；
2. **跑通 MBPO**：开源实现 [JannerM/mbpo](https://github.com/JannerM/mbpo)，在 Hopper 上跑，体验 sample efficiency。**这一步会让你彻底懂"为什么要 model"**；
3. **理解 TD-MPC2**：开源 [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2)，在 Walker / ManiSkill 跑，体验**MPC 在决策时的"思考"过程**；
4. **进阶 DreamerV3**：[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) 或 danijar 官方版，体验**像素输入的 latent world model**；
5. **应用到 Kuavo**：如果是状态输入（关节角度+物体位姿）— 优先 **MBPO/TD-MPC2**；如果是像素 — **DreamerV3 / DayDreamer**。

**调试 model-based 算法的几个坑（早知道少走弯路）**：

- **Model overfit 比 underfit 更危险**：模型在 buffer 上拟合太好但泛化差，rollout 会越走越偏。所以 ensemble + 短 horizon 是标配；
- **Reward predictor 是关键短板**：在你 shaped reward 那种"平坦"奖励函数下，reward predictor 会学得不好。**建议用 dense reward + 必要时做 reward symlog**（DreamerV3 的做法）；
- **Latent collapse**：encoder 把不同状态都映到相同的 latent。Latent consistency loss + KL 约束是解决方案；
- **训练顺序**：先用真实数据让 model 收敛到一个合理水平，再开始 imagination 训 policy，否则 garbage in garbage out。

如果你想我把上面任何一个算法（比如 MBPO 或 TD-MPC2）的**核心代码逐行解读**，或者**对应到你 SBL3 的 SAC 上改成 MBPO 风格**，可以告诉我，我们可以下一步实战。