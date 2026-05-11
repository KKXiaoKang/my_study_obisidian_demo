下面是完整的对比表，从最早的 DQN v1 一路到 SAC，每个算法都拆开讲 **online 网络** 和 **target 网络** 的具体分工。

---

# 一、总览大表

| 算法 | 年份 | 动作 | 网络结构 | TD Target | Critic Loss | Actor Loss / 选动作 |
|---|---|---|---|---|---|---|
| **DQN v1** | 2013 | 离散 | **1 个** Q 网络 $Q_\theta$ | $r + \gamma\, \max_{a'} Q_\theta(s', a')$ | $\mathbb{E}\big[(Q_\theta(s,a) - y)^2\big]$ | $a = \arg\max_a Q_\theta(s, a)$ |
| **DQN v2** | 2015 | 离散 | online $Q_\theta$ + target $Q_{\bar\theta}$（硬更新） | $r + \gamma\, \max_{a'} Q_{\bar\theta}(s', a')$ | $\mathbb{E}\big[(Q_\theta(s,a) - y)^2\big]$ | $a = \arg\max_a Q_\theta(s, a)$ |
| **Double DQN** | 2016 | 离散 | online + target（同 v2） | $r + \gamma\, Q_{\bar\theta}\big(s',\, \arg\max_{a'} Q_\theta(s', a')\big)$ | $\mathbb{E}\big[(Q_\theta(s,a) - y)^2\big]$ | $a = \arg\max_a Q_\theta(s, a)$ |
| **DDPG** | 2015 | 连续 | actor $\mu_\phi$ + critic $Q_\omega$ + 各自 target（**软更新**） | $r + \gamma\, Q_{\bar\omega}\big(s',\, \mu_{\bar\phi}(s')\big)$ | $\mathbb{E}\big[(Q_\omega(s,a) - y)^2\big]$ | $-\mathbb{E}\big[Q_\omega(s, \mu_\phi(s))\big]$ |
| **TD3** | 2018 | 连续 | actor + **twin** critic + 各自 target | $r + \gamma\, \min_{i=1,2} Q_{\bar\omega_i}\big(s',\, \mu_{\bar\phi}(s') + \epsilon\big)$ | $\mathbb{E}\big[(Q_{\omega_i}(s,a) - y)^2\big],\, i=1,2$ | $-\mathbb{E}\big[Q_{\omega_1}(s, \mu_\phi(s))\big]$（**延迟更新**） |
| **SAC** | 2018 | 连续 | 随机 actor $\pi_\theta$ + twin critic + critic target（**无 actor target**） | $r + \gamma\big[\min_{i=1,2} Q_{\bar\omega_i}(s', a') - \alpha\log\pi_\theta(a'\|s')\big]$, $a' \sim \pi_\theta$ | 同上 | $\mathbb{E}_{a\sim\pi}\big[\alpha\log\pi(a\|s) - \min_i Q_{\omega_i}(s,a)\big]$ |

---

# 二、逐项详解（重点回答你想强调的几个点）

## 1. DQN v1（2013, NIPS workshop）—— 起点，**只有一个 Q 网络**

**网络**：单一 $Q_\theta$（没有 target network！）

**TD Target**：
$$y = r + \gamma\, \max_{a'} Q_\theta(s', a')$$

**Loss**：
$$L(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\Big[\big(Q_\theta(s,a) - y\big)^2\Big]$$

**逐项功能**：
| 项 | 用什么网络 | 功能 |
|---|---|---|
| $Q_\theta(s,a)$ | $Q_\theta$ | TD 误差的"当前估计" |
| $\max_{a'} Q_\theta(s', a')$ | $Q_\theta$ | TD 误差的"目标值" |

**致命问题**：左边 $Q_\theta(s,a)$ 和右边 $\max Q_\theta(s',a')$ 用的是**同一个**网络。一旦更新 $\theta$，target 也跟着变 → "追自己尾巴" → 训练经常发散。

**只引入了**：experience replay buffer。

---

## 2. DQN v2（2015 Nature）—— 加入 target network（**硬更新**）

**网络**：online $Q_\theta$ + target $Q_{\bar\theta}$（每 $C=10000$ 步硬复制 $\bar\theta \leftarrow \theta$）

**TD Target**：
$$y = r + \gamma\, \max_{a'} Q_{\bar\theta}(s', a')$$

### ⭐ Online critic 和 target critic 的分工（DQN v2 版）

| online $Q_\theta$ 在哪里用 | target $Q_{\bar\theta}$ 在哪里用 |
|---|---|
| ① 行为策略选动作：$a = \arg\max_a Q_\theta(s, a)$ | ① 算 TD target 里的 $\arg\max_{a'}$（**target 既选 a'**） |
| ② 算 TD loss **左边** $Q_\theta(s_t, a_t)$ | ② 算 TD target 里的 $Q$ 值（**target 也估值**） |
| ③ 被梯度下降训练 | ③ 每 C 步从 online 整体硬复制，**不被训练** |

⚠️ **关键点**：在 DQN v2 的 TD target 公式 $\max_{a'} Q_{\bar\theta}(s', a')$ 里——
- **"选 a'"和"估 Q"两件事都由 target 一个网络包办**
- online 网络只在 TD loss 的**左半边**出场，不参与 target 的计算

**改进于谁**：相对 v1 加入了 target network，解决了"追自己尾巴"的不稳定问题。

---

## 3. Double DQN（2016）—— ⭐ 重点：把 TD target 里的"选"和"估"解耦

**网络**：与 DQN v2 **完全一样**（online + target），**网络数量没有变**。

**TD Target**（只改了这一行）：
$$y = r + \gamma\, Q_{\bar\theta}\Big(s',\, \underbrace{\arg\max_{a'} Q_{\theta}(s', a')}_{\text{改用 online 选}}\Big)$$

### ⭐ Online critic 和 target critic 的分工（Double DQN 版）

| online $Q_\theta$ 在哪里用 | target $Q_{\bar\theta}$ 在哪里用 |
|---|---|
| ① 行为策略选动作 | ① ❌（不再在 target 里选 a'） |
| ② 算 TD loss **左边** $Q_\theta(s_t, a_t)$ | ② 算 TD target 里的 $Q$ 值（**只估值**） |
| **③ 新增：在 TD target 里挑 a'**：$\arg\max_{a'} Q_\theta(s', a')$ | ③ 每 C 步从 online 硬复制 |
| ④ 被梯度下降训练 | ④ 不被训练 |

### 🎯 DQN v2 与 Double DQN 的核心差异（你要的对比）

```
═══════════ DQN v2 ═══════════════════════════════
  TD target = r + γ · max_{a'} Q_θ̄(s', a')
                      ─────  ─────────────
                      选 a'   估这个 a' 的 Q 值
                       ↑          ↑
                       两个都用 target 网络

═══════════ Double DQN ═══════════════════════════
  TD target = r + γ · Q_θ̄(s',  argmax_{a'} Q_θ(s', a'))
                      ─────             ─────
                      估这个 a' 的 Q 值    选 a'
                        ↑                  ↑
                      用 target          用 online   ← 解耦！
```

**一句话总结**：**DQN v2 是"target 既选 a' 又估 Q 值"；Double DQN 把"选"交给 online、"估"留给 target**。网络结构没变，只是 TD target 公式里 online 多了一个用途。

**为什么这么改能减小高估**：online 和 target 是不同时刻的快照，它们的随机误差不容易在**同一个动作**上正向叠加 → 互相抵消 overestimation。

---

## 4. DDPG（2015/2016, ICLR）—— ⭐ 重点：online/target critic + 软更新 + actor 替代 argmax

**网络**：4 个
- **online actor** $\mu_\phi$（输入 s，输出确定性动作）
- **target actor** $\mu_{\bar\phi}$
- **online critic** $Q_\omega$（输入 s+a，输出标量 Q）
- **target critic** $Q_{\bar\omega}$

**软更新（DDPG 的标志性改动）**：
$$\bar\omega \leftarrow \tau\omega + (1-\tau)\bar\omega,\quad \bar\phi \leftarrow \tau\phi + (1-\tau)\bar\phi,\quad \tau \approx 0.005$$
每一步都做一次小幅 EMA，不再像 DQN 那样每 C 步硬跳。

**TD Target**：
$$y = r + \gamma\, Q_{\bar\omega}\Big(s',\, \underbrace{\mu_{\bar\phi}(s')}_{\text{target actor 直接给动作，替代 argmax}}\Big)$$

**Critic Loss**：
$$L(\omega) = \mathbb{E}\Big[\big(Q_\omega(s,a) - y\big)^2\Big]$$

**Actor Loss**（让 actor 输出能最大化 Q 的动作）：
$$L(\phi) = -\mathbb{E}_{s\sim\mathcal{D}}\Big[Q_\omega\big(s, \mu_\phi(s)\big)\Big]$$

### ⭐ Online critic 和 target critic 的分工（DDPG 版）

| online $Q_\omega$ 在哪里用 | target $Q_{\bar\omega}$ 在哪里用 |
|---|---|
| ① 算 TD loss **左边** $Q_\omega(s_t, a_t)$ | ① 算 TD target 里的 Q 值 |
| **② 给 actor 提供"评分"**：actor loss 用 $-Q_\omega(s, \mu_\phi(s))$ 引导 actor 朝高 Q 方向走 | ② 不参与 actor 更新 |
| ③ 被梯度下降训练 | ③ 软更新慢慢跟随 online |

**Online actor $\mu_\phi$ 和 target actor $\mu_{\bar\phi}$ 的分工**：

| online $\mu_\phi$ | target $\mu_{\bar\phi}$ |
|---|---|
| 与环境交互选动作（加探索噪声） | 只在 TD target 里给出 $a' = \mu_{\bar\phi}(s')$ |
| 被 actor loss 训练 | 软更新跟随 online actor |

**改进于谁（相对 Double DQN）**：
- 离散 → 连续：用 actor $\mu$ 替代 argmax
- critic 输出从向量改为标量
- 硬更新 → 软更新（EMA）

---

## 5. TD3（2018）—— ⭐ 重点：twin critic + min Q（悲观估计）

**网络**：6 个（critic 翻倍）
- online actor $\mu_\phi$、target actor $\mu_{\bar\phi}$
- **online twin critics $Q_{\omega_1}, Q_{\omega_2}$**（独立初始化、独立训练）
- **target twin critics $Q_{\bar\omega_1}, Q_{\bar\omega_2}$**

**TD Target**：
$$y = r + \gamma\, \underbrace{\min_{i=1,2}}_{\text{⭐ TD3 招牌}} Q_{\bar\omega_i}\Big(s',\, \mu_{\bar\phi}(s') + \underbrace{\epsilon}_{\text{policy smoothing}}\Big)$$
其中 $\epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$。

**Critic Loss**（两个 critic 都用同一个 y 训练）：
$$L(\omega_i) = \mathbb{E}\Big[\big(Q_{\omega_i}(s,a) - y\big)^2\Big],\quad i = 1, 2$$

**Actor Loss**（只用 critic 1，**每 $d$ 步才更新一次**）：
$$L(\phi) = -\mathbb{E}\Big[Q_{\omega_1}\big(s, \mu_\phi(s)\big)\Big]$$

### ⭐ 重点：min Q 是怎么工作的

```
                ┌─ Q_ω̄_1(s', μ_φ̄(s')+ε) ──┐
                │                          │
target action ──┤                          ├──→ min ──→ 接入 TD target
μ_φ̄(s')+ε       │                          │
                └─ Q_ω̄_2(s', μ_φ̄(s')+ε) ──┘
```

**为什么取 min 能抑制高估**：
- 神经网络估的 Q 都有正负误差
- 单一 critic 取 max 后被系统性高估
- 两个 critic 独立训练 → 它们的高估方向**不会完全一致**
- 取最小值 = 一个保守的（悲观的）下界估计

**TD3 三个改进**：
| 改进 | 公式上的体现 | 解决什么 |
|---|---|---|
| ① Clipped double-Q | $\min_{i=1,2}$ | Q 值高估偏差 |
| ② Target policy smoothing | $+ \epsilon$ | critic 过拟合 narrow peak |
| ③ Delayed actor update | actor 每 $d=2$ 步更一次 | actor 学到不准的 Q |

**改进于谁（相对 DDPG）**：连续动作版的"Double DQN + 防过拟合 + 更稳"，**所有改动都在解决 DDPG 训练崩塌、Q 值爆炸的问题**。

---

## 6. SAC（2018/2019）—— ⭐ 重点：引入熵项做最大熵 RL

**网络**：5 个（比 TD3 少一个 target actor）
- **随机** online actor $\pi_\theta(a|s)$（输出动作分布的 mean、std）—— **没有 target actor**
- online twin critics $Q_{\omega_1}, Q_{\omega_2}$
- target twin critics $Q_{\bar\omega_1}, Q_{\bar\omega_2}$
- （可学习温度 $\alpha$）

**最大熵 RL 目标**（区别于 DQN 系的根源）：
$$J(\pi) = \mathbb{E}\Big[\sum_t \gamma^t \big(\, r_t + \alpha\, \mathcal{H}(\pi(\cdot|s_t))\,\big)\Big]$$
"在最大化奖励的同时也最大化策略熵"。

**TD Target**（核心！）：
$$y = r + \gamma\,\Big[\, \underbrace{\min_{i=1,2} Q_{\bar\omega_i}(s', a')}_{\text{① 继承 TD3 的 min}} \;-\; \underbrace{\alpha\, \log\pi_\theta(a'|s')}_{\text{② SAC 引入的熵项}}\,\Big]$$
其中 $a' \sim \pi_\theta(\cdot|s')$。

**Critic Loss**：
$$L(\omega_i) = \mathbb{E}\Big[\big(Q_{\omega_i}(s,a) - y\big)^2\Big],\quad i = 1, 2$$

**Actor Loss**（让动作 Q 大且熵大）：
$$L(\theta) = \mathbb{E}_{s\sim\mathcal{D},\, a\sim\pi_\theta(\cdot|s)}\Big[\, \alpha\, \log\pi_\theta(a|s) - \min_{i=1,2} Q_{\omega_i}(s, a)\,\Big]$$

**Temperature Loss**（α 自动调参）：
$$L(\alpha) = \mathbb{E}_{a\sim\pi_\theta}\Big[-\alpha\big(\log\pi_\theta(a|s) + \mathcal{H}_{\text{target}}\big)\Big]$$

### ⭐ 重点：熵项 $-\alpha\log\pi(a'|s')$ 是怎么来的、起什么作用

**数学来源**：在最大熵 RL 框架下，贝尔曼方程变成 **soft Bellman equation**：
$$
V^*(s) = \mathbb{E}_{a\sim\pi}\big[\, Q^*(s, a) - \alpha\log\pi(a|s)\,\big]
$$
那个 $-\alpha\log\pi$ 就是熵的具体形式（因为 $\mathcal{H}(\pi) = \mathbb{E}_{a\sim\pi}[-\log\pi]$）。所以 DQN 的 $\max_a Q$ 在 SAC 里变成：

$$\max_a Q(s', a) \;\longrightarrow\; \mathbb{E}_{a'\sim\pi}\big[Q(s', a') - \alpha\log\pi(a'|s')\big]$$

**逐项含义**：
| 项 | 含义 |
|---|---|
| $a' \sim \pi_\theta$ | 从随机策略采样一个动作（不是确定性的 argmax）|
| $\min_{i=1,2} Q_{\bar\omega_i}$ | TD3 的悲观 Q 估计 |
| $\log\pi_\theta(a'\|s')$ | actor 在 a' 处的对数概率（actor 提供）|
| $\alpha$ | 温度系数，控制熵权重 |
| 整个 $-\alpha\log\pi$ | 熵奖励——动作越随机，加分越多 |

**作用**：
1. **鼓励探索**：熵大 → 动作分布更平坦 → 不会塌缩到单一动作
2. **多模态最优**：如果有多条同样好的策略，π 可以同时保留它们的概率
3. **鲁棒性**：对环境扰动更稳定

**为什么 SAC 不需要 target actor**：策略本身是随机的，已经自带 stochasticity，不需要额外做"延迟动作"来稳定。

**改进于谁（相对 TD3）**：
| 改动 | 目的 |
|---|---|
| 确定性 actor → **随机 actor** | 支持熵框架 |
| 加 $-\alpha\log\pi$ 熵项 | 鼓励探索 / 多模态 |
| 去掉 target actor | 随机策略不需要 |
| α 自动调参 | 免手动调温度 |

---

# 三、Online vs Target 网络分工的**完整演化路径**

| 阶段 | online 干什么 | target 干什么 |
|---|---|---|
| **DQN v1** | 全部！既算 loss 左边，也算 TD target | ❌ 不存在 |
| **DQN v2** | 算 loss 左边 + 行为策略选动作 | TD target 里**既选 a' 又估 Q**（一手包办）|
| **Double DQN** | 算 loss 左边 + 行为策略选动作 + **TD target 里选 a'** | TD target 里**只估 Q 值** |
| **DDPG** | online critic 算 loss 左边 + 给 actor 评分；online actor 选动作 | target critic 估 Q；target actor 给 a' |
| **TD3** | 同 DDPG，但 online critic 变成两个独立的（$Q_{\omega_1}, Q_{\omega_2}$）| 同 DDPG，但 target critic 也变两个，**取 min** |
| **SAC** | online actor 是随机的，提供 a' 和 $\log\pi$；online critics 同 TD3 | target critics 同 TD3，**没有 target actor** |

---

# 四、TD Target 演化的"一行串接"

把每个算法的 TD target 写在一起，能清楚看到每代到底加了什么：

$$
\begin{aligned}
\text{DQN v1:}\quad   y &= r + \gamma\, \boxed{\max_{a'} Q_\theta(s', a')} \\[4pt]
\text{DQN v2:}\quad   y &= r + \gamma\, \boxed{\max_{a'} Q_{\bar\theta}(s', a')} &&\leftarrow \text{加 target} \\[4pt]
\text{Double DQN:}\quad y &= r + \gamma\, Q_{\bar\theta}\!\left(s',\, \boxed{\arg\max_{a'} Q_\theta(s', a')}\right) &&\leftarrow \text{选/估解耦} \\[4pt]
\text{DDPG:}\quad     y &= r + \gamma\, Q_{\bar\omega}\!\left(s',\, \boxed{\mu_{\bar\phi}(s')}\right) &&\leftarrow \text{连续动作 + 软更新} \\[4pt]
\text{TD3:}\quad      y &= r + \gamma\, \boxed{\min_{i=1,2}} Q_{\bar\omega_i}\!\left(s',\, \mu_{\bar\phi}(s') + \boxed{\epsilon}\right) &&\leftarrow \text{min + smoothing} \\[4pt]
\text{SAC:}\quad      y &= r + \gamma\, \Big[\min_{i=1,2} Q_{\bar\omega_i}(s', a') \;\boxed{-\,\alpha\log\pi_\theta(a'|s')}\Big] &&\leftarrow \text{熵项}
\end{aligned}
$$

每一行框出来的部分，就是这一代相对上一代的"新增"或"改写"。

---

把这张完整表存到你的 `DQN_vs_贝尔曼.md` 里，以后看任何 critic 系强化学习代码都能秒对应到对应的算法和创新点。