下面把三个模型的训练 loss 完整写出来，并附代码引用与数学一一对应，便于你直接对照。

---

## 1. ACT — Conditional VAE：**重构 + KL 正则**

### 1.1 模型本质

ACT 是一个 **CVAE**：用 vae_encoder 把（state, action）编码成 latent $z$，decoder 用 $(s, z)$ 重构 action。训练时 KL 把 $q_\phi(z\mid s, a)$ 拉向 $\mathcal N(0, I)$。

### 1.2 训练 loss（代码）

```141:159:src/lerobot/policies/act/modeling_act.py
actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

l1_loss = (
    F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
).mean()

loss_dict = {"l1_loss": l1_loss.item()}
if self.config.use_vae:
    mean_kld = (
        (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
    )
    loss_dict["kld_loss"] = mean_kld.item()
    loss = l1_loss + mean_kld * self.config.kl_weight
else:
    loss = l1_loss
```

### 1.3 公式（注意是 **L1 重构**，不是 MSE）

$$
\boxed{\;
\mathcal L_{\text{ACT}} \;=\; \underbrace{\mathbb E_{(s,a)\sim\mathcal D}\bigl[m \odot \|a - \hat a_\theta(s, z)\|_1\bigr]}_{\text{L1 reconstruction}}
\;+\; \lambda_{\text{KL}}\cdot\underbrace{\mathbb E_{(s,a)}\bigl[\mathrm{KL}\bigl(\,\mathcal N(\mu_\phi, \sigma_\phi^2)\;\Vert\;\mathcal N(0, I)\bigr)\bigr]}_{\text{latent KL on }z}
\;}
$$

其中 KL 由 reparameterization trick + 闭式 Gaussian KL：

$$
\mathrm{KL}\bigl(\mathcal N(\mu, \sigma^2)\,\Vert\,\mathcal N(0, I)\bigr) \;=\; -\tfrac{1}{2}\sum_d\bigl(1 + \log\sigma_d^2 - \mu_d^2 - \sigma_d^2\bigr)
$$

→ **ACT 的 KL 是显式的**，作用在 **latent $z$ 空间**（而不是 action 空间），目的是把 posterior 拉成可以采样的标准正态。

---

## 2. DP — Diffusion Policy (DDPM/DDIM)：**前向加噪 + 反向去噪 MSE**

### 2.1 模型本质

UNet $\hat y_\theta(x_t, t, c)$ 在加噪轨迹上预测**噪声 $\epsilon$ 或干净动作 $x_0$**。

### 2.2 训练 loss（代码）

```328:353:src/lerobot/policies/diffusion/modeling_diffusion.py
trajectory = batch[ACTION]
eps = torch.randn(trajectory.shape, device=trajectory.device)
timesteps = torch.randint(low=0, high=T, size=(B,), device=device).long()
noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

if self.config.prediction_type == "epsilon":
    target = eps
elif self.config.prediction_type == "sample":
    target = batch[ACTION]
else:
    raise ValueError(...)

loss = F.mse_loss(pred, target, reduction="none")
```

### 2.3 公式

**前向加噪（VP-SDE / DDPM 闭式跳跃）：**

$$
x_t \;=\; \sqrt{\bar\alpha_t}\,x_0 \;+\; \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad t\in\{1,\ldots,T\},\ \epsilon\sim\mathcal N(0, I)
$$

**训练损失（simplified DDPM objective，Ho et al. 2020）：**

$$
\boxed{\;
\mathcal L_{\text{DP}} \;=\; \mathbb E_{x_0,\,t\sim\mathcal U\{1,T\},\,\epsilon\sim\mathcal N(0, I)}\Bigl\|\,\hat y_\theta(x_t, t, c)\;-\;\text{target}_t\,\Bigr\|_2^2
\;}
$$

其中：
- $\text{target}_t = \epsilon$ 时，称 ε-prediction（最常用）
- $\text{target}_t = x_0$ 时，称 sample/x0-prediction

**它和 KL 的关系**：原始 ELBO 是

$$
\mathcal L_{\text{ELBO}} = \sum_{t=1}^T \mathrm{KL}\bigl(q(x_{t-1}\mid x_t, x_0)\,\Vert\,p_\theta(x_{t-1}\mid x_t)\bigr)
$$

在两边 Gaussian + 同方差 schedule 下，每个 KL 项**化简成 MSE on noise**，再去掉 timestep 加权 $w_t$ 就得到代码里的 simplified MSE。**所以 DP 的 MSE 是"被吸收/简化掉的 KL"**——不像 ACT 那样显式写出来。

---

## 3. FM — Conditional Flow Matching：**OT path 上的速度场 MSE**

### 3.1 模型本质

DiT + 多头 decoder $v_\theta(x_t, t, c)$ 在线性插值路径上预测**速度向量场**。

### 3.2 训练 loss（代码）

```925:940:src/lerobot/policies/groot/action_head/flow_matching_action_head.py
noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
t = t[:, None, None]
noisy_trajectory = (1 - t) * noise + t * actions
...
velocity = actions[:, :, :self.actual_action_dim] - noise[:, :, :self.actual_action_dim]
```

```1154:1196:src/lerobot/policies/groot/action_head/flow_matching_action_head.py
loss_left_arm  = F.mse_loss(pred_left_arm,  velocity_left_arm,  reduction="none") * action_mask_left_arm
loss_right_arm = F.mse_loss(pred_right_arm, velocity_right_arm, reduction="none") * action_mask_right_arm
loss_claw      = F.mse_loss(pred_claw,      velocity_claw,      reduction="none") * action_mask_claw
...
data_ps = (
    precision_left_arm  * ps_la
  + precision_right_arm * ps_ra
  + precision_claw      * ps_c
)
aux_loss = s_left_arm + s_right_arm + s_claw
if coordination_loss is not None:
    aux_loss = aux_loss + self.config.arm_coordination_loss_weight * coordination_loss
loss = data_ps.mean() + aux_loss
```

### 3.3 公式

**前向加噪（OT 直线路径）：**

$$
x_t \;=\; (1 - t)\,x_0 \;+\; t\,x_1,\qquad t\sim\mathcal U[0, 1],\ x_0 = \epsilon\sim\mathcal N(0, I),\ x_1 = a_{\text{GT}}
$$

**目标速度场：**

$$
v_t^{\text{GT}} \;=\; \frac{d x_t}{dt} \;=\; x_1 - x_0 \;=\; a - \epsilon
$$

**训练损失（你这版的完整形式 = CFM-MSE + 多头 + uncertainty + coordination）：**

$$
\boxed{\;
\mathcal L_{\text{FM}}
\;=\; \underbrace{\sum_{k\in\{\text{LA, RA, C}\}}\frac{1}{\sigma_k^2}\,\mathbb E_{t,\,a,\,\epsilon}\bigl[m_k\odot\|v_\theta^k(x_t, t, c) - (a_k - \epsilon_k)\|_2^2\bigr]}_{\text{Conditional Flow Matching MSE (per task, uncertainty-weighted)}}
\;+\; \underbrace{\sum_k \log\sigma_k}_{\text{anti-degen.}}
\;+\; \underbrace{\lambda_{\text{coord}}\bigl(\|v_\theta^L\| - \|v_\theta^R\|\bigr)^2}_{\text{coordination reg}}
\;}
$$

→ **完全没有 KL 项**。理论保证由 Lipman et al. 2023 的 CFM 等价定理给出：最小化条件 velocity MSE 等价于最小化边缘 velocity MSE，进而保证 $v_\theta$ 学到从 $\mathcal N(0, I)$ 到 $p_{\text{data}}$ 的传输向量场。

---

## 4. 三者并排对比

| 维度              | **ACT（CVAE）**                                                    | **DP（DDPM/DDIM）**                                                             | **FM（OT-CFM）**                              |
| --------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------- |
| **生成模型范式**      | 变分自编码器                                                           | 扩散模型                                                                          | 条件流匹配                                       |
| **核心数学对象**      | latent $z$ 后验 $q_\phi(z\mid s, a)$                               | 反向 Markov 链 $p_\theta(x_{t-1}\mid x_t)$                                       | 时间相关向量场 $v_\theta(x_t, t)$                  |
| **前向"加噪"路径**    | 无（直接 amortized inference 出 $z$）                                  | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$（**曲线 VP-SDE**） | $x_t = (1-t)\epsilon + t\,a$（**OT 直线**）     |
| **时间步 $t$**     | 无时间                                                              | 离散 $t\in\{1,\ldots,T\}$，典型 $T=1000$                                           | 连续 $t\in[0,1]$                              |
| **模型预测什么**      | 重构动作 $\hat a$（同时预测 $\mu_\phi, \log\sigma_\phi^2$）                | $\epsilon$ 或 $x_0$                                                            | velocity $v = a - \epsilon$                 |
| **重构损失类型**      | **L1**（mask 后求平均）                                                | **MSE on noise/sample**                                                       | **MSE on velocity**                         |
| **是否有显式 KL**    | ✅ **有**：$\mathrm{KL}(q_\phi(z\mid s,a)\,\Vert\,\mathcal N(0,I))$ | ❌ 没有（被简化进 MSE）                                                                | ❌ 没有（不需要）                                   |
| **KL 的角色**      | 把 latent 后验拉成可采样的高斯 prior                                        | 推导起源（ELBO），但代码里被 reparameterize 成 MSE                                         | 完全不出现，**CFM 等价定理代替**                        |
| **理论保证机制**      | 变分下界（ELBO）                                                       | 变分下界 + Gaussian KL 闭式化简                                                       | OT 流的边缘等价定理（Lipman et al. 2023）             |
| **推理阶段**        | 单步 forward（$z=0$ 或采样）                                            | 多步去噪（$T \sim 100$ for DDIM）                                                   | 少步 ODE 积分（$T \sim 4{-}10$ Euler 步）          |
| **额外辅助损失（你这套）** | —                                                                | —                                                                             | + Uncertainty weighting + Coordination loss |
| **代码 loss 函数**  | `F.l1_loss + Gaussian KL`                                        | `F.mse_loss(pred, eps or x0)`                                                 | `F.mse_loss(pred, a − ε)`                   |

---

## 5. 三个公式总览（一页可背版）

$$
\boxed{
\begin{aligned}
\mathcal L_{\text{ACT}} \;&=\; \mathbb E\bigl[m\odot\|a - \hat a_\theta(s, z)\|_1\bigr]
\;+\; \lambda_{\text{KL}}\cdot\mathbb E\bigl[\,\mathrm{KL}\bigl(\mathcal N(\mu_\phi, \sigma_\phi^2)\,\Vert\,\mathcal N(0, I)\bigr)\bigr] \\[6pt]
\mathcal L_{\text{DP}} \;&=\; \mathbb E_{x_0,\,t,\,\epsilon}\,\bigl\|\,\hat y_\theta\bigl(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\, t,\, c\bigr) - \text{target}_t\,\bigr\|_2^2,\quad \text{target}_t\in\{\epsilon,\,x_0\} \\[6pt]
\mathcal L_{\text{FM}} \;&=\; \sum_k\frac{1}{\sigma_k^2}\,\mathbb E_{t,\,a,\,\epsilon}\bigl\|\,v_\theta^k\bigl((1-t)\epsilon + t\,a,\, t,\, c\bigr) - (a_k - \epsilon_k)\,\bigr\|_2^2
\;+\; \sum_k \log\sigma_k
\;+\; \lambda_{\text{coord}}\bigl(\|v^L\| - \|v^R\|\bigr)^2
\end{aligned}
}
$$

---

## 6. 三个 loss 在"概率/几何"层面的差异

| | **ACT** | **DP** | **FM** |
|---|---|---|---|
| 学的"分布"对象 | latent space $q(z)$ | data distribution $p(x_0)$（隐式） | data distribution $p(x_1)$（隐式） |
| 是否拟合显式概率密度 | ✅ 拟合 latent 后验 | ❌ 通过 score / noise 间接 | ❌ 通过 velocity 间接 |
| 是否需要 KL 项 | ✅ 是机制核心 | △ 推导起源，代码里没有 | ❌ 不需要 |
| 一次推理的成本 | O(1) | O(几十~百步) | O(几步) |
| Loss 与"数据似然"的关系 | ELBO 下界（紧致） | simplified ELBO（去权重后近似） | 边缘 vector field MSE（精确等价定理） |

---

## 7. 一句话收口（可背版）

> "**三套 policy 的 loss 表面都长得像 MSE / L1，但理论根基完全不同：**
>
> - **ACT** = L1 重构 + 显式 KL 正则在 latent；KL 是核心机制，把 posterior 拉成可采样的 Gaussian prior；
> - **DP** = MSE on $\epsilon$ 或 $x_0$；KL 在 ELBO 推导里出现，但被 Gaussian + 闭式简化吸收成纯 MSE，所以代码里看不见；
> - **FM** = MSE on velocity（$a - \epsilon$）；不需要 KL，理论保证来自 Lipman et al. 2023 的 OT 流等价定理。
>
> **'有没有 KL'不是质量好坏的标准**——它反映了三种生成范式不同的概率公理：变分下界（ACT）、变分马尔可夫链（DP）、边缘流匹配（FM）。"



# Loss解析
# 3. L1 Loss 到底长什么样？

举例：

真实：

$$  
a=[1,2,3]  
$$

预测：

$$  
\hat a=[1.5,1,5]  
$$

误差：

$$  
[0.5,1,2]  
$$

L1：

# $$  
|0.5|+|1|+|2|

3.5  
$$

---

# 4. L2 Loss 又是什么？

L2：

# $$  
|a-\hat a|_2

\sqrt{\sum_i (a_i-\hat a_i)^2}  
$$

注意：

## L2 是“欧氏距离”

它是真正几何意义上的距离。

上面的例子：

# $$  
\sqrt{0.5^2+1^2+2^2}

\sqrt{5.25}  
\approx2.29  
$$

---

# 5. MSE 又是什么？

MSE：

# $$  
\text{MSE}

\frac1D\sum_i(a_i-\hat a_i)^2  
$$

其实就是：

> “平方后的平均”

上例：

# $$  
\frac{0.5^2+1^2+2^2}{3}

1.75  
$$
