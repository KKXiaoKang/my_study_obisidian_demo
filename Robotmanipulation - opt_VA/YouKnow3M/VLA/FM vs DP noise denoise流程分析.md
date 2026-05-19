###  DP / FM：表征是"采样过程"而不是"显式分布"

DP 和 FM 的确解决了 mode averaging（因为它们隐式表达分布），但有几个深层局限：

- 隐式表征不可解释：你拿不到一个"动作 embedding"做下游任务（检索、组合、解释）。
- 生成是黑盒：模型直接学 $\varepsilon$ 或 $v$ ，但没有显式的"动作概念"。比如"抓"和"推"在网络里没有任何符号化的区分。
- 难以做 plan/compose：DP/FM 生成的是 numerical sequence，没法像 LLM 那样做 chain-of-thought、做 in-context learning。
- 每个任务都要 fine-tune：因为动作空间是 raw float，跟语言空间断裂。

## 一、你对 FM 流程的理解：基本正确

你说的：
> 首先确定一个噪声 → 对动作加噪 → 干净动作和噪声相减得到 GT velocity → 模型学习预测速度场 → 推理时从噪声出发用速度场还原动作

完全对。再精确一点，看 GR00T 的训练代码：

```925:940:src/lerobot/policies/groot/action_head/flow_matching_action_head.py
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        # 3) 随机采样时间步 t ∈ [0, 1]
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast
        # 4) 创建加噪轨迹（Flow Matching 核心）
        # 当 t=0：纯噪声
        # 当 t=1：真实 action
        noisy_trajectory = (1 - t) * noise + t * actions
        ...
        velocity = actions[:, :, :self.actual_action_dim] - noise[:, :, :self.actual_action_dim]
```

对应的数学公式是：

**加噪轨迹（直线插值）：**

$$
x_t = (1-t)\,\varepsilon + t\,x_1, \quad t \in [0, 1]
$$

其中 $\varepsilon \sim \mathcal{N}(0, I)$ 是噪声，$x_1$ 是真实动作。

**GT velocity（路径切向量）：**

$$
v^* = \frac{dx_t}{dt} = x_1 - \varepsilon
$$

注意一个细节：**GT velocity 是 $x_1 - \varepsilon$，跟 $t$ 没有关系**（同一对 $(x_1, \varepsilon)$ 在直线路径上任何位置 $t$ 的切向量都是同一个常数向量），但训练时**输入给模型的 $x_t$ 是带 $t$ 的**——也就是说模型要学到：**无论我在直线路径上哪个位置，都能正确指出朝着干净动作的方向**。

**训练目标：**

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\, x_1,\, \varepsilon}\Big[\,\big\|\, v_\theta(x_t, t, c) - (x_1 - \varepsilon)\,\big\|^2 \,\Big]
$$

其中 $c$ 是条件（视觉+语言+state），$v_\theta$ 是模型预测的速度场。

**推理（欧拉积分）：**

```1356:1382:src/lerobot/policies/groot/action_head/flow_matching_action_head.py
        dt = 1.0 / num_steps
        # 2. 迭代去噪（例如 4 步）
        for t in range(num_steps):
            ...
            v_t = denoise_step_partial_call(x_t)
            ...
            x_t = x_t + dt * v_t
```

对应公式：

$$
x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c), \quad \Delta t = \frac{1}{N}
$$

从 $x_0 = \varepsilon$ 出发，迭代 $N$ 步（GR00T 默认 $N=4$）到达 $x_1 \approx$ 真实动作。

## 二、需要纠正的误区：前半部分加噪也不同

> "DP 和 FM 的前半部分处理都是对真实动作加噪，只是后半部分不同"

这里**不太准确**。两者的加噪公式形式上虽然都是"动作 + 噪声的混合"，但**加噪轨迹的几何形状完全不同**。

**DP 的加噪公式：**

$$
x_t = \sqrt{\bar\alpha_t}\, x_1 + \sqrt{1 - \bar\alpha_t}\, \varepsilon
$$

其中 $\bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$ 由 noise schedule（linear / cosine）决定，是 $t$ 的**非线性函数**。比如 cosine schedule：

$$
\bar\alpha_t = \cos^2\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)
$$

**FM 的加噪公式：**

$$
x_t = (1-t)\,\varepsilon + t\,x_1
$$

| 维度 | DP | FM |
|------|----|----|
| 加噪公式 | $x_t = \sqrt{\bar\alpha_t}\, x_1 + \sqrt{1-\bar\alpha_t}\, \varepsilon$ | $x_t = (1-t)\,\varepsilon + t\, x_1$ |
| 系数 | **非线性**（带 $\sqrt{\cdot}$ 的 schedule） | **线性** |
| 时间方向 | $t=0$ 干净，$t=T$ 纯噪声 | **反过来**：$t=0$ 纯噪声，$t=1$ 干净 |
| 路径几何 | **弯曲** | **直线** |
| 端点行为 | 渐进（$T\to\infty$ 才到纯噪声） | **精确**（$t=0$ 严格是 $\varepsilon$，$t=1$ 严格是 $x_1$） |

看 DP 代码：

```329:339:src/lerobot/policies/diffusion/modeling_diffusion.py
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
```

那个 `noise_scheduler.add_noise()` 内部就是 $\sqrt{\bar\alpha_t}\, x_1 + \sqrt{1-\bar\alpha_t}\, \varepsilon$。因为 $\sqrt{\bar\alpha_t}$ 和 $\sqrt{1-\bar\alpha_t}$ 这两个系数**不满足 $a(t) + b(t) = 1$**（只满足 $a(t)^2 + b(t)^2 = 1$，是个**圆弧约束**），所以 $x_t$ 在 $(x_1, \varepsilon)$ 之间走的不是直线，而是**一条弯曲的弧线**。

更精确地说，DP 的路径在 $(\sqrt{\bar\alpha_t}, \sqrt{1-\bar\alpha_t})$ 这个二维系数空间里走的是**单位圆的一段弧**：

$$
\big(\sqrt{\bar\alpha_t}\big)^2 + \big(\sqrt{1-\bar\alpha_t}\big)^2 = 1
$$

而 FM 的路径在系数空间里走的是 $(1-t, t)$ 这条**单位线段**：

$$
(1-t) + t = 1
$$

示意图：

```
干净动作 x₁              干净动作 x₁
   ●─.                      ●
       `.                   │
         `.  DP             │ FM
           `.               │
             `.             │
               `.           │
                 `.         │
                   `─●      ●
                   噪声 ε   噪声 ε
                   
     弯曲弧线              直线路径
   （需要50步逼近）       （4步即可）
```

DP 的弧线必须很多步才能精确逼近；FM 的直线本质上**只需要 1 步**就能精确到达（如果速度估计完全准确的话），实际中因为模型有误差需要 4~10 步做修正。

## 三、监督目标的对比

**DP 预测噪声（$\epsilon$-prediction）：**

$$
\mathcal{L}_{\text{DP}} = \mathbb{E}_{t,\,x_1,\,\varepsilon}\Big[\,\big\|\,\epsilon_\theta(x_t, t, c) - \varepsilon\,\big\|^2\,\Big]
$$

要从 $\epsilon_\theta$ 反推出干净动作 $x_1$，需要 Tweedie 公式：

$$
\hat{x}_1 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}}
$$

这一步需要 $\sqrt{\bar\alpha_t}$ 做分母，在 $t \to T$（$\bar\alpha_t \to 0$）时**数值不稳定**。

**FM 预测速度场（v-prediction in OT-CFM sense）：**

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\,x_1,\,\varepsilon}\Big[\,\big\|\,v_\theta(x_t, t, c) - (x_1 - \varepsilon)\,\big\|^2\,\Big]
$$

速度本身就有明确的几何意义（路径切向量），不需要再做仿射变换，所有 $t$ 处的量纲都是一致的。

**两者的等价关系**：在 OT-CFM 框架下，FM 速度和 DP score 函数之间存在如下关系（高级一点的内容）：

$$
v_\theta(x_t, t) = x_1 - \varepsilon = -\sigma_t^2 \cdot \nabla_{x_t} \log p_t(x_t) + \text{drift term}
$$

也就是说**两者在数学上是可互相转换的**，但 FM 选择了一个**几何上更对称的表征**，所以训练更稳、推理更快。

## 四、采样过程的对比

**DP 的 reverse process（DDPM）：**

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

可以看到每一步要做 $\sqrt{\alpha_t}$、$\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}$ 这些**系数换算**，本质上是在弯曲路径上做切线近似（一阶 Taylor），所以曲率大的地方步长必须小，需要 50~100 步。

**FM 的 ODE 积分（欧拉法）：**

$$
x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c)
$$

干净的一阶欧拉，没有任何额外的系数变换。在直线路径上，欧拉法即使用很大的步长 $\Delta t$ 也能很好地逼近。

## 五、所以更准确的说法是

**DP 和 FM 都有"加噪 → 预测某个量 → 迭代采样"的三段式结构，但三段都不完全一样**：

1. **加噪（前半部分）**：
   - DP 用**弯曲非线性**路径：$x_t = \sqrt{\bar\alpha_t}\, x_1 + \sqrt{1-\bar\alpha_t}\, \varepsilon$
   - FM 用**直线**路径：$x_t = (1-t)\,\varepsilon + t\,x_1$

2. **监督目标（核心区别）**：
   - DP 预测 $\varepsilon$（间接量）
   - FM 直接预测 $v = x_1 - \varepsilon$（几何切向量）

3. **采样（后半部分）**：
   - DP：reverse SDE/ODE，需要 schedule 系数变换
   - FM：直接欧拉积分 $x \mathrel{+}= \Delta t \cdot v$

**三段是连成一体的**：因为 FM 选了"直线路径"，所以 $\frac{dx_t}{dt} = x_1 - \varepsilon$ 是常数，所以监督目标可以直接是这个常数速度，所以推理可以简单到只用欧拉法。**路径选择决定了后续一切**。

## 六、一句话总结

> **FM 把整个 noise → action 的变换设计成一条直线 $x_t = (1-t)\,\varepsilon + t\,x_1$，所以前半部分（加噪轨迹形状）和后半部分（监督目标 $v = x_1 - \varepsilon$ + 采样方式 $x \mathrel{+}= \Delta t \cdot v$）都跟 DP 不一样。DP 的弯曲路径 $x_t = \sqrt{\bar\alpha_t}\, x_1 + \sqrt{1-\bar\alpha_t}\, \varepsilon$ 是历史包袱（从 score-based / SDE 框架继承下来的），FM 用 ODE 视角重写了整套流程，得到一个更简洁、推理更快的方案。**

你脑子里把 FM 想成"DP 的进化版"是没错的，但本质上 FM 不是改了一个 loss 函数那么简单，而是**把整条变换路径都拉直了**——这才是它能 4 步推理还效果不差的根本原因。