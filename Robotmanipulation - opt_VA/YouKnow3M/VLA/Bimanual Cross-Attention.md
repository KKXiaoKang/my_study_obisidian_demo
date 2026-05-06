非常好的问题——你的措辞是“数论”，但你真正想要的是**矩阵分析 / 链式法则下的精确梯度推导**。我用最严格的形式来推一遍：从两条架构的损失定义出发，把"梯度抵消到底发生在哪里、为什么发生、cross-attention 又是怎么把它降下去的"完整写下来。

---

## 0. 共同 setup（两边对照）

为了让推导对齐，两边都用相同变量：

- 输入共享特征：$h \in \mathbb{R}^d$（来自 DiT 主干）
- 隐藏维度：$H$（代码里是 1024）
- 左/右臂动作维度：$D_L = D_R$（位置 3D 或全 9D）
- 损失：$\mathcal L = \mathcal L_L + \mathcal L_R$（先忽略 claw 简化论证）
- 学习率：$\eta$

记号：

- $g^L \;:=\; \nabla_{\hat a^L} \mathcal L_L \in \mathbb R^{D_L}$（L 损失对 L 输出的梯度）
- $g^R \;:=\; \nabla_{\hat a^R} \mathcal L_R \in \mathbb R^{D_R}$
- 反传到任意中间张量 $z$ 的梯度记为 $\nabla_z \mathcal L$

---

## 1. 方案 A：共享 2 层 MLP——**结构性梯度抵消，硬编码进了架构**

### 1.1 架构

$$
z \;=\; \mathrm{ReLU}(W_1 h + b_1),\qquad z \in \mathbb R^{H}
$$

$$
\hat a^L \;=\; W_2^L z + b_2^L,\qquad \hat a^R \;=\; W_2^R z + b_2^R
$$

注意 $W_2^L, W_2^R$ 都从**同一个 $z$ 投影出来**——这是 hard parameter sharing 的核心。

### 1.2 反传到共享瓶颈 $z$

由链式法则：

$$
\nabla_z \mathcal L
\;=\; \underbrace{(W_2^L)^\top g^L}_{u^L \in\mathbb R^H} \;+\; \underbrace{(W_2^R)^\top g^R}_{u^R \in\mathbb R^H}
\;=\; u^L + u^R
$$

**两条任务的反传信号在 $z$ 上做了向量加法。** 这个加法是**架构层面强制的**——没有任何机制可以避开它，只要还共享 $z$，就必加。

### 1.3 量化"L 任务这一步赚了多少"

把更新 $\Delta z = -\eta(u^L + u^R)$ 投影回 L 任务自己的下降方向（即 $u^L$ 方向）：

$$
\Delta \mathcal L_L^{\text{shared}}
\;\approx\; \langle u^L,\, \Delta z\rangle
\;=\; -\eta\,\|u^L\|^2 \;-\; \eta\,\langle u^L, u^R\rangle
$$

同时定义"理想情形"——如果 R 任务不存在，L 自己独占 $z$：

$$
\Delta \mathcal L_L^{\text{selfish}} \;=\; -\eta\,\|u^L\|^2
$$

**梯度干扰量**：

$$
\boxed{\;\delta_L \;:=\; \Delta\mathcal L_L^{\text{shared}} - \Delta\mathcal L_L^{\text{selfish}} \;=\; -\eta\,\langle u^L, u^R\rangle\;}
$$

### 1.4 抵消条件（你最关心的）

设 $u^R = -\alpha\,u^L$，即 $u^R$ 在 $u^L$ 反方向有分量 $\alpha\|u^L\|$：

$$
\Delta \mathcal L_L^{\text{shared}}
\;=\; -\eta\,\|u^L\|^2 \;+\; \eta\,\alpha\,\|u^L\|^2
\;=\; -\eta\,(1-\alpha)\,\|u^L\|^2
$$

| $\alpha$ | 含义 | 后果 |
|---|---|---|
| $\alpha = 0$ | 两梯度正交 | L 正常下降（理想） |
| $0 < \alpha < 1$ | 部分对冲 | L 下降但比理想慢 |
| $\alpha = 1$ | 完全抵消 | **L 这一步 0 进展** |
| $\alpha > 1$ | 反向支配 | **L 任务 loss 上升**（被 R 反向拽着走） |

→ **方案 A 的"梯度抵消"是一个数学上必然存在的结构现象**，不是优化器调出来的。$u^L, u^R$ 完全由 $W_2^L, W_2^R$ 的列空间几何决定，**和模型有没有学好无关**。

### 1.5 数值 demo（左手 vs 右手）

设 $H=3$（瓶颈 3 维方便手算），左手：

$$W_2^L=(1,\,0,\,0),\quad g^L = 1$$

右手（claw + 大梯度尺度）的反向冲突情形：

$$W_2^R=(-2,\,1,\,0),\quad g^R = 1$$

那么：

$$u^L = (1,\,0,\,0),\quad \|u^L\| = 1$$

$$u^R = (-2,\,1,\,0),\quad \|u^R\| = \sqrt{5}\approx 2.24$$

$$\langle u^L, u^R\rangle = -2$$

夹角：$\cos\theta = \frac{-2}{1\cdot\sqrt 5}\approx -0.89 \Rightarrow \theta\approx 153°$。

L 实际进展（学习率 $\eta=0.1$）：

$$
\Delta\mathcal L_L^{\text{shared}} = -0.1\cdot 1 - 0.1\cdot(-2) = -0.1 + 0.2 = +0.1
$$

**L 的损失这一步还涨了 0.1**——这就是你观察到的"左手抓不开"的最直接数学根源。

---

## 2. 方案 B：Cross-attention + 独立输出层——**梯度抵消变成可学习的、且自带 identity 保护**

### 2.1 架构（贴你代码 `SharedBottomArmDecoder`）

$$
z_0 \;=\; \mathrm{ReLU}(W_1 h + b_1) \quad\text{（共享底层）}
$$

$$
z^L = \mathrm{LN}_L(z_0),\qquad z^R = \mathrm{LN}_R(z_0)
$$

$$
\tilde z^L = z^L + A^L(z^L;\, z^R, z^R),\qquad \tilde z^R = z^R + A^R(z^R;\, z^L, z^L)
$$

$$
\hat p^L = W^L_p \tilde z^L,\qquad \hat p^R = W^R_p \tilde z^R
$$

注意三件大事：

1. **输出层 $W^L_p, W^R_p$ 完全独立**（参数不共享）；
2. **cross-attention 模块 $A^L, A^R$ 也完全独立**（参数不共享）；
3. **残差连接**：$\tilde z^L = z^L + A^L(...)$，意味着即使 $A^L$ 是 0，L 的反传链路也仍然完整。

### 2.2 第 1 个梯度路径：在 $\tilde z^L$（最靠近输出的中间张量）

$$
\nabla_{\tilde z^L} \mathcal L \;=\; \underbrace{(W^L_p)^\top g^L}_{\text{来自 L 任务}} \;+\; \underbrace{0}_{\text{来自 R 任务}}
$$

**注意：这里 R 的梯度是 0**——因为 $\hat p^R$ 通过 $W^R_p$ 而不是 $W^L_p$ 算出，跟 $\tilde z^L$ 没有依赖关系。

→ **$\tilde z^L$ 处 0 干扰**。这是方案 A 完全没有的东西。

### 2.3 第 2 个梯度路径：从 $\tilde z^L$ 反传到 $z^L$

$$
\frac{\partial \tilde z^L}{\partial z^L} \;=\; I \;+\; \underbrace{J^Q_L}_{\text{Attn}_L\text{ 对其 query 的 Jacobian}}
$$

所以：

$$
\nabla_{z^L} \mathcal L_L \;=\; (I + J^Q_L)^\top \cdot (W^L_p)^\top g^L
$$

**关键**：里面有个 $I$ 是常数恒等映射——这意味着无论 attention 学成什么样，**L 的梯度信号始终能"原汁原味地走 identity 这条路"传回到 $z^L$**。

### 2.4 第 3 个梯度路径：R 通过 cross-attention 也能流到 $z^L$

$\tilde z^R$ 的计算里用了 $z^L$ 作 K, V：

$$
\nabla_{z^L} \mathcal L_R \;=\; (J^{KV}_R)^\top \cdot (W^R_p)^\top g^R
$$

其中 $J^{KV}_R = \partial A^R(z^R;\, z^L, z^L) / \partial z^L$ 是 attention 对其 K,V 输入的 Jacobian。

总梯度：

$$
\boxed{\;\nabla_{z^L}\mathcal L \;=\; \underbrace{(I + J^Q_L)^\top (W^L_p)^\top g^L}_{\text{自路径，含 identity 保护}} \;+\; \underbrace{(J^{KV}_R)^\top (W^R_p)^\top g^R}_{\text{跨路径，全程经 attention 调制}}\;}
$$

### 2.5 为什么这条公式比方案 A 强？三点

**(a) 自路径有 identity 保证下界。**

设 R 任务发疯，$g^R$ 巨大且方向冲突。即便如此，自路径含 $(I + J^Q_L)$。当 $J^Q_L$ 不"作恶"（注意力没把自己抹掉），$I$ 这一项保证了 L 至少能以**接近完整强度**反传：

$$
(I + J^Q_L)^\top (W^L_p)^\top g^L \;\geq\; (W^L_p)^\top g^L \quad\text{（在 } J^Q_L\succeq 0 \text{ 假设下）}
$$

方案 A 没有这个**结构性下界**——$u^L$ 完全可能被 $u^R$ 全反向抵消。

**(b) 跨路径是数据相关、可学习的"门"。**

$J^{KV}_R$ 是 softmax-attention 关于 K, V 的 Jacobian，**它的强度由 attention weight 控制**。如果模型发现 R 的 cross-talk 信号有害，**它可以学到把对应 attention weight 压低**，从而让这一项的范数变小。换言之：

> **方案 A 的梯度耦合是 hard-coded（由 $W_2$ 的列空间决定，无法关闭）；方案 B 的梯度耦合是 learnable（由 attention 的 softmax 加权决定，模型可以训出"几乎不耦合"）。**

**(c) 输出层完全 0 干扰。**

$W^L_p$ 只接 L 损失梯度，$W^R_p$ 只接 R 损失梯度——这俩矩阵的更新**互相不影响**。这是方案 A 做不到的（方案 A 里 $W_2$ 的不同行属于同一个矩阵，更新时虽然按 row 分开但因为前向时共用同一个 $z$，参数是耦合的）。

### 2.6 数值 demo（同样的设定，对比方案 A）

为了对比，假设同样的 L 任务梯度强度 $\|g^L\|=1$、同样的 R 任务方向冲突 $\|g^R\|=\sqrt 5$。

方案 B 里 $W^L_p$ 与 $W^R_p$ 解耦，所以输出层那一层 0 干扰。

考虑 attention 训练初期（attention weight 接近均匀，等价于 $J^{KV}_R$ 是个小尺度矩阵，假设 spectral norm $\|J^{KV}_R\|_2\approx 0.3$）：

L 在 $z^L$ 处的总梯度范数：

- 自路径贡献（identity 主导）：$\approx \|g^L\|=1$
- 跨路径贡献（被 attention 衰减）：$\leq 0.3 \cdot \|g^R\| \approx 0.67$

最坏情况下两者夹角 180°：

$$\Delta\mathcal L_L^{\text{cross}} \approx -\eta \cdot 1 - \eta\cdot(\text{cross-term}\le 0.67)$$

但因为 cross-term 是**可学习的 0.3**，模型很快会让它降到 0.05 量级，从而 L 的实际进展接近：

$$
\Delta\mathcal L_L^{\text{cross}} \approx -\eta \cdot 1 - \eta \cdot 0.05 = -0.105
$$

vs 方案 A 是 $+0.1$（loss 反涨）。**两个数量级的差异**。

---

## 3. 旋转子空间为什么不做 cross-attention？数学上看

代码里 rotation 子空间走的是 `shared_features` 直通，不经 attention：

$$
\hat r^L = W^L_r z_0,\quad \hat r^R = W^R_r z_0
$$

也就是 rotation 路径**结构上等同于"split MLP with shared bottom $z_0$"**——梯度在 $z_0$ 处仍然存在 hard parameter sharing 类型的耦合，**但耦合方向被 $W^L_r, W^R_r$ 解耦**（输出层独立）。

为什么不也做 attention？因为 6D 旋转表示是 **$\mathbb R^6$ 中嵌入的 SO(3) 流形**，需要正交化才能解出旋转矩阵（Zhou et al., 2019）。Attention 是个**线性混合**算子（softmax 加权求和），它会破坏 6D 子向量的正交关系——这是几何损失，不是梯度问题。

→ **Rotation 用回 hard sharing 的"梯度小代价"，换取几何稳定性**。这是一个有意识的 trade-off。

---

## 4. 全场对比表

| | 方案 A（共享 MLP） | 方案 B（cross-attn + 独立输出） |
|---|---|---|
| 输出层耦合 | 共用 $z$，结构性耦合 | $W^L_p, W^R_p$ 完全独立，**0 干扰** |
| 梯度耦合 | $(W_2^L)^\top g^L + (W_2^R)^\top g^R$，**hard-coded** | $(I+J^Q_L)^\top g^L + (J^{KV}_R)^\top g^R$，**learnable + identity 保护** |
| 自路径强度下界 | 无下界，可被完全抵消 | 含 $I$，**保证 L 始终有完整反传** |
| 跨路径可关 | 不可关 | 可学到 attention weight ≈ 0 → 跨耦合 ≈ 0 |
| 抵消上界 | $\delta_L = -\eta\langle u^L, u^R\rangle$ 无界 | $\delta_L \le -\eta\|J^{KV}_R\|\|W^R_p\|\|g^R\|$，受 attention 调控 |
| 极端"抵消"是否会让 L 发散 | 会（$\alpha > 1$ 时 L loss 上升） | 不会（attention 学会压跨路径） |
| 旋转几何约束 | 与位置一起被混合，**破坏正交性** | 旋转独立路径，**保留正交** |

---

## 5. 一句话收尾（可背版）

> "**方案 A 的梯度抵消是结构性的——它由 $W_2^L, W_2^R$ 的列空间几何决定，是个 hard-coded 现象，没法关。** 数学上具体是 $\delta_L = -\eta\langle u^L, u^R\rangle$，当 $u^R$ 在 $u^L$ 反方向占优时（你观测到的左手抓不开正属于这种），L 的 loss 实际可能不降反升。
>
> **方案 B 把抵消改成了软性、可学习的：** 输出层完全解耦保证最直接的 0 干扰；中间层有 identity 残差保证自路径下界；跨路径全程经 attention 调制，模型可以自己学会把跨耦合压到接近 0。**最坏情况下方案 B 退化为'split MLP with shared bottom'，最好情况下显式地做了双臂协同**——单调优于方案 A。
>
> **但方案 B 仍非完美**：rotation 子空间为了保 6D 表示的几何正交性退回到 hard-shared bottom，那里仍有梯度耦合，只是被独立输出层 $W^L_r, W^R_r$ 解耦了一半——这是几何稳定性 vs 梯度独立性的工程 trade-off。"

---

如果你下一步还想再深一层，可以追问："那 attention 的 softmax weight 在多任务训练下会不会**稳定**？会不会因为它本身也有梯度，就被 R 任务'拐跑'？"——答案是 attention 模块的参数 $W_q, W_k, W_v$ 也是训练目标的一部分，它们的稳定性反而是这套机制能否生效的关键。这就引出一个新问题：**cross-attention 引入了多大量级的额外参数与训练负担？** 这是个工程类的下一步问题，可以单独分析。