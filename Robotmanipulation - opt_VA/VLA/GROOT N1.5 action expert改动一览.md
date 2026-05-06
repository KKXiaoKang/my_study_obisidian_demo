下面给你三段可以直接放进 论文 / 技术报告 / 答辩 PPT「方法贡献」章节的学术化描述。  
所有表述都和你 `flow_matching_action_head.py` 里的实现一一对应，**没有任何夸大**——主要替换了术语、补齐了 motivation 与对应文献。

---

### 1) Task-decomposed Multi-Head Action Decoder（按运动学语义解耦的多头动作解码器）

**问题动机.** 在 GR00T-N1.5 的原始 action expert 中，DiT 主干预测的 token 表征经由单一线性 / MLP 投影一次性回归整段动作向量 $a \in \mathbb{R}^{D}$。然而在双臂人形操作（bimanual humanoid manipulation）中，动作向量天然由若干**异质子空间**构成：左臂关节、右臂关节与夹爪。三者在 (i) 物理量纲（rad 与 0–100 的位移指令）、(ii) 输出动态范围、(iii) 噪声水平、(iv) 任务相关性 上差异显著，使用单一 head 进行联合回归会引入跨子空间的梯度干扰（gradient interference / negative transfer）。

Q：为什么会造成梯度干扰？给出详细的例子说明
A：

**方法.** 我们将动作头改造为典型的「**hard parameter sharing**」多任务学习架构（Caruana, 1997；Ruder, 2017）：DiT 提取的共享表征 $h_{1:T} \in \mathbb{R}^{T \times d}$ 被复用为多个**任务特定解码头**（task-specific decoding head）的输入，每个 head 是一个 embodiment-conditioned 的 `CategorySpecificMLP`，分别输出三个互不重叠的子流形上的 flow-matching 速度场：

$$
\hat{v}^{(\mathrm{LA})}_t = f_{\theta_{LA}}(h_t),\quad
\hat{v}^{(\mathrm{RA})}_t = f_{\theta_{RA}}(h_t),\quad
\hat{v}^{(\mathrm{C})}_t  = f_{\theta_{C}}(h_t),
$$

并在监督端将 ground-truth velocity、`action_mask` 也一并按子空间索引切片，独立计算 masked MSE：

$$
\mathcal{L}_k = \frac{\sum_{b,t,c} M^{(k)}_{b,t,c}\,\bigl\|\hat{v}^{(k)}_{b,t,c}-v^{(k)}_{b,t,c}\bigr\|^2}{\sum M^{(k)}+\varepsilon},\quad k\in\{\mathrm{LA},\mathrm{RA},\mathrm{C}\}.
$$

**带来的收益.**  
(1) 解耦了异质子空间的梯度路径，缓解 negative transfer；  
(2) 为后续两项贡献（不确定性加权、跨臂注意力）提供了天然的 head 级接口；  
(3) 统一支持单臂 / 双臂、关节空间 / EEF 空间四种配置，且各 head 维度、激活范围可分别约束。

---

### 2) Homoscedastic Uncertainty-Weighted Multi-Task Loss（基于同方差不确定性的可学习多任务损失加权）

**问题动机.** 当 multi-head 解耦后，损失合成 $\mathcal{L} = \sum_k w_k \mathcal{L}_k$ 中的 $w_k$ 通常需人为调参。由于左臂、右臂与夹爪三个子任务的**任务噪声尺度不同、难度不同**，固定权重在不同数据集与不同动作空间（Joint / Absolute eef / Delta eef）下都需要重新搜索，极不稳健。

**方法.** 我们采用 **Kendall, Gal & Cipolla（CVPR 2018, *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*, arXiv:1705.07115）** 提出的同方差任务不确定性（homoscedastic task-dependent uncertainty）加权：将每个回归头视为同方差高斯似然 $p(y_k \mid f_{\theta_k}(x)) = \mathcal{N}(f_{\theta_k}(x),\sigma_k^2)$，在负对数似然推导后得到带可学习权重的复合损失：

$$
\mathcal{L} = \sum_{k}\Bigl[\frac{1}{2\sigma_k^2}\mathcal{L}_k + \log\sigma_k\Bigr].
$$

为数值稳定，我们将其参数化为 $s_k := \log\sigma_k$（即 `task_log_sigma`，作为 `nn.Parameter`），在实现中等价为：

$$
\mathcal{L} = \sum_k \exp(-2 s_k)\,\mathcal{L}_k + \sum_k s_k.
$$

其中 $\exp(-2 s_k)$ 自动充当 task-precision 权重；$\sum_k s_k$ 起到 anti-degeneracy 正则项的作用，防止 $\sigma_k \to \infty$ 平凡解。整个加权过程**无需手工超参**，并随训练自适应调整。

**带来的收益.**  
(1) 把多任务权重的超参搜索空间从 $K-1$ 维压缩到 0 维；  
(2) 在 left/right/claw 三头任务噪声差异极大的双臂场景下，自动赋予高难度子任务更高 precision；  
(3) 与 component-wise `action_mask` 兼容，对部分缺失的子空间样本仍可稳健加权。

---

### 3) Bimanual Cross-Attention with Geometry-Preserving Decoupling（带几何约束保护的双臂跨注意力解码器）

**问题动机.** 单纯的 hard parameter sharing 只在共享主干上学习了**隐式**的双臂耦合。在协同性强的 bimanual 任务（如双手扶箱、对接、协同搬运）中，左右臂之间的几何/时间一致性对成功率至关重要，但解码端缺乏**显式**的左右信息流，模型只能依赖 DiT 内部隐式协同，样本效率较低。

**方法.** 我们在多头解码端引入**对称双向跨臂注意力**（symmetric bimanual cross-attention），构成「shared-bottom MLP → cross-attention → separate output」的混合解码结构（`SharedBottomArmDecoder`）：

1. **共享底层映射.** 共享 `CategorySpecificLinear` 将 $h_t$ 映射到共享 hidden 空间，再 LayerNorm 后分流为 left/right query：
   $$z^L_t = \mathrm{LN}(\phi(h_t)),\quad z^R_t = \mathrm{LN}(\phi(h_t)).$$
2. **对称跨臂注意力.** 用两组 4-head `MultiheadAttention` 形成左右**对偶**注意力，使左臂 query 关注右臂 key/value，反之亦然，并加残差连接：
   $$\tilde z^L_t = z^L_t + \mathrm{Attn}_L(z^L_t,\,z^R_t,\,z^R_t),\qquad
     \tilde z^R_t = z^R_t + \mathrm{Attn}_R(z^R_t,\,z^L_t,\,z^L_t).$$
3. **几何感知的输出解耦.** 当动作空间为 9D EEF（3D translation + 6D 旋转表示，Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019）时，**仅对位置子向量做跨臂注意力**，旋转 6D 子向量**不参与跨注意力，由独立线性头解码**：
   $$\hat p^L = W^L_p \tilde z^L_t,\;\hat p^R = W^R_p \tilde z^R_t,\quad \hat r^L = W^L_r z^L_t,\;\hat r^R = W^R_r z^R_t,$$
   $$\hat a^{L/R} = [\hat p^{L/R};\, \hat r^{L/R}].$$
   该设计的目的是**保留 6D 旋转表示的正交性 / 流形约束**，避免 attention 引入的线性混合破坏其几何不变量；这是 6D 连续旋转表示在 ML 中已知的表示几何约束（Zhou et al., 2019）。
4. **协调性辅助损失.** 进一步引入 magnitude-level 的双臂协调正则：
   $$\mathcal{L}_{\mathrm{coord}} = \mathbb{E}_{b,t}\bigl(\|\hat v^L_t\|_2 - \|\hat v^R_t\|_2\bigr)^2,$$
   并以可调权重 $\lambda_{\mathrm{coord}}$ 与不确定性加权后的主损失合成。该正则**只约束速度幅值**，而非方向或姿态，从而鼓励双臂同步运动节奏却不限制其各自的运动学自由度。

**带来的收益.**  
(1) 在 hard parameter sharing 之上引入显式的跨臂归纳偏置（inductive bias），改善 bimanual 协同任务的样本效率；  
(2) 通过「pos-only attention + rot-independent head」保护 6D 旋转的几何约束；  
(3) 协调正则提供任务级、方向无关的同步性 prior，与 head-wise uncertainty weighting 解耦设计无冲突，可即插即用。