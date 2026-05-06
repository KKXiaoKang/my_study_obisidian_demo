太好的两个问题了——一个是把梯度抵消的"具体诱因"系统列一遍，另一个是揭露 cross-attention **不能解决**的另一类失败模式（你已经在现场观察到了）。我把两件事分开讲清楚，并在最后给治法。

---

# Part 1：在我们这个 bimanual policy 里，**梯度抵消会发生在以下 6 种典型场景**

按"诱因"分类，每个场景都给"为什么会发生" + "在哪个变量上抵消"。

## 场景 1：左右臂物理动作方向相反

**例子**：拉箱子打开（左手向外拉、右手向外推），转身递物（左手前伸、右手后撤）。

**数学形式**：在动作空间里，$g^L \approx -\beta g^R$（输出空间梯度方向就反），通过共享投影后放大为：

$$
u^L \cdot u^R \;=\; (W_2^L)^\top g^L \cdot (W_2^R)^\top g^R \;\propto\; -\beta\,\|g^L\|\,\|g^R\|\,\cos\theta
$$

**抵消位置**：共享 $z$。

## 场景 2：claw 跳变 vs arm 平滑（**最常见、尺度差最大**）

**例子**：抓取关键帧。

- arm 梯度：$\|g^{\text{arm}}\| \sim 0.05$（关节角微调）
- claw 梯度：$\|g^{\text{c}}\| \sim 1.5$（0→1 跳变）

**数学形式**：尺度差 30 倍，$u^{\text{arm}}$ 完全被 $u^{\text{c}}$ 主导。

**抵消位置**：共享 $z$。

**特征**：在跳变帧附近 arm 的反传方向被 "claw 的尖峰"裹挟，arm 在该帧学不到正确修正。

## 场景 3：时序不同步（左右臂动作不在同一帧）

**例子**：左手 t=10 抓住，右手 t=20 才开始动。

**数学形式**：在 $t \in [0, 20]$ 区间内逐帧梯度：

$$
g^L_t = \mathbb 1[t < 10]\cdot g^L,\qquad g^R_t = \mathbb 1[t > 20]\cdot g^R
$$

某帧上**只有一只手有非零梯度**——但反传到共享 backbone 时，先前/未来帧的梯度也通过 chunk-level 聚合传回。**结果是 backbone 的某些通道被左手 dominated，另一些被右手 dominated**——模型被迫在共享通道上做 "时序复用"，效率低、易冲突。

**抵消位置**：DiT 的 future_token / hidden state。

## 场景 4：subtask 相位边界（**这是你后面那个问题的根源**）

**例子**：开箱 phase 和搬运 phase 的过渡帧。

- 开箱阶段：左右手向外
- 搬运阶段：左右手向同方向

**数学形式**：在 phase 1 中 $u^L_{\text{phase1}}$ 和 phase 2 中的 $u^L_{\text{phase2}}$ **方向矛盾**，模型如果用同一个共享 $z$ 编码两个 phase，必然在 $z$ 上发生**自我对抗**。

**抵消位置**：共享 $z$ 上的"phase-discriminative subspace"。

→ **这种抵消比左右手抵消还难处理**——它是**时间维度上的"自抵消"**，不是空间维度上的"对抗抵消"。

## 场景 5：表示容量瓶颈（hidden dim $H$ 太小）

**例子**：claw + 左 arm + 右 arm + cmd_pose 全挤进 $H=1024$。

**数学形式**：理想情况下不同任务应在 $z$ 中找正交子空间，但 $H$ 不够大时只能"挤同一组通道"。

**抵消位置**：共享 $z$（不可避免）。

## 场景 6：Loss 量级差（rotation vs position）

**例子**：6D rotation MSE ~ 0.5（角度差 0.7 rad），3D position MSE ~ 0.001（毫米级）。

**数学形式**：rotation 梯度比 position 大 500 倍，rotation 任务永久主导 backbone 更新方向。

**抵消位置**：DiT backbone。

→ **这正是为什么你做了 rotation/position 解耦输出层**：让两者至少在输出投影上不互相挤压（解决"输出层抵消"），代价是它们仍共享 $z_0$ → 共享 backbone 那层的抵消还在，但被压到了一个更小的范围。

---

## 场景小结：抵消位置一览

| 场景 | 抵消发生在 | 你的架构是否缓解？ |
|---|---|---|
| 1. 双臂方向反 | 输出层 + 共享 $z$ | ✅ multi-head 输出层解耦 + cross-attn 让模型可学控制信号 |
| 2. claw vs arm 尺度差 | 输出层 + 共享 $z$ | ✅ multi-head + uncertainty weighting |
| 3. 时序不同步 | DiT backbone | ❌ 没动它，只能靠 backbone 容量大 |
| 4. **Phase 边界自抵消** | 共享 $z$ + DiT backbone | ❌ **没有任何机制处理它** |
| 5. 容量瓶颈 | 共享 $z$ | △ 间接（multi-head 减轻了一点） |
| 6. rot vs pos 尺度差 | 共享 $z_0$ + 输出层 | ✅ rot/pos 输出层解耦 |

→ **场景 4（phase 边界）就是你下面那个问题的数学根源**。

---

# Part 2：你观察到的"坍塌为搬"——这是 **mode collapse to terminal goal**，不是梯度抵消

你描述的现象：

> 抓取任务：拿起 → 开箱 → 放进去 → 关箱 → **搬运**。  
> 模型训完后，**前面所有的精细任务都被"过滤"掉，直接输出搬运的动作**。

这是一个**完全不同于梯度抵消**的失败模式。让我把它正名并解释清楚。

## 2.1 这个现象的学术名

它不是 long-tail，不是 gradient interference，是：

> **Behavioral / Mode Collapse to Trajectory Endpoint**（轨迹终态坍塌）  
> 在长程多阶段任务的 imitation learning 中常见。也叫 "trajectory averaging" / "endpoint bias"。

## 2.2 为什么会发生？六个叠加的原因

### (a) Flow matching 的训练过程**天然偏向终态**

回顾代码 L924–940：

```python
noise = torch.randn(...)
t = sample_time(...)             # t ∈ [0, 1]
noisy_trajectory = (1-t)*noise + t*actions
velocity = actions - noise
```

- $t \sim \mathcal U[0,1]$ 均匀采样
- $t \approx 1$ 时 noisy_trajectory ≈ actions（接近真实终态）
- $t \approx 0$ 时 noisy_trajectory ≈ noise（远离动作）

**关键**：训练损失里 $t$ 越接近 1，noisy_trajectory 离真实 actions 越近，模型**实际只需要输出"接近 actions 本身"就能让 velocity 预测对**。这意味着训练信号**偏向"知道终态长什么样"**而非"知道每个过渡阶段长什么样"。

→ 模型的"最容易学"模式是**输出终态附近的速度场**——也就是搬运阶段。

### (b) 搬运阶段是整个任务里 **loss 最低、最稳定**的局部最优

| 阶段 | claw loss | arm loss | coordination loss | 总 loss |
|---|---|---|---|---|
| 抓取关键帧 | **大**（跳变） | 中（精细微调） | 中 | **高** |
| 开/关箱 | 中 | **大**（双手反向） | **大**（不同步） | **高** |
| 搬运 | 0（claw 不动） | 小（平滑） | 0（双手同向） | **低** |

→ **搬运是这个任务序列里 loss 最容易压低的阶段**。SGD 只要让模型在所有 $t$ 都输出搬运的速度场，就能拿到一个不错的 chunk-level 平均 loss。

### (c) Coordination loss 在搬运阶段**最容易满足**

回顾你代码 L1164–1167：

```python
left_arm_magnitude = torch.norm(pred_left_arm, dim=-1, keepdim=True)
right_arm_magnitude = torch.norm(pred_right_arm, dim=-1, keepdim=True)
coordination_loss = F.mse_loss(left_arm_magnitude, right_arm_magnitude, ...)
```

**搬运阶段 $\|\text{left}\| \approx \|\text{right}\|$（双手同向同速），coordination_loss = 0。**  
开箱阶段 $\|\text{left}\| \neq \|\text{right}\|$（双手反向），coordination_loss 大。

→ coordination loss **奖励了搬运、惩罚了开箱**——你的 inductive bias 直接把模型推向了坍塌！

### (d) Cross-attention 训练初期会"坍塌为均值 attention"

这是 transformer 训练的经典现象：softmax 在均匀初始化下，attention weight 接近 $1/T$ 均匀分布，输出变成 $\frac{1}{T}\sum_t V_t$ 的近似均值——**所有 phase 的特征被平均掉**。

**结果**：cross-attention 不再区分"现在是抓取还是搬运"——它把整个 chunk 看作"一种动作"，自然就坍塌到主导分量（搬运）。

### (e) Behavior cloning 的 distribution shift / 长程 trajectory 的 covariance shift

- 早期的小误差会沿着 chunk 累积；
- 模型在训练时没见过"开箱失败的状态"——一旦推理时左手没抓到，进入 OOD 状态，模型再也没法 recover；
- **最稳定的策略就是不去尝试有失败风险的子任务**——直接坍塌到"安全的搬运"。

### (f) GR00T-N1.5 预训练 prior 偏 smooth motion

GR00T 的预训练数据里，长程平滑搬运类动作占比高于精细 manipulation。微调时，**预训练 prior 把模型"拉回"它熟悉的搬运模式**——这是 LoRA / 全微调里都常见的现象。

## 2.3 为什么 cross-attention 解决不了这个问题？

**Cross-attention 解决的是"空间层面的双臂协同"，不是"时间层面的 phase 区分"**。

具体说：

| 问题 | Cross-attention 能解决吗？ |
|---|---|
| 左右手动作不协调 | ✅（这是它的目的） |
| 左手抓不开 | ✅（梯度抵消缓解） |
| **模型 skip 开箱直接搬** | ❌ **它没有 phase 区分能力** |
| **chunk 内 trajectory 坍塌** | ❌ **它在每个 t 都做同样的 cross-attn，不关心 t** |

数学上：cross-attention 是**任务空间内的耦合算子**，不是**时间维度上的 phase discriminator**。要解决 mode collapse，需要时间维度的归纳偏置。

---

# Part 3：怎么治"坍塌为搬"？候选解药

按"侵入性"从低到高排：

## 解药 1：把 coordination loss 改成 "phase-aware"

当前公式：

$$
\mathcal L_{\text{coord}} = \mathbb E\bigl(\|\hat v^L\| - \|\hat v^R\|\bigr)^2
$$

它在抓取/开箱阶段**反向激励了模型**。改成只在"高同步阶段"激活：

$$
\mathcal L_{\text{coord}} = m_t \cdot \bigl(\|\hat v^L\| - \|\hat v^R\|\bigr)^2,\quad m_t \in \{0,1\}
$$

其中 $m_t$ 由 phase label / sub-task ID 决定。**抓取/开箱阶段 $m_t=0$，搬运阶段 $m_t=1$**。

→ 立即停止用协调损失"奖励搬运"。

## 解药 2：降低或动态调度 `arm_coordination_loss_weight`

当前默认 0.2，对长程任务可能偏高。可以：

- 调到 0.05；
- 或者随训练步数 decay：早期高（学协同），后期低（让 phase 区分自然分化）。

## 解药 3：给 chunk 加上 phase / sub-goal 监督

在 chunk 上加一个**辅助 phase 分类头**：

$$
\mathcal L_{\text{phase}} = \text{CE}(\hat \phi_t,\, \phi_t^*)
$$

phase label 来自数据标注或自动分段。这样：
- $z_0$ 被迫保留 phase-discriminative subspace；
- attention 也被迫对不同 phase 给不同 weight。

## 解药 4：在 cross-attention 加 phase / time encoding

让 attention 能区分"现在第几帧、哪个 phase"。具体：

- 给 query/key 加 sinusoidal time encoding；
- 或者加 learned phase embedding（如果有 phase label）。

## 解药 5：训练时 chunk 切片要均衡 phase

现在如果数据里**搬运帧数远多于抓取帧数**（很常见，因为搬运耗时长），梯度自然偏搬运。建议 dataloader 上做 **per-phase resampling**——让抓取/开箱帧出现频率上调到与搬运齐平。

## 解药 6：Curriculum / staged fine-tuning

先用纯抓取/开箱数据微调（让模型**先学会**精细 manipulation），再混搬运数据微调。这样先验从 GR00T 的"smooth motion"逐渐过渡到"manipulation 能力"。

## 解药 7：Action chunk 拆分为 sub-chunks

不要一次预测从抓取到搬运的整个 100 帧 chunk。改成：每次只预测 10–20 帧的小 chunk，并条件化于"上一个 chunk 的终态"。这样**每个 chunk 内 phase 一致**，模型不容易在 chunk 内做 trajectory averaging。

---

## 一句话收尾

> "**梯度抵消和坍塌为搬是两类独立的失败**：前者是空间层面的 task 间反传冲突，由 multi-head + cross-attention 缓解；后者是时间层面的 phase mode collapse，由 flow-matching 的训练动力学 + coordination loss 偏好 + behavior cloning trajectory 累积偏差**共同**导致——cross-attention 在它面前是无力的。
>
> **后者真正的解药不是再加注意力，而是给模型施加 phase-discriminative inductive bias**：要么改 coordination loss 让它别再奖励搬运，要么加 phase embedding / phase 分类头让 attention 能区分子任务，要么在 dataloader 层做 phase resampling，要么用 curriculum 训练。"

---

如果你愿意，下一步我可以基于你的代码具体实现"解药 1（phase-aware coordination loss）"或者"解药 4（time encoding in cross-attention）"——这两个改动量最小、不改架构，直接动 loss 函数或 attention 输入即可。要选哪个？