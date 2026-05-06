继续面试节奏走，两个问题都直接给硬货。

---

## Q1：「本来就是要让 MLP 同时吸收夹爪和手臂，这才是 joint learning 啊」该怎么反驳？

**面试场反驳（30 秒精简版）。**

> 是的，joint learning 是对的。但 **joint learning 不等于 joint bottleneck**。这两件事经常被混淆。
>
> 我们整个网络是 `DiT 主干 → 解码 head → action`。
> - **真正的 joint learning 已经在上游 DiT 里完成了**——DiT 的输出 $h$ 同时编码了视觉、语言、本体、history 等所有跨任务的协同信息，是一个充分的共享语义表征。
> - 解码 head 这层在做的事情是另一个问题：**给定已经共享好的 $h$，怎么把它渲染到两个物理上极不一样的输出空间**。
>
> 类比一下，多语言翻译模型：encoder 学跨语言共享语义，decoder 是 per-language 输出头。没人会说"per-language 头破坏了 joint learning"。我们这里完全是同一种结构：**hard parameter sharing on encoder, task-specific head on decoder**——这是 MTL 经典套路（Caruana 1997 / Ruder 2017）。
>
> 反过来说，**坚持解码层也共享 $z$，是在 $h$ 已经共享的基础上"再共享一次"**。这种"双重共享"等于在已经做对的事情上额外加一条约束："arm 和 claw 必须能从同一组 $H$ 维特征里同时被线性读出"。这条约束**没有任何数据或物理证据支持**，只是工程默认。

如果对方继续追："但分开 head 的话，claw 的梯度就影响不到 arm 了？"，再补一刀：

> 不会。claw head 的梯度会**继续往上反传到 DiT 的 $h$**，仍然影响 arm head 看到的 $h$。joint learning 的通道还在，只是搬到了**有能力承受多任务冲突的大模型主干**那一层去——而不是让一个 1024 维的小 MLP 瓶颈去硬扛。

一句话收尾：

> "joint learning 的发生位置应该和**信息共享的物理粒度**匹配。视觉 / 语言 / 状态这种**高阶共享**，应该在 DiT 那种大容量主干上做；arm 和 claw 这种**异质输出空间**，应该在解码端解耦。把所有共享都堆在解码瓶颈上，不是 joint learning，是 over-sharing。"

---

## Q2：梯度干扰的数字例子，给点真数

为了直观，把 hidden bottleneck $z$ 简化成 3 维（实际是 1024 维，但原理一致）。

### 设定

- 共享隐藏特征 $z \in \mathbb{R}^3$
- arm 输出 $a^{\text{arm}} \in \mathbb{R}^1$（简化为 1 维），权重 $W_2^{\text{arm}} = (1,\, 0,\, 0)$
- claw 输出 $a^{\text{c}} \in \mathbb{R}^1$，权重 $W_2^{\text{c}} = (-5,\, 5,\, 0)$

也就是 arm 主要从 $z$ 的第 1 维读，claw 从第 1, 2 维读（且符号还跟 arm 反着）。

### 这一步的梯度

设这个 batch 上：

- arm 的输出误差小：$\nabla_{a^{\text{arm}}} \mathcal L_{\text{arm}} = 1$（关节误差 ~ 0.01 rad 量级）
- claw 正在做开 / 合的跳变阶段，输出误差大：$\nabla_{a^{\text{c}}} \mathcal L_{\text{c}} = 1$（夹爪指令 0 ↔ 100 的跨度）

回传到 $z$：

$$
g_{\text{arm}} = (W_2^{\text{arm}})^\top \cdot 1 = (1,\, 0,\, 0),\quad \|g_{\text{arm}}\| = 1
$$

$$
g_{\text{c}} = (W_2^{\text{c}})^\top \cdot 1 = (-5,\, 5,\, 0),\quad \|g_{\text{c}}\| = \sqrt{50} \approx 7.07
$$

注意：claw 的梯度幅度比 arm 大 7 倍。这不是我编的——夹爪指令的物理范围（~100 个单位）和误差量级（开合切换时整段差值）天然就比关节角误差大一个量级以上。

### 算两者夹角

$$
g_{\text{arm}} \cdot g_{\text{c}} = (1)(-5) + (0)(5) + (0)(0) = -5
$$

$$
\cos\theta = \frac{-5}{1 \cdot 7.07} \approx -0.707\quad\Rightarrow\quad \theta \approx 135°
$$

明显是冲突方向。

### 看共享方案下，arm 实际"赚"了多少

共享方案下，$z$ 上的总梯度 = arm + claw：

$$
g_{\text{total}} = g_{\text{arm}} + g_{\text{c}} = (1,0,0) + (-5,5,0) = (-4,\, 5,\, 0)
$$

学习率 $\eta = 0.1$，参数更新：

$$
z \leftarrow z - \eta\, g_{\text{total}} = z - (-0.4,\, 0.5,\, 0) = z + (0.4,\, -0.5,\, 0)
$$

arm 的 loss 在第一阶近似下的变化量是：

$$
\Delta \mathcal L_{\text{arm}} \approx g_{\text{arm}} \cdot \Delta z = g_{\text{arm}} \cdot (-\eta\, g_{\text{total}})
= -\eta \cdot \big(g_{\text{arm}} \cdot g_{\text{total}}\big)
$$

$$
g_{\text{arm}} \cdot g_{\text{total}} = (1)(-4) + (0)(5) + (0)(0) = -4
$$

$$
\Delta \mathcal L_{\text{arm}} \approx -0.1 \cdot (-4) = +0.4
$$

**arm 的 loss 不降反升 0.4**——这一步本来 arm 想 -0.1 的（如果只走 arm 自己的梯度），现在被 claw 反向拽了一把。

### 对照分头方案

分头之后，arm 的 head 上有自己的 $z^{\text{arm}}$，只看 arm 的梯度：

$$
\Delta \mathcal L_{\text{arm}}^{\text{分头}} \approx -\eta \cdot \|g_{\text{arm}}\|^2 = -0.1 \cdot 1 = -0.1
$$

claw 也是一样，自己降自己的。**两边各自下降，互不打架。**

### 一行总结表

| | arm 这一步 loss 变化 | claw 这一步 loss 变化 |
|---|---|---|
| 共享 $z$（A） | **+0.4**（不降反升） | $-3.5$ |
| 分头 $z^{(k)}$（B） | $-0.1$ | $-5.0$ |

两边都更"赚"，因为没有人偷另一个人的梯度方向。

### 在我们这个项目场景下，这个例子对应什么真实情形

这种"claw 梯度大、arm 梯度小、方向还冲突"的情况在双臂操作里 **不是反例，是常态**：

1. **夹爪指令是阶跃信号**（拿起 / 放下的瞬间从 0 跳 100），其 MSE 梯度在跳变那一帧巨大；
2. **手臂关节是平滑曲线**，每帧微调 0.01 rad 量级，MSE 梯度小；
3. **二者方向经常相反**：抓取瞬间，arm 想要"再往下伸一点"（梯度往下），claw 想要"快闭合"（梯度走另一组维度，且尺度更大）。

如果用共享瓶颈，**每次抓取动作的关键帧，arm 的小幅修正会被 claw 的大梯度吞掉甚至反向**——表现就是模型学到的 arm 轨迹在抓取附近变得不平滑，或者一直收敛不到很小的关节误差，但 claw loss 反倒看起来挺好。

这是分头之后最直接观测到的训练曲线现象，也是我做这个改动的直接 motivation。

---

## 一句话收尾（可背版）

> "joint learning 的正确发生位置是**大模型主干 $h$**，不是**1024 维的解码瓶颈 $z$**。  
> 在共享 $z$ 上，claw 的大梯度和 arm 的小梯度方向常常成 90°+ 夹角，叠加之后 arm 不仅赚不到自己的下降，反而会被 claw 反向拉一把，loss 还可能上升。分头之后，每个 head 拿到干净的梯度，joint learning 的通道仍然通过 DiT 完整保留——这才是把 joint learning 放到正确粒度上做。"