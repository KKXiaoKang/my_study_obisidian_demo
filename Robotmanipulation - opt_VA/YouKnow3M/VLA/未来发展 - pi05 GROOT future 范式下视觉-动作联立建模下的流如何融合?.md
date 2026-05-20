你这个问题真是问到核心了——**动作必须和视觉联动才能判断"是否可行"**。这其实是 VLA 的灵魂所在。让我先把 π0.5 的真实 joint attention 看仔细，再讨论"动作流如何与多视觉流融合"的范式。

## 一、π0.5 的 Joint Attention 真实实现解构

之前看过 π0.5 代码，现在把它**逐层拆解**：

### 第 1 步：两个独立的 token 序列

```588:629:src/lerobot/policies/pi05/modeling_pi05.py
    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
```

```631:676:src/lerobot/policies/pi05/modeling_pi05.py
    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)
        ...
        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
        ...
        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))
```

**关键观察**：

- `prefix_embs` = `[img_tokens; lang_tokens]`，所有 `att_mask = 0`
- `suffix_embs` = `[action_tokens]`，`att_mask = [1, 0, 0, ..., 0]`（**第一个 1，后续 0**）

### 第 2 步：cumsum 构造 2D Mask（这是 prefix-LM 的精髓）

```93:120:src/lerobot/policies/pi05/modeling_pi05.py
def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
        each other and the last 3 tokens have a causal attention. The first
        entry could also be a 1 without changing behaviour.
    """
```

规则：**token $i$ 能 attend token $j$ iff $\text{cumsum}(i) \geq \text{cumsum}(j)$**

具体到 π0.5：
- `[prefix_att | suffix_att]` = `[0, 0, ..., 0 | 1, 0, 0, ..., 0]`
- `cumsum` = `[0, 0, ..., 0 | 1, 1, 1, ..., 1]`

得出 attention 矩阵：

$$
M_{ij} = \mathbb{1}[\text{cumsum}_i \geq \text{cumsum}_j]
$$

| | IMG | LANG | ACT |
|---|---|---|---|
| **IMG** (cumsum=0) | ✓ | ✓ | ✗ |
| **LANG** (cumsum=0) | ✓ | ✓ | ✗ |
| **ACT** (cumsum=1) | ✓ | ✓ | ✓ |

✗ 表示该位置被 mask 掉，✓ 表示可见。

### 第 3 步：每层做 MoT（Mixture-of-Transformers）Joint Attention

```219:280:src/lerobot/policies/pi05/modeling_pi05.py
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    ...
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    ...
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        ...
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        ...
        out_emb = layer.mlp(out_emb)
```

**π0.5 的 joint attention 数学描述**（每一层）：

$$
\begin{aligned}
& \text{VL 端用 PaliGemma 权重}: && Q^{vl}_i = W^Q_{\text{vl}} x^{vl}_i, \quad K^{vl}_i = W^K_{\text{vl}} x^{vl}_i, \quad V^{vl}_i = W^V_{\text{vl}} x^{vl}_i \\
& \text{Action 端用 Gemma Expert 权重}: && Q^{a}_i = W^Q_{\text{act}} x^{a}_i, \quad K^{a}_i = W^K_{\text{act}} x^{a}_i, \quad V^{a}_i = W^V_{\text{act}} x^{a}_i \\
& \text{在 sequence 维度拼接}: && Q = [Q^{vl}; Q^{a}], \quad K = [K^{vl}; K^{a}], \quad V = [V^{vl}; V^{a}] \\
& \text{一次 attention 跨流}: && O = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V \\
& \text{切回各自}: && O^{vl} = O_{:T_{vl}}, \quad O^{a} = O_{T_{vl}:} \\
& \text{各自走自己的 o\_proj 和 MLP}: && y^{vl} = \text{MLP}_{\text{vl}}(W^O_{\text{vl}} O^{vl} + x^{vl}), \quad y^a = \text{MLP}_{\text{act}}(W^O_{\text{act}} O^{a} + x^a)
\end{aligned}
$$

**π0.5 的核心特征**：

1. **两套独立权重**：VL 用 PaliGemma 全套（q/k/v/o/MLP/LN），Action 用 Gemma Expert 全套
2. **唯一耦合点是 attention**：只有 Q·K^T·V 这一步两边的 K/V 池共享
3. **head_dim=256 故意对齐**：让 K/V 能直接拼接
4. **prefix-LM mask**：VL 不看 action，action 看所有

**所以严格来说，π0.5 的 joint attention 是单向的（VL → action）**——VL 端的表征不会因 action 改变。但它比 GR00T 的 cross-attention 强在：**VL 的 K/V 没经过任何 "翻译投影"**（GR00T 每层要 to_k/to_v 重投影 2048→1536），保留了 PaliGemma 完整的预训练知识。

---

## 二、五流融合（4 World + 1 Action）的两种范式扩展

回到你的问题——现在有 5 种流，怎么做？我把 GR00T 和 π0.5 的范式分别推广出来：

### 范式 A：**GR00T 风格扩展**（Cross-Attention DiT，单点融合）

```text
img_tokens ──┐
dyn_tokens ──┤
3d_tokens  ──┼──► [World Fusion Block]──► world_features (融合后)
lang_tokens──┘    (内部 self-attn 让四流融合)         │
                                                     │ as K, V
                                                     │
                action_tokens ────► [DiT Block × N]──┴──► action_features
                                       (Q ← action)
                                       (K,V ← world)
                                    每层都 cross-attn
```

数学：

**Stage 1**（World 融合）：
$$
\tilde{w} = \text{SelfAttn}^{L_w}([T_{\text{img}}; T_{\text{dyn}}; T_{\text{3d}}; T_{\text{lang}}])
$$

**Stage 2**（DiT × N 层）：
$$
\begin{aligned}
a^{(l+1/2)} &= a^{(l)} + \text{CrossAttn}(Q = a^{(l)}, K = V = \tilde{w}) \\
a^{(l+1)} &= a^{(l+1/2)} + \text{SelfAttn}(a^{(l+1/2)}) \\
a^{(l+1)} &= a^{(l+1)} + \text{MLP}(a^{(l+1)})
\end{aligned}
$$

**特点**：
- World 一次性融合好，可以 cache
- Action 每层从 world "取"信息
- World 不被 action 影响
- **N 层 DiT 中，K/V 每次都从 world 重新投影**（这是 GR00T 当前的做法）

### 范式 B：**π0.5 风格扩展**（MoT Joint Attention，每流独立权重）

```text
                  [img_tokens]  [dyn_tokens]  [3d_tokens]  [lang_tokens]  [action_tokens]
                       │             │             │              │              │
                       │             │             │              │              │
   每层都有 ↓        ImgGemma     DynGemma      3dGemma       LangGemma      ActExpert
                       │             │             │              │              │
                  ┌────┴─────────────┴─────────────┴──────────────┴──────────────┴────┐
                  │  Joint Attention（在 sequence 维度拼接 K/V）                       │
                  │  Q_i = W^Q_{type(i)} x_i,  K_i = W^K_{type(i)} x_i, ...           │
                  │  Mask: world↔world 双向，world→action 单向                        │
                  └────┬─────────────┬─────────────┬──────────────┬──────────────┬────┘
                       │             │             │              │              │
                  各自 o_proj    各自 o_proj    各自 o_proj    各自 o_proj    各自 o_proj
                  各自 MLP       各自 MLP       各自 MLP       各自 MLP       各自 MLP
```

数学（每层）：

$$
\begin{aligned}
& \text{为每个 type 维护独立权重}: && (W^Q_k, W^K_k, W^V_k, W^O_k, W^{\text{MLP}}_k), \quad k \in \{\text{img, dyn, 3d, lang, act}\} \\
& \text{各自计算 Q/K/V}: && Q^k = W^Q_k x^k, \quad K^k = W^K_k x^k, \quad V^k = W^V_k x^k \\
& \text{拼接}: && Q = [Q^{\text{img}}; Q^{\text{dyn}}; Q^{\text{3d}}; Q^{\text{lang}}; Q^{\text{act}}] \\
& \text{Joint Attn}: && O = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M_{\text{prefix-LM}}\right) V \\
& \text{切回}: && O^k = O_{T_{<k} : T_{<k}+T_k} \\
& \text{各自处理}: && y^k = \text{MLP}_k(W^O_k O^k + x^k)
\end{aligned}
$$

mask 设计：

| | img | dyn | 3d | lang | **act** |
|---|---|---|---|---|---|
| img | ✓ | ✓ | ✓ | ✓ | ✗ |
| dyn | ✓ | ✓ | ✓ | ✓ | ✗ |
| 3d | ✓ | ✓ | ✓ | ✓ | ✗ |
| lang | ✓ | ✓ | ✓ | ✓ | ✗ |
| **act** | ✓ | ✓ | ✓ | ✓ | ✓ |

**特点**：
- 每个流有自己的"专家"权重（5 个 expert）
- 每层都有 K/V 池共享
- 推理时 world 部分可 KV cache（prefix-LM 的关键好处）
- World 之间是双向融合，action 单向取信息

### 两种范式对比

| 维度 | GR00T 风格 | π0.5 风格 |
|------|------------|------------|
| 流的处理 | World 先融合后给 action | 每流独立专家 + joint K/V |
| 权重共享 | World 共享 SelfAttn | 每个 type 独立 q/k/v/o/MLP |
| K/V cache | World 一次算好，后续都用 | World 部分能 cache（prefix-LM） |
| World 双向融合 | ✓ Stage 1 一次性 | ✓ 每层都做 |
| Action 信息流 | 仅在 DiT cross-attn | Joint K/V 池 |
| 每层信息更新 | 仅 action 更新 | 每层 world 也在更新 |
| 推理延迟 | 较高（DiT cross-attn × N） | 较低（KV cache） |
| 跨流学习能力 | 中等（cross-attn 单点） | 强（每层都互相 attend） |

---

## 三、我认为最好的"视觉-动作联立建模"方式

我想跳出 GR00T 和 π0.5 的框架，提出一个我认为更接近"真正联立建模"的方案。先讲清楚**为什么现有方案都不够好**。

### 现有方案的根本问题：**单向信息流忽略了"主动感知"**

你说的非常对：**动作必须和视觉联动判断"是否有效"**。但是无论 GR00T 还是 π0.5，**信息流都是单向的**：world → action，**action 永远不会反过来影响 world 的表征**。

这违反了人类感知的核心原则——**Active Perception**（主动感知）：

> 当我想抓杯子，我会**主动看杯子的把手**，而不是被动地全场景均匀感知。
> 我的运动意图反过来引导了我的视觉注意力。

这就是为什么 VLA 在"需要精细 grounding 的任务"上一直拉胯——它**永远是 passive perception**。

### 我提出的方案：**Three-Tier Hybrid Active VLA (APVLA)**

```text
═══════════════════════════════════════════════════════════════════════
                  TIER 1: World Stream Fusion (L₁ 层)
                  目的: 让四个 world 流互相融合形成 "world state"
═══════════════════════════════════════════════════════════════════════

   [img]  [dyn]  [3d]  [lang]    ← action NOT involved
       │     │     │      │
       └──┬──┴──┬──┴──┬───┘
          │     │     │
          ▼     ▼     ▼
     Bidirectional Self-Attention (MoT 风格，每流独立 q/k/v)
     Mask: world↔world 全连接（无 action mask）
          │
          ▼
     fused_world (可 cache，整段推理只算一次)

═══════════════════════════════════════════════════════════════════════
                  TIER 2: Action-World Cross-Attention (L₂ 层)
                  目的: 让 action 从 fused_world 高效抽取信息
═══════════════════════════════════════════════════════════════════════

     action_tokens
          │
          ▼
     SelfAttn(action) ─┐
                       ├─► action_layer_out
     CrossAttn(Q=action, K=V=fused_world) ─┘

     (每层 action 互相协调，再从 world 拉信息)

═══════════════════════════════════════════════════════════════════════
                  TIER 3: Active Perception Refinement (L₃ 层)
                  ⭐ 关键创新: 让 action 反向引导 world 的注意力
═══════════════════════════════════════════════════════════════════════

     Step A: action-conditioned gate
        g_w = σ(MLP([action_pool; world_token]))  ∈ [0,1]
        
     Step B: action → world cross-attention (selective)
        Δw = g_w ⊙ CrossAttn(Q=fused_world, K=V=action)
        fused_world ← fused_world + Δw  
        
     Step C: 重新让 action 拉一次（现在 world 已经知道 action 的意图）
        action ← action + CrossAttn(Q=action, K=V=fused_world)
        
     ⭐ 这一层是"主动感知"的核心
       - action 通过 gate 选择性影响 world  
       - 类似"想要抓杯子 → 重新关注杯子区域"
```

### 数学公式

**Tier 1：World Fusion**（$L_1$ 层）

$$
\tilde{w}^{(l+1)} = \text{MoTSelfAttn}\!\left(\tilde{w}^{(l)}, \text{Mask}_{\text{world-only}}\right), \quad l = 0, \ldots, L_1 - 1
$$

$$
\tilde{w} = [\,\tilde{w}_{\text{img}}; \tilde{w}_{\text{dyn}}; \tilde{w}_{\text{3d}}; \tilde{w}_{\text{lang}}\,]
$$

**Tier 2：Action Reads World**（$L_2$ 层）

$$
\begin{aligned}
a^{(l+1/2)} &= a^{(l)} + \text{SelfAttn}(a^{(l)}) \\
a^{(l+1)} &= a^{(l+1/2)} + \text{CrossAttn}(Q = a^{(l+1/2)}, K = V = \tilde{w}) \\
a^{(l+1)} &= a^{(l+1)} + \text{MLP}(a^{(l+1)})
\end{aligned}
$$

**Tier 3：Active Perception**（$L_3$ 层）——这才是关键

$$
\begin{aligned}
\bar{a} &= \text{Pool}(a) \in \mathbb{R}^d \quad \text{(action context)} \\
g_w &= \sigma\big(W_g \cdot [\bar{a} \,;\, \tilde{w}]\big) \in [0,1]^{T_w \times d} \\
\Delta \tilde{w} &= g_w \odot \text{CrossAttn}(Q = \tilde{w}, K = V = a) \\
\tilde{w}^{\text{active}} &= \tilde{w} + \Delta \tilde{w} \\
a^{\text{final}} &= a + \text{CrossAttn}(Q = a, K = V = \tilde{w}^{\text{active}})
\end{aligned}
$$

**关键的 gate $g_w$ 控制 action 对 world 的影响强度**：
- $g_w \to 0$：action 不影响 world（退化为 π0.5）
- $g_w \to 1$：action 完全主导 world 的更新（极端 active）
- 真实情况是**学习到的混合**

### 为什么这样设计？三个核心 insight

#### Insight 1：World 融合应该是 task-agnostic 的

Tier 1 没有 action 参与，让 world 表征**先形成一个稳定的"场景理解"**。这个理解可以跨任务复用、可以 cache。

#### Insight 2：Action 读取应该是 task-conditioned 的

Tier 2 让 action 根据当前需求去 world 里抓信息——这是 π0.5 已经做的。

#### Insight 3：Active Perception 闭环

Tier 3 加入了 action → world 的反馈通道，**但通过 gate 控制强度**。这模仿了人脑的 top-down attention：

> "看到杯子（passive）→ 我想抓它（intention）→ **重点看杯子的把手**（active refinement）→ 输出更精准的抓取动作"

**这种闭环结构在 flow matching 多步迭代中天然有意义**：

```python
# Flow matching 推理
x_t = noise
for t in range(num_steps):
    # 第一次迭代：粗略的 action 意图
    # 第二次迭代：基于上一步意图，更精细地"看"目标区域
    # 第三次迭代：进一步收紧
    v_t = model(world_tokens, x_t, t)
    x_t = x_t + dt * v_t
```

**每一步 flow matching iteration 都让 world 和 action 更紧密地协同**。这就是为什么 Tier 3 应该用 gate—— gate 让模型自己决定每次迭代要多深入地"主动看一眼"。

### 代码骨架（落地角度）

```python
class APVLA(nn.Module):
    def __init__(self):
        # Tier 1: World MoT (类似 π0.5，但只有 world 流)
        self.world_streams = nn.ModuleList([
            GemmaModel(config_img),     # 图像流
            GemmaModel(config_dyn),     # 动力学流
            GemmaModel(config_3d),      # 3D 流
            GemmaModel(config_lang),    # 语言流
        ])
        self.world_layers = L1  # 6 层
        
        # Tier 2: Action Reader (action 单向读 world)
        self.action_reader = GemmaModel(config_act)
        self.reader_layers = L2  # 8 层
        
        # Tier 3: Active Perception
        self.active_gate = nn.Linear(2 * d_model, d_model)
        self.active_layers = L3  # 4 层
    
    def forward(self, img, dyn, point, lang, noisy_action, time):
        # === Tier 1: World Fusion (action 不参与) ===
        world = mot_joint_attn(
            [img, dyn, point, lang],
            mask=full_bidirectional,
            num_layers=self.world_layers
        )
        # world 可以 cache (跨 flow matching 多步)
        
        # === Tier 2: Action Reads World ===
        action = embed_action(noisy_action, time)
        for layer in self.reader_layers:
            action = self_attn(action) + cross_attn(Q=action, K=V=world)
        
        # === Tier 3: Active Perception (核心) ===
        for layer in self.active_layers:
            # action 反向改 world
            action_pool = action.mean(dim=1)  # (B, d)
            gate = sigmoid(self.active_gate(cat([action_pool[:, None], world], dim=-1)))
            world_delta = gate * cross_attn(Q=world, K=V=action)
            world = world + world_delta  # 注意: world 只在这一阶段被改
            
            # action 再读一次更新后的 world
            action = action + cross_attn(Q=action, K=V=world)
            action = action + self_attn(action)
        
        v_t = self.action_out_proj(action)
        return v_t
```

### 关键创新点总结

| 维度 | GR00T | π0.5 | **APVLA（我的方案）** |
|------|-------|------|---------------------|
| World 流融合 | 串行 SelfAttn | MoT Joint Attn | **MoT + 多层深度融合** |
| Action 读 World | Cross-Attn × N | Joint Attn (prefix-LM) | **专属 Cross-Attn 层** |
| Action 影响 World | ❌ | ❌ | **✅ Gated Cross-Attn** |
| Active Perception | 无 | 无 | **✅ 任务意图引导感知** |
| Flow Matching 配合 | 每步重算 cross-attn | 每步用同样的 world cache | **每步逐渐 active refine** |
| KV Cache | 难 | World prefix 可 cache | **Tier 1+2 可 cache**，Tier 3 每步重算 |

## 四、为什么 Tier 3 是真正的"灵魂"？

这是 **VLA → World Model 的桥梁**。

GR00T 和 π0.5 都把"动作生成"看作**纯粹的解码问题**——给定 world，输出 action。
**但人类不是这样工作的**。人类是：

```
看 → 想做什么 → 重点看 → 修正想法 → 重点看 → 输出动作
```

这是个**闭环**。每一次"重点看"都是在用动作意图重新组织感知。

Tier 3 的 gate 让模型学到：

- 抓取任务 → gate 让 world 重点看"被抓物体"
- 避障任务 → gate 让 world 重点看"障碍物"
- 长程任务 → gate 让 world 重点看"未来子目标的位置"

**而且最妙的是**：在 flow matching 多步迭代里，每一步 action 都在更新，所以**每一步 world 也会被 active refine 得更精准**。这跟 LLM 的多步 CoT 是同构的——但发生在视觉-动作的共享 latent 空间里。

## 五、一句话精髓

> **GR00T 是 "World 给 Action 一次性提交感知报告"（cross-attn 单点）；π0.5 是 "World 和 Action 平行思考，互相查 K/V 池"（MoT joint attn）；但都缺一个东西——Action 的意图永远无法反向引导 World 的注意力。**
>
> **真正的视觉-动作联立建模应该是 Active Perception 闭环：World 先 task-agnostic 融合（可 cache）→ Action 单向 task-conditioned 读取（高效）→ Action 用 gate 选择性影响 World 注意力（主动感知）→ 多步 flow matching 中逐步收紧 World 和 Action 的协同。**
>
> **这是 GR00T 和 π0.5 都还没做、但 2027 年应该会出现的下一代架构。**

工程上实现这个不难，关键是 **Tier 3 的 gate 不能粗暴全连接**——一定要 gated cross-attn，否则会破坏 Tier 1 的 world 表征稳定性。这是我留给你思考的实现细节。