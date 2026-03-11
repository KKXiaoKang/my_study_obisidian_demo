[7 tools called]

非常好的问题！你对 Kai0 和 π*₀.₆ 的理解是准确的。让我详细分析当前 GROOT N1.5 的 Flow Matching Action Head 架构，以及如何添加 advantage conditioning。

## 当前架构核心数据流

```
训练:
  vl_embs (VL特征) ─────────────────────────────────> DiT (K/V)
  sa_embs = [state, future_tokens, action_features] ─> DiT (Q)
  t_discretized ─────────────────────────────────────> DiT (temb → AdaLayerNorm)
  DiT输出 → action_decoders → pred_velocity

推理 (denoise loop):
  同上，但action从噪声开始，迭代去噪
```

关键注入点如下图所示：

```
                         ┌──────────────────────────────┐
                         │    DiT Cross-Attention       │
                         │                              │
 vl_embs (VL features) ─>│  K/V: encoder_hidden_states │
                         │                              │
                         │  Q: hidden_states (sa_embs)  │<── 【注入点A】advantage token
                         │                              │
 temb (timestep_emb) ──>│  AdaLayerNorm conditioning   │<── 【注入点B】advantage + temb
                         │                              │
                         └──────────────────────────────┘
                                      │
                         model_output_actions
                                      │
                         ┌──────────────────┐
                         │  Action Decoders  │
                         └──────────────────┘
```

---

## 三种实现方案

### 方案一：π*₀.₆ 风格 — Advantage Token 注入 SA 序列（推荐 ⭐）

**原理**：将二值化 advantage indicator 编码为一个 token，拼入 `sa_embs` 序列，让 DiT 自然地通过 self-attention 关注它。训练时 30% dropout 用于 CFG。

```
sa_embs = [advantage_token, state_features, future_tokens, action_features]
                  ↑
        新增的 advantage 条件 token
```

**核心改动**:

```python
# === Config 新增参数 ===
# FlowmatchingActionHeadConfig 中新增:
use_advantage_conditioning: bool = False    # 是否启用 advantage conditioning
advantage_dropout_rate: float = 0.3         # 训练时 advantage dropout 概率 (用于 CFG)
advantage_cfg_scale: float = 1.5            # 推理时 CFG guidance scale β

# === FlowmatchingActionHead.__init__ 中新增 ===
if config.use_advantage_conditioning:
    # 3 个 advantage 状态: 0=negative, 1=positive, 2=null(dropout/unconditioned)
    self.advantage_embedding = nn.Embedding(3, self.input_embedding_dim)
    nn.init.normal_(self.advantage_embedding.weight, mean=0.0, std=0.02)
    self.advantage_dropout_rate = config.advantage_dropout_rate
    self.advantage_cfg_scale = config.advantage_cfg_scale
```

**训练 forward 中的修改**（在拼接 `sa_embs` 之前）：

```python
# forward() 方法中，拼接 sa_embs 之前:
if self.config.use_advantage_conditioning:
    # action_input.advantage_indicator: (B,) ∈ {0, 1}
    adv_ids = action_input.advantage_indicator.clone()  # (B,)
    
    # 训练时 30% dropout: 将 advantage 设为 null (id=2)
    if self.training:
        dropout_mask = torch.rand(adv_ids.shape[0], device=adv_ids.device) < self.advantage_dropout_rate
        adv_ids[dropout_mask] = 2  # null token
    
    advantage_token = self.advantage_embedding(adv_ids).unsqueeze(1)  # (B, 1, D)
    
    # 拼入 sa_embs
    if state_features is not None:
        sa_embs = torch.cat((advantage_token, state_features, future_tokens, action_features), dim=1)
    else:
        sa_embs = torch.cat((advantage_token, future_tokens, action_features), dim=1)
else:
    # 原始逻辑
    if state_features is not None:
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
    else:
        sa_embs = torch.cat((future_tokens, action_features), dim=1)
```

**推理 `get_action` 中的 CFG**:

```python
# get_action() 推理时使用 CFG:
if self.config.use_advantage_conditioning and self.advantage_cfg_scale > 1.0:
    # 两次前向传播
    # 1. 条件预测 (advantage=1, positive)
    v_cond = self.denoise_step(
        x_t, timestep, vl_embs, state_features, embodiment_id,
        advantage_indicator=torch.ones(batch_size, device=device, dtype=torch.long)  # positive
    )
    # 2. 无条件预测 (advantage=2, null)
    v_uncond = self.denoise_step(
        x_t, timestep, vl_embs, state_features, embodiment_id,
        advantage_indicator=torch.full((batch_size,), 2, device=device, dtype=torch.long)  # null
    )
    # 3. CFG 组合
    β = self.advantage_cfg_scale
    v_t = v_uncond + β * (v_cond - v_uncond)
else:
    v_t = self.denoise_step(x_t, timestep, vl_embs, state_features, embodiment_id)
```

---

### 方案二：Advantage 注入 Timestep Embedding

**原理**：将 advantage 编码后叠加到 DiT 的 `temb`（timestep embedding），通过 AdaLayerNorm 影响每一层。

```python
# cross_attention_dit.py 的 DiT 中新增:
class DiT(ModelMixin, ConfigMixin):
    def __init__(self, ..., use_advantage_conditioning=False):
        ...
        if use_advantage_conditioning:
            # advantage encoder: 映射 advantage_id → embedding
            self.advantage_encoder = nn.Sequential(
                nn.Embedding(3, 256),  # 0=neg, 1=pos, 2=null
                nn.Linear(256, self.inner_dim),
                nn.SiLU(),
                nn.Linear(self.inner_dim, self.inner_dim),
            )
    
    def forward(self, hidden_states, encoder_hidden_states, timestep, 
                advantage_indicator=None, ...):
        temb = self.timestep_encoder(timestep)  # (B, D)
        
        # 将 advantage 信息叠加到 temb
        if advantage_indicator is not None and hasattr(self, 'advantage_encoder'):
            adv_emb = self.advantage_encoder(advantage_indicator)  # (B, D)
            temb = temb + adv_emb  # 叠加：影响所有 AdaLayerNorm
        
        # 后续逻辑不变...
```

这个方案更简洁，但表达能力略弱于方案一（因为只通过 scale/shift 影响，没有 attention 机制）。

---

### 方案三：Kai0 风格 — 通过 Text Prompt

**原理**：不修改 `flow_matching_action_head.py`，而是在数据预处理时修改 prompt。

```python
# 在数据处理/预处理阶段:
# 原始 prompt: "pick up the cup"
# 修改后: "pick up the cup, Advantage: positive"  或  "pick up the cup, Advantage: negative"

# processor_groot.py 或数据集处理中:
if advantage_indicator == 1:
    task_prompt = f"{original_prompt}, Advantage: positive"
elif advantage_indicator == 0:
    task_prompt = f"{original_prompt}, Advantage: negative"
else:
    task_prompt = original_prompt  # dropout 时不加
```

**优点**：零代码修改 action head。**缺点**：Advantage 信息只能通过语言模型间接影响，精度有限。

---

## 推荐方案：方案一（π*₀.₆ 风格 Advantage Token）

理由：
1. **与 Flow Matching 天然兼容**：π*₀.₆ 就是在 flow matching 模型上做的 advantage conditioning
2. **CFG 能力**：训练时 dropout → 推理时 CFG，可以动态调整 guidance 强度
3. **对现有参数无侵入**：只新增一个 `nn.Embedding(3, D)` 和拼接逻辑
4. **预训练兼容**：可以先加载预训练权重，只新增的 `advantage_embedding` 从零初始化

---

## 完整的数据流对比

```
原始 GROOT N1.5:
  数据 → [state, action, video, task_prompt] → backbone → vl_embs
                                                          ↓
  sa_embs = [state(1), future_tokens(32), action_features(T)]  
            ↓ DiT cross-attention (Q=sa, K/V=vl, temb=timestep) ↓
  → action_decoders → velocity → loss

新增 π*₀.₆ 风格 advantage conditioning:
  数据 → [state, action, video, task_prompt, advantage_indicator] → backbone → vl_embs
                                                                              ↓
  sa_embs = [advantage_token(1), state(1), future_tokens(32), action_features(T)]  ← 新增
            ↓ DiT cross-attention (Q=sa, K/V=vl, temb=timestep) ↓
  → action_decoders → velocity → loss (不变)

推理 (CFG):
  v_cond   = denoise(advantage=1)   # positive
  v_uncond = denoise(advantage=2)   # null
  v_final  = v_uncond + β*(v_cond - v_uncond)   # β > 1 时偏向 "好" 的动作
```

---

## 数据侧需要准备什么？

无论选哪个方案，你都需要在训练数据中提供 `advantage_indicator`：

1. **训练一个 Value Function** $V(o_t)$（可以是简单的 MLP，也可以是独立的小型 VLA）
2. **计算 Advantage**：$A_t = R_t - V(o_t)$（或使用 progress 差值 $V(o_{t+1}) - V(o_t)$）
3. **二值化**：$I_t = \mathbf{1}[A_t > \epsilon]$
4. **将 `advantage_indicator` 加入数据集**：作为一个额外字段传入 `action_input`

需要我直接在 `flow_matching_action_head.py` 中实现方案一的完整代码吗？