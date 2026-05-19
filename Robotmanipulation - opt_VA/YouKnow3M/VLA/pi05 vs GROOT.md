先郑重纠正一下我上一轮的错误——我说 GR00T 是 "VL Transformer + 3 层 MLP"是**严重低估了 GR00T**。真实结构是：

- **Action Encoder** 才是 3 层 MLP（只是输入 projection）
- 真正处理动作的主体是 **16 层 DiT Transformer**（`num_layers=16`, `num_attention_heads=32`, `attention_head_dim=48`, `inner_dim=1536`，每层含 Cross-Attn + Self-Attn + FFN，`interleave_self_attention=True`）

所以 GR00T 的动作侧其实是 **"3 层 MLP encoder + 16 层 DiT + 共享底层 Decoder"**，参数量也是几百 M 级别的，绝不是"3 层 MLP 就完了"。你的指正非常关键，我之前的对比是错的。

---

现在正面回答你这两个问题——π0.5 的 action expert 是不是更小的 VLM？怎么编解码？要不要用 FAST token？

## 一、Action Expert 到底是什么？

从代码看真相：

```297:317:src/lerobot/policies/pi05/modeling_pi05.py
def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
```

```369:371:src/lerobot/policies/pi05/modeling_pi05.py
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
```

**Action Expert 的真实样子**：

| 项目 | PaliGemma（VL） | Gemma Expert（Action） |
|------|-----------------|------------------------|
| 模型类 | `PaliGemmaForConditionalGeneration` | `GemmaForCausalLM` |
| width | 2048 | **1024** |
| depth | **18** | **18** ← 一样深 |
| mlp_dim | 16384 | 4096 |
| num_heads | 8 | 8 |
| 视觉编码器 | ✅ SigLIP | ❌ **没有** |
| 文本 embed_tokens | ✅ | ❌ **手动置 None** |
| 参数量 | ~2.6B | ~300M |

所以**严格来说**，你的问题"action expert 可不可以理解为更小的 VLM"——**这个说法不太准确**，更精确的描述应该是：

> **Action Expert 是一个 18 层、width=1024 的"裸 Gemma Decoder"，故意去掉了 vision tower 和 token embedding 表，专门处理连续 action 的 latent embedding，与 PaliGemma 层数对齐，在每层做 joint attention。**

它**没有任何 VLM 能力**——不能看图、不能读文字、不能输出离散 token。它**唯一的功能**就是"接收 action embedding，输出 action embedding"，所有跨模态信息都是从 PaliGemma 那一边通过 joint attention 流过来的。

你可以把它理解成：**"借用了 LLM 架构（Gemma Transformer 结构）作为动作侧的处理骨架"**——不是"小 VLM"，而是"借 Gemma 的壳来装 action 处理"。这是这个设计的精妙之处。

## 二、动作怎么编解码？关键代码

```519:523:src/lerobot/policies/pi05/modeling_pi05.py
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
```

**Action 编码端**（极简）：

$$
\text{action embedding} = W_{\text{in}} \cdot a_t \in \mathbb{R}^{1024}, \quad W_{\text{in}} \in \mathbb{R}^{1024 \times 32}
$$

```648:651:src/lerobot/policies/pi05/modeling_pi05.py
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
```

**就是一个 Linear 层**，把 32 维 noisy action 投影到 1024 维。然后这些 embedding 作为 token 喂给 Gemma Expert。

**Action 解码端**（也是极简）：

```723:729:src/lerobot/policies/pi05/modeling_pi05.py
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
```

也是一个 Linear 层，把 Gemma Expert 的 1024 维输出投影回 32 维。注意输出叫 `v_t`——这是 **flow matching velocity**，不是 action 本身。

**Flow Matching 训练目标**（注意时间约定跟 GR00T 反过来）：

```686:688:src/lerobot/policies/pi05/modeling_pi05.py
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
```

对应数学：

$$
x_t = t \cdot \varepsilon + (1-t) \cdot a, \quad u^* = \varepsilon - a
$$

π0.5 的约定：$t=0$ 是干净动作，$t=1$ 是纯噪声（GR00T 反过来）。本质相同。

## 三、那 FAST Token 在哪？

**关键答案：FAST token 只在 pre-training 阶段使用，post-training 阶段完全是连续 flow matching**。

你给我那张论文图把这个看得最清楚——左右两半截然不同：

### Pre-training（图左）

> 输入：multimodal web data + robot data + task-specific prompts
> 
> 输出（next-token prediction）：
> - `language subtasks` —— "put the plate in the sink"
> - **`discretized actions` —— `(-17, 12, 34, 142, -72, -135)` ← 这就是 FAST token！**
> - `open vocabulary captions` —— "a dog catches a frisbee"
> - `bounding boxes` —— `(3, 35, 145, 223)`

这一阶段**只有 VLM（PaliGemma）**，Action Expert 还没引入。FAST token 是用 DCT + BPE 把连续动作压缩出来的离散整数序列，**当成普通文本 token** 让 PaliGemma 学 next-token prediction。

注意 vocab_size：

```337:338:src/lerobot/policies/pi05/modeling_pi05.py
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
```

257152 = PaliGemma 原生 vocab 256000 + 一些 extra token。FAST token 会扩展到这个 vocab 里（每个动作 bin 对应一个 token id），让模型学会"在文本流里穿插动作 token"。

**目的**：让 VLM **预训练阶段就接触动作语义**，理解"画面+指令 → 应该生成什么动作 token"，建立一个**离散动作的世界模型 prior**。

### Post-training & Inference（图右）

> 输入：图像 + high-level prompt + low-level command + noise
> 
> 输出：
> - `subtask prediction` —— "pick up the pillow"（**离散 token，VLM 输出**）
> - `continuous actions` —— `(-1.7, 1.25, 3.14, 1.42)`（**连续 float，Action Expert 输出**）

这一阶段**新增 Action Expert（300M Gemma）**，把动作从"离散 token"换成"连续 flow matching"。注意：
- VLM 仍然生成 **subtask 文本**（hierarchical 的高层"思考"，"我下一步要做什么"）
- Action Expert 生成 **连续动作向量**（精细的关节/eef 控制）

这就是 π0.5 的**分层（hierarchical）**设计：
- **慢思考（System 2）**：VLM 输出 subtask "pick up the pillow"（语言层面的规划）
- **快执行（System 1）**：Action Expert 输出 continuous actions（精确的运动控制）

## 四、为什么这么设计？两全其美的工程哲学

理解了这两阶段，π0.5 的精妙之处就出来了：

| 阶段 | 表征 | 优势 | 劣势 |
|------|------|------|------|
| Pre-train (FAST token) | 离散 token | ✅ 用海量混合数据预训练<br>✅ 复用 LLM 的 next-token loss<br>✅ 跟语言、bbox、caption 在同一个 token 流里联合训练 | ❌ 离散化损失精度<br>❌ 推理慢 |
| Post-train (Flow Matching) | 连续 float | ✅ 高精度<br>✅ 推理快（几步欧拉积分）<br>✅ 适合实时控制 | ❌ 没有 LLM 那种 next-token prior |

**π0.5 的解法是"取两者之长"**：

1. **用 FAST token 预训练 VLM** —— 让 VLM 学到"动作的语义先验"。即便后续不再用 FAST token，这些预训练权重已经"知道"了**动作和视觉、语言之间的关联**。
2. **用 Flow Matching 后训练** —— 实际推理时用连续动作，避免量化损失，享受 FM 的低步推理优势。
3. **共享 PaliGemma 主干** —— pre-train 的 VLM 权重直接被 post-train 复用（FAST token 时学的视觉-动作关联仍然保留在 attention 权重里）。
4. **加 Action Expert** —— 引入一个轻量的 Gemma Decoder 专门做 FM 速度预测，**不破坏 VLM 主干**。

这就是为什么图里 pre-training 阶段画的是 "pre-trained VLM"，post-training 阶段画的是 "pre-trained VLA"——VLM 通过 FAST token 预训练**变成了 VLA**，然后再加 action expert 做精细化。

## 五、对你最初问题的简洁回答

> "action expert 可以理解为一个更小的 VLM 模块吗？"

**不能**。它是一个**没有视觉、没有 vocabulary、专门处理连续 action embedding 的 18 层 Gemma Decoder（300M）**。结构借自 LLM，功能完全是动作处理。可以理解为"**借了 LLM 壳的 action transformer**"。

> "怎么和动作编解码？FAST token？"

- **Post-training/Inference 阶段（你这份代码所在的阶段）**：**不用 FAST token**。编码是 `Linear(32→1024)`，解码是 `Linear(1024→32)`，输出是 flow matching 的 velocity，连续动作通过欧拉积分得到。
- **Pre-training 阶段**：用 **FAST token**，把动作离散化成整数序列，跟 caption、bbox 一起做 next-token prediction，让 VLM 学会动作的语义先验。

> "和 GR00T 比谁好？"（顺带补一句对前面错误的总结）

更公平的对比应该是：

| 维度 | GR00T N1.5 | π0.5 |
|------|------------|------|
| VL 主干 | Eagle (SigLIP + Qwen3 12 层) | PaliGemma (SigLIP + Gemma 18 层) |
| 动作主干 | **16 层 DiT (1536 inner_dim, 32 heads)** | **18 层 Gemma Expert (1024 width, 8 heads)** |
| 动作主干预训练 | **从零训练**（DiT 没有预训练） | **可继承 LLM 初始化**（Gemma 架构，可加载 LLM 权重 + FAST token pre-train） |
| 信息流 | DiT cross-attn（单向 VL→action） | MoT joint attention（prefix-LM，仍单向） |
| 动作表征预训练 | 仅靠 demonstration | **FAST token pre-train + flow matching post-train** |

所以 π0.5 真正甩开 GR00T 的不是"双向 vs 单向"，也不是"Transformer vs DiT"，而是：

> **π0.5 通过 FAST token 让 VLM 在 pre-training 阶段就把动作语义"吃进去"，然后 post-training 加 action expert 做连续输出 —— 这条"discrete pre-train + continuous post-train"的双阶段路线，是 GR00T 完全没做的。**

这是 π0.5 论文最重要的贡献。我之前因为没仔细看图、没仔细看 DiT 配置，把 GR00T 说得太弱、把 π0.5 简化成"双向 attention"，都是错的。再次为之前的草率道歉。


非常好的问题，这其实是 π0.5 设计上一个**重要的细节**。我从代码和架构两个角度给你看清楚。

## 一、先看 lerobot 这边 `from_pretrained` 怎么加载

```893:944:src/lerobot/policies/pi05/modeling_pi05.py
        model = cls(config, **kwargs)

        # Now manually load and remap the state dict
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)
            ...
            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)
```

这里 lerobot 干的事情很清楚：
1. 用 config **实例化一个空模型**（PaliGemma + Gemma Expert 都是 HF Transformers 默认的 `_init_weights`，正态分布随机初始化）
2. 然后**整个从 OpenPI 发布的 checkpoint 加载所有权重**（PaliGemma + Gemma Expert + action_in_proj + action_out_proj 全部一起）

所以**当你用 lerobot 加载 π0.5 时，你拿到的是 OpenPI 训练好的 checkpoint**，Action Expert 的权重已经是训练好的状态了。

但你真正问的是更深一层——**OpenPI 第一次训练 Action Expert 时，它是怎么初始化的？是不是从 PaliGemma 拷贝？**

## 二、Action Expert 第一次训练时的初始化（关键）

### 1. 维度根本对不上，无法直接复制

```297:317:src/lerobot/policies/pi05/modeling_pi05.py
def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
```

| 参数 | PaliGemma（2B） | Gemma Expert（300M） |
|------|-----------------|----------------------|
| `width` (hidden_size) | **2048** | **1024** |
| `mlp_dim` | **16384** | **4096** |
| depth | 18 | 18 |
| head_dim | 256 | 256 |
| num_heads | 8 | 8 |

**hidden_size 差一倍，mlp_dim 差四倍**——所以 **PaliGemma 的 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`MLP` 这些权重矩阵的 shape 跟 Gemma Expert 完全不匹配**，根本不可能直接 copy 过去。

唯一**可能**对齐的就是 `head_dim=256` 这一项——这是设计上的精明：保证两边在 attention 计算时每个 head 的维度一致，**joint attention 时 K/V 能在 head 维度上对齐拼接**。但这只是结构对齐，跟权重继承没关系。

### 2. 代码层面看：用 config 实例化 + `embed_tokens = None`

```369:371:src/lerobot/policies/pi05/modeling_pi05.py
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
```

注意三个事实：

1. **`GemmaForCausalLM(config=...)`** —— HF Transformers 用 config 实例化时会调用每个子模块的 `_init_weights()`，**默认是截断正态分布 init（std=0.02 左右）**。完全随机。
2. **没有任何 `load_state_dict` 从 PaliGemma 拷贝到 Gemma Expert** —— 整份代码里你找不到这种操作。
3. **`embed_tokens = None`** —— 直接把 token embedding 表删除，说明这个 Expert 根本不接收离散 token 输入，它的"输入语义"完全靠 `action_in_proj` 这个 Linear 投影来构造。

### 3. OpenPI 原版训练流程（用论文/官方信息补充）

π0.5 论文（Physical Intelligence, 2025）和 OpenPI 开源代码里描述的训练流程是：

**Phase 1: Pre-training**
- 模型：**只有 PaliGemma**（SigLIP + Gemma 2B），**没有 Action Expert**
- PaliGemma 本身是从 Google PaliGemma 的预训练 checkpoint 加载 → **有完整的视觉-语言 prior**
- 训练数据：multimodal web data + robot data
- 任务：用 FAST token 把动作离散化，和 caption、bbox、subtask 一起做 next-token prediction

**Phase 2: Post-training**
- **此时才把 Gemma Expert（300M）接到 PaliGemma 旁边**
- PaliGemma 主干：**继承 Phase 1 的权重**（保留了 FAST token 阶段学到的动作语义）
- Gemma Expert：**完全从随机初始化开始**
- 训练目标：flow matching（连续动作）+ subtask LM loss
- 所有部分一起训练，但学习率/调度可能不同

所以回到你的问题：

| 问题 | 答案 |
|------|------|
| Action Expert 只在 post-train 引入？ | **是的**，pre-train 阶段不存在 |
| Action Expert 权重是随机初始化？ | **是的**，截断正态分布 |
| 是否继承 PaliGemma 权重？ | **不继承**（width 1024 vs 2048，维度对不上） |
| 那 PaliGemma 主干呢？ | **继承 Phase 1 pre-train 权重**（FAST token 阶段学到的视觉-动作-语言对齐） |

## 三、为什么 OpenPI 不让 Action Expert 继承 PaliGemma？

这其实是个**有意的设计选择**，不是技术限制。让我讲三个理由：

### 理由 1：让 Action Expert 保持"小而专"

PaliGemma 是 2B，Gemma Expert 才 300M。**故意做小**：
- 推理时 KV cache 占用小（实时控制需要）
- 训练时 Action Expert 是 trainable 的（PaliGemma 主干可以部分 frozen），梯度只在 300M 上算
- 部署时 latency 低

如果硬要让 Action Expert 继承 PaliGemma，那就得用 2B 同样的 width——参数量直接翻倍。**不划算**。

### 理由 2：避免"语言先验"污染"动作语义"

PaliGemma 的权重是面向**离散 token 序列**优化的（自然语言、图像 patch）。Action 是**连续控制信号**——空间结构、时序连续性、运动学约束都不同。

**直接拿 LLM 权重去 init action transformer，反而可能引入错误的归纳偏置**。比如：
- LLM 的 attention 倾向于"局部+长程语义"
- Action 的 attention 应该更关注"短程时序+空间相关"
- 用 LLM 权重 warm-start 反而要花更多迭代去 "un-learn" 这些偏置

OpenPI 的选择是：**让 Action Expert 完全从动作数据里学，避免被语言 prior 误导**。

### 理由 3：信息流向是"VL → Action"，不是"Action 自我演化"

回到 joint attention 那个分析：

$$
\text{Action's K/V (random init)} \to \text{Action's Attention} \xleftarrow{\text{shared in joint attn}} \text{VL's K/V (pre-trained)}
$$

Action Expert 真正"吃饭的本事"在于**能从 VL 的预训练 K/V 池里抓信息**，而不在于自己的初始权重。所以 Action 端的 K/V 是不是预训练其实没那么重要——**反正每一层 attention 它都能拿到 PaliGemma 的 pre-trained K/V**。

类比：你不需要让一个"学徒"提前学满所有知识，只要让他**坐在大师旁边能随时听课**，他就能逐渐学到东西。π0.5 的 Action Expert 就是这个"学徒"，PaliGemma 是"大师"，joint attention 是"教室"。

## 四、对比 GR00T 的初始化策略

```text
GR00T:
  ├─ Eagle/VLM:    Eagle-2 预训练（继承）
  ├─ Action Encoder (3层MLP):     随机初始化
  ├─ VL Self-Attn (4层):          随机初始化  
  ├─ DiT (16层):                  随机初始化 ← 动作侧主干完全从零
  └─ Decoders:                    随机初始化

π0.5:
  ├─ PaliGemma (VL + 18层 LLM):   Google PaliGemma 预训练 + FAST token pre-train（继承）
  │                                ↑ 经过两段预训练，强 prior
  ├─ Gemma Expert (18层):          随机初始化 ← 动作侧主干也是从零
  ├─ action_in_proj (Linear):     随机初始化
  └─ action_out_proj (Linear):    随机初始化
```

仔细看你会发现：**两个模型的动作侧主干（DiT vs Gemma Expert）都是随机初始化的**！

那 π0.5 真正比 GR00T 强在哪？答案在 **VL 端的 pre-training**：

- GR00T 的 VL backbone（Eagle-2）只经过通用的 VLM pre-train，**没有专门让 VL backbone "看见过" 动作**
- π0.5 的 VL backbone（PaliGemma）经过 **FAST token pre-training**，**已经学过"看到画面+指令 → 生成什么离散动作"的映射**

所以 π0.5 的优势不在 action expert 本身（它一样是随机初始化），而在于：

> **它的 VL 主干已经把"动作语义"吃进去了。当 Action Expert 通过 joint attention 从 VL 主干抓 K/V 时，它抓到的是一个"懂动作的 VL 表征"，而不是 GR00T 那种"只懂视觉/语言、不懂动作"的 VL 表征。**

这就是 FAST token pre-training 真正的作用——**不是为了让模型最终输出离散 token，而是为了让 VL backbone 在表征空间里"知道动作长什么样"**。然后 Action Expert 这个随机初始化的小学徒，只要能"听到"这些 VL 表征里的动作语义，就能比 GR00T 学得快、学得好。

## 五、一句话总结

> **Action Expert 是 post-training 阶段才引入的、从零随机初始化的 300M Gemma Decoder（去掉了 vision 和 token embedding），它的权重不继承自 PaliGemma（维度根本对不上）。但它"沾光"的方式是通过每层 joint attention 从 PaliGemma 那里抓 pre-trained K/V——而 PaliGemma 经过 FAST token pre-training 后，K/V 里已经编码了"动作语义"，这才是 π0.5 比 GR00T 强的根本原因。**

Action Expert 自己虽然随机初始化，但它**站在一个"懂动作"的 VL 主干肩膀上**。GR00T 的 DiT 自己也随机初始化，但它**站在一个"只懂视觉语言、不懂动作"的 VL 主干肩膀上**——这就是 2025 年 VLA 架构进化的本质方向。

确认了：π0.5 的 `forward` 和 `sample_actions` 签名里**没有 state 参数**——状态信息完全靠语言 prompt 传入。现在我有足够信息做这个对比。

# π0.5 Action Expert vs GR00T DiT Flow-Matching Head 全面对比

两者都是"动作侧 transformer，用 flow matching 训练"，但设计哲学差异非常大。我从 10 个维度逐一对比。

## 一、整体结构对比

### GR00T `FlowmatchingActionHead`（动作侧完整组件）

```text
ActionHead
├─ vlln: LayerNorm(2048)
├─ vl_self_attention: SelfAttentionTransformer (4层)         ← VL 二次增强
├─ state_encoder: CategorySpecificMLP (64 → 1024 → 1536)     ← 显式状态编码
├─ action_encoder: MultiEmbodimentActionEncoder (3层MLP)      ← 动作输入投影
├─ position_embedding: Embedding(max_seq, 1536)              ← 可选
├─ future_tokens: Embedding(32, 1536)                        ← 任务无关 query token
├─ model: DiT (16层, inner_dim=1536, cross_attn_dim=2048)    ← 主干
│   每层 BasicTransformerBlock：
│   - 偶数层: Cross-Attention + FFN
│   - 奇数层: Self-Attention + FFN   ← interleave_self_attention
└─ Decoders:
    ├─ SharedBottomArmDecoder: 共享底层 + 左右手交叉注意力
    ├─ action_arm_decoder / action_claw_decoder
    └─ EEF 模式下: pos + rot 分开输出
```

### π0.5 Action Expert（轻量极简）

```text
Action Side
├─ action_in_proj: Linear(action_dim → 1024)                  ← 唯一输入编码
├─ time_mlp_in/out: Linear(1024 → 1024) × 2                  ← time embedding
├─ gemma_expert: GemmaForCausalLM (18层, width=1024)         ← 主干
│   每层：
│   - input_layernorm (AdaRMSNorm w/ time cond)
│   - q_proj/k_proj/v_proj/o_proj
│   - **joint attention with PaliGemma's K/V**              ← 核心信息流
│   - post_attention_layernorm (AdaRMSNorm w/ time cond)
│   - MLP (GeGLU)
└─ action_out_proj: Linear(1024 → action_dim)                 ← 唯一输出解码
```

**第一眼差异**：GR00T 是"动作侧自己也是一个独立小系统（带 vl_self_attn、state encoder、future tokens、复杂 decoder）"；π0.5 是"动作侧只是一个 Gemma Decoder，所有跨模态交互全靠 joint attention"。

## 二、主干 Transformer 配置对比

| 项目 | GR00T DiT | π0.5 Gemma Expert |
|------|-----------|--------------------|
| 层数 | **16** | **18** |
| inner_dim / width | **1536** | **1024** |
| num_heads | **32** | **8** |
| head_dim | **48** | **256** |
| MLP intermediate | 6144（4×inner） | **4096** |
| MLP 激活 | GELU-approx | **GeGLU** (Gemma) |
| Attention 类型 | 自定义 Cross-Attn + Self-Attn | LLM 风格 Self-Attn (with joint K/V) |
| LayerNorm 类型 | **AdaLayerNorm** (有 elementwise affine) | **AdaRMSNorm** (Gemma 原生) |
| Cross-attn 维度 | 2048 (VL) → 1536 (action) | 256 (head_dim，跨流共享) |
| Position embedding | sinusoidal (可选) | **RoPE** (Gemma 原生) |

注意 **head_dim 的巨大差异**：
- GR00T：32 头 × 48 维 = 1536 → "多头细颗粒"，每个头关注 48 维子空间，适合 diffusion 的多模态去噪
- π0.5：8 头 × 256 维 = 1024 → "少头粗颗粒"，跟 LLM 一致，**关键是 head_dim=256 能跟 PaliGemma 的 head_dim=256 对齐**，joint attention 才能在 head 维度直接拼接！

这是 π0.5 的精妙设计：**故意让 head_dim 等于 PaliGemma**，width 减半就靠减 head 数实现，保证 joint attention 时 K/V 能直接拼起来。

## 三、信息流向：Cross-Attention vs Joint Attention

### GR00T：每层 cross-attention（投影后单点交叉）

```274:291:src/lerobot/policies/groot/action_head/cross_attention_dit.py
        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )
```

每个偶数层：
- Q 来自 action tokens (1536维)
- K, V 来自 VL features (2048维) → 通过本层独有的 `to_k`, `to_v` **重新投影到 1536**

**16 层中有 8 层做 cross-attn，每层都要重新投影 VL** → 等于 8 套独立的"VL → action 翻译器"。

### π0.5：每层 joint attention（K/V 池跨流共享）

```219:256:src/lerobot/policies/pi05/modeling_pi05.py
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
```

每一层：
- VL 用 PaliGemma 的 `q_proj_VL, k_proj_VL, v_proj_VL` 算 K/V
- Action 用 Gemma Expert 的 `q_proj_A, k_proj_A, v_proj_A` 算 K/V
- 在 sequence 维度拼接，**一次 attention 同时处理两路**

**关键差异**：
- GR00T：每层重新算 VL 的 K/V（用动作侧专门的 `to_k`, `to_v` 投影矩阵），**VL 的预训练表征被"翻译"了 8 次**
- π0.5：VL 的 K/V 一直由 PaliGemma 自己的 k_proj/v_proj 算，**没有任何"翻译损失"，保留完整的预训练 prior**

## 四、Time（噪声步）的注入方式

### GR00T：离散 bucket + AdaLayerNorm

```943:945:src/lerobot/policies/groot/action_head/flow_matching_action_head.py
        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
```

- 连续 $t \in [0,1]$ 离散化到 1000 个 bucket
- 通过 `TimestepEncoder` 编码
- 每层 DiT 的 AdaLN 用 timestep 调制 `scale` 和 `shift`

### π0.5：连续 sinusoidal + AdaRMSNorm

```638:661:src/lerobot/policies/pi05/modeling_pi05.py
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
        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)
```

- 连续 $t$ 直接用 sinusoidal 编码（**不离散化**）
- 通过 2 层 MLP 处理成 `adarms_cond`
- 注入到 Gemma Expert 的 AdaRMSNorm（Gemma 原生支持）

**差异**：GR00T 把 t 当成"离散标签"，π0.5 把 t 当成"连续标量"。后者理论上对 4~10 步推理更友好（不会被 1000 个 bucket 限制）。

## 五、State（机器人本体状态）的处理 ⚠️

这是个**巨大的设计差异**。

### GR00T：显式 state encoder + state token

```247:247:docs/GROOT_N1_DETAILED_ARCHITECTURE.md
| State Encoder | CategoryMLP | B×64 | B×1×1536 | 状态编码<br/>**可选：RGB-only模式下为None** |
```

- 64 维 state → CategorySpecificMLP → 1536 维 → 加入 DiT 输入序列
- 跟 future_tokens 和 action_features 拼接作为 hidden_states

### π0.5：**完全没有 state encoder**

```678:678:src/lerobot/policies/pi05/modeling_pi05.py
    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
```

```734:734:src/lerobot/policies/pi05/modeling_pi05.py
    def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=None) -> Tensor:
```

签名里**没有 state 参数**！π0.5 的做法是：
- **本体状态通过 prompt 文本注入**（比如 "current joint angles are [...]"）
- 或者完全不使用本体状态（依靠多视角图像 + 历史观测）

**哲学差异**：
- GR00T：state 是一等公民，单独编码
- π0.5：state 是文本的一种，进 tokenizer 就行（统一表征）

π0.5 这么做的好处：**跨机器人迁移更容易**（不同机器人的 state 维度不同，但语言可以统一描述）；坏处：**精度可能不如显式 encoder**（文本量化精度不如 float64）。

## 六、Query / Future Tokens 的存在与否

### GR00T：有 32 个固定 future_tokens

```96:96:docs/GROOT_N1_DETAILED_ARCHITECTURE.md
            FutureTok["Future Tokens<br/>Embedding(32, 1536)"]
```

```974:977:src/lerobot/policies/groot/action_head/flow_matching_action_head.py
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        # 6) 拼接为 hidden_states
        # RGB-only mode: exclude state_features if state_encoder is disabled
        if state_features is not None:
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
```

32 个**任务无关的 learnable embedding**，作为额外的 query token 跟 cross-attention 一起在 DiT 里走 16 层。设计目的：让模型有"额外的思考槽位"去吸收 VL 信息。这其实是 **Perceiver-style query** 的简化版。

### π0.5：没有 future_tokens

直接把 action_features（chunk_size 个）作为 hidden_states 进 Gemma Expert：

```663:666:src/lerobot/policies/pi05/modeling_pi05.py
        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)
```

**没有额外的"思考槽位"**。π0.5 的设计哲学是：**action token 自己就够 expressive 了**，每个 action token 通过 18 层 joint attention 已经能充分吸收信息。

## 七、Decoder（输出层）

### GR00T：极度复杂

```114:143:docs/GROOT_N1_DETAILED_ARCHITECTURE.md
        subgraph Decoders["解码器 tune_projector控制"]
            subgraph ArmDecoder["Arm Decoder (可选架构)"]
                SharedLayer["共享底层特征提取<br/>CategorySpecificLinear<br/>1024 to 1024<br/>ReLU激活"]
                
                subgraph CrossAttn["交叉注意力机制 (可选)<br/>use_cross_attention_arms"]
                    LayerNormL["LayerNorm Left<br/>归一化左手特征"]
                    LayerNormR["LayerNorm Right<br/>归一化右手特征"]
                    CrossAttnL["Cross-Attn Left<br/>MultiheadAttention<br/>query: left_features<br/>key/value: right_features"]
                    CrossAttnR["Cross-Attn Right<br/>MultiheadAttention<br/>query: right_features<br/>key/value: left_features"]
                    ResidualL["残差连接<br/>left + left_attended"]
                    ResidualR["残差连接<br/>right + right_attended"]
                end
```

GR00T 有：
- **SharedBottomArmDecoder**：左右手共享底层 + 各自输出层
- **可选 cross-attention**：左右手交叉关注（双手协调）
- **EEF 模式下**：position 用 cross-attn，rotation 独立处理（保几何约束）
- **separate claw decoder**：夹爪单独处理
- **CategorySpecific**：每个 embodiment 独立权重

### π0.5：一个 Linear 完事

```519:520:src/lerobot/policies/pi05/modeling_pi05.py
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
```

**就这两行**——一个 Linear 进，一个 Linear 出。

**哲学差异**：
- GR00T：**结构先验显式注入**（左右手对称、EEF 几何约束、夹爪独立性）
- π0.5：**让 Transformer 自己学**（不假设动作的任何结构）

GR00T 的方法在**特定机器人形态**（人形双臂+夹爪）上数据效率更高；π0.5 的方法在**跨机器人迁移**上更通用。

## 八、多 Embodiment 支持

### GR00T：每层都是 category-specific

```44:55:src/lerobot/policies/groot/action_head/flow_matching_action_head.py
class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_w = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)
```

**Action Encoder、State Encoder、Decoder 每一层**都是 category-specific，每个 embodiment 维护 32 套权重。

### π0.5：完全没有 embodiment 概念

单一权重处理所有机器人，**靠数据多样性 + 文本 prompt 区分**。

**这反映了两个团队不同的目标**：
- NVIDIA GR00T：面向**有限种类的人形机器人**深度优化
- Physical Intelligence π0.5：面向**任意机器人**的通用基础模型

## 九、训练 Loss 复杂度

### GR00T：多任务复杂 loss

看你那份 flow_matching_action_head.py 的 1019~1196 行，GR00T 有：
- arm loss、claw loss 分别计算
- 可学习的 task uncertainty（`task_log_sigma`）：

$$
\mathcal{L} = \sum_k \frac{1}{2\sigma_k^2}\mathcal{L}_k + \log\sigma_k
$$

- arm coordination loss（左右手协调辅助 loss）
- learnable loss weights
- detection 辅助 loss

### π0.5：纯 MSE

```1148:1159:src/lerobot/policies/pi05/modeling_pi05.py
        # Truncate losses to actual action dimensions
```

就是 `F.mse_loss(v_t, u_t)`，没有任何花哨的东西。

## 十、参数量与推理速度

| 项目 | GR00T DiT + Decoder | π0.5 Gemma Expert |
|------|---------------------|--------------------|
| 主干参数量 | ~500M（DiT 16 层 1536 维 + decoder） | **~300M**（Gemma 18 层 1024 维） |
| 推理步数 | 4~16 步 | **10 步** |
| 是否支持 KV cache | 难（每层 cross-attn 重算） | **✅ 支持 prefix KV cache** |
| 单步推理延迟 | 较高 | **较低**（KV cache 加持） |

π0.5 的 prefix KV cache 是关键优势：

```751:768:src/lerobot/policies/pi05/modeling_pi05.py
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
```

VL 部分 **算一次就缓存**，后续 10 步 flow matching 迭代时**只重算 action 部分**。GR00T 因为 cross-attn 每层都要重投影 VL，没法做这种优化。

## 十一、终极总结表

| 维度 | GR00T DiT Flow-Matching Head | π0.5 Action Expert |
|------|-------------------------------|---------------------|
| 主干 | 自定义 DiT（diffusion specific） | Gemma Decoder（LLM specific） |
| 层数 | 16 | 18 |
| width × heads × head_dim | 1536 = 32 × 48 | 1024 = 8 × **256** |
| 跨模态融合 | Cross-attention（每层重投影 VL） | **Joint attention**（共享 K/V 池） |
| Self/Cross 交替 | 偶 cross + 奇 self（interleave） | 全部 joint attn |
| 时间注入 | 离散 bucket + AdaLayerNorm | 连续 sinusoidal + AdaRMSNorm |
| State 处理 | **显式 CategoryMLP encoder** | **没有，靠文本 prompt** |
| Query tokens | **32 future_tokens** | 无 |
| 输入序列 | [state(1), future(32), action(T)] | [action(T)] |
| Position encoding | sinusoidal (可选) | **RoPE** |
| Decoder | SharedBottom + 左右手 cross-attn + EEF/Joint 分离 + claw 独立 | **一个 Linear** |
| Multi-embodiment | **CategorySpecific 每层一套权重** | 无显式支持 |
| Loss | 多任务 + 可学习权重 + 协调 loss | 纯 MSE |
| KV cache | ❌ 难 | **✅ Prefix cache** |
| 设计哲学 | **结构先验显式注入**，特定形态优化 | **让 Transformer 自己学**，通用基础模型 |

## 一句话精髓

> **GR00T 的 DiT 是"针对人形双臂机器人深度定制的 diffusion transformer"——结构上塞满了 inductive bias（state encoder、future tokens、左右手 cross-attn、EEF 几何约束、claw 独立、category-specific 权重）。π0.5 的 Gemma Expert 是"LLM 风格的极简 action transformer"——除了一个 Linear 进、一个 Linear 出、然后跟 PaliGemma 做 joint attention，其他全部交给 Transformer 自己学。**

这两种哲学没有绝对的优劣：
- **如果你只做一类人形机器人**，GR00T 的归纳偏置会让你数据效率更高、特定任务效果更好
- **如果你要做"机器人界的 GPT"**，π0.5 的极简通用设计更有 scaling 潜力——参数翻倍、数据翻倍，性能更容易跟上

而 GR00T 真正的进化方向应该是：

1. **去掉 cross-attention，换成 joint attention** —— 保留 VL 的预训练 K/V 表征不被翻译损失
2. **去掉 state encoder，让 state 进 prompt** —— 跨机器人通用
3. **统一 head_dim** —— 跟 Eagle/Qwen3 的 head_dim 对齐，让 joint attention 能直接拼
4. **简化 decoder** —— 让 Transformer 自己学动作结构

这条路线，本质上就是**让 GR00T 变得更像 π0.5**。我个人觉得这是 2026~2027 年 NVIDIA 会走的方向。