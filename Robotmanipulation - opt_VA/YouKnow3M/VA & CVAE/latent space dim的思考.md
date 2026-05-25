下面我把综述（2605.12090 WAM survey）、DiT4DiT（2603.10448）、lingbot-va 仓库本身、以及 V-JEPA 2 / iVideoGPT / Wan2.2-VAE 这几条线交叉对齐，给你一份关于"WAM latent space dim 该怎么衡量、怎么选、怎么验证"的系统思考。

---

## 1. 先把"latent space dim"拆成 4 个互不相同的轴

很多人混着说"latent 维度"，但在 WAM 里其实至少要分清四层：

| 轴 | 含义 | lingbot-va 当前值 | DiT4DiT | V-JEPA 2 | iVideoGPT |
|---|---|---|---|---|---|
| 1. **VAE 通道维 z_dim**（每个 latent voxel 的连续通道数） | 信息容量 / 重建上限 | **48**（Wan2.2 VAE，见下） | Cosmos-Predict2.5 VAE（连续） | 不用 VAE（直接 ViT 特征） | 离散 VQ token（无连续通道） |
| 2. **时空网格分辨率**（latent T×H×W） | token 数量，注意力计算量 | 时间 ÷4，空间 ÷16；再 patch (1,2,2) | 空间/时间下采样 | tubelet 2×16×16，256² 输入 → 16×16×D | ctx_vqgan，渐近 ≈ 16× token 削减 |
| 3. **Transformer 隐空间维 d_model**（骨干内部 token 宽度） | 表达力 / 参数量 | inner_dim = 24×128 = **3072**，FFN 14336 | DiT 隐藏维（被 Action DiT 通过 hook 抽出） | 编码器 1024/1280/1408/1536；预测器 384 | LLaMA 风格 138M / 436M |
| 4. **"瓶颈"传递维**（World → Action 那一段送出去的 dim） | 解耦泛化 / 抗 shortcut | 与 d_model 相同（直接联合训练） | DiT 中间 hidden state（model dim） | 预测器输出回到 encoder dim | 离散 token 序列（每 token logits） |

只看一个数字（比如"latent 是 48"）是没有意义的，要谈维度大小必须**指明这四层之一**。

---

## 2. lingbot-va 里"latent 各层维度"的硬事实

把仓库里实打实的配置摆出来。

VAE（`lingbot_model_config/vae/config.json`）：

```17:128:lingbot_model_config/vae/config.json
"latents_mean": [
    -0.2289,
    -0.0052,
    ...
],
"latents_std": [
    0.4765,
    1.0364,
    ...
],
"num_res_blocks": 2,
"out_channels": 12,
"patch_size": 2,
"scale_factor_spatial": 16,
"scale_factor_temporal": 4,
"temperal_downsample": [
    false,
    true,
    true
],
"z_dim": 48
```

Transformer 主干（`lingbot_model_config/transformer/config.json`）：

```1:23:lingbot_model_config/transformer/config.json
{
  "patch_size": [
    1,
    2,
    2
  ],
  "num_attention_heads": 24,
  "attention_head_dim": 128,
  "in_channels": 48,
  "out_channels": 48,
  "action_dim": 30,
  "text_dim": 4096,
  "freq_dim": 256,
  "ffn_dim": 14336,
  "num_layers": 30,
  ...
}
```

代码里把 latent 打成 token 的部分（你正在看的 `model.py`）：

```635:646:wan_va/modules/model.py
self.patch_embedding_mlp = nn.Linear(
    in_channels * patch_size[0] * patch_size[1] * patch_size[2],
    inner_dim)
self.action_embedder = nn.Linear(action_dim, inner_dim)
self.condition_embedder = WanTimeTextImageEmbedding(
    dim=inner_dim,
    time_freq_dim=freq_dim,
    time_proj_dim=inner_dim * 6,
    text_embed_dim=text_dim,
    pos_embed_seq_len=pos_embed_seq_len,
)
```

把这些数字解释成"维度账"：

- **像素 → VAE latent**：对 RGB 帧（C=3）做 16× 空间 / 4× 时间下采样，通道升到 z=48。
  - 信息压缩率（仅看张量元素数）：`(3·T·H·W) / (48·T/4·H/16·W/16) = 3·16·16·4 / 48 = 64×`。
  - 即一个 latent voxel 浓缩了 4 帧 × 16×16 像素 × 3 通道（3072 个 raw scalar）→ 48 个连续 scalar。
- **VAE latent → Transformer token**：再做 (1,2,2) patch 化，单个 token 覆盖 `1·2·2 = 4` 个 latent voxel，所以 token 的"原始内容维"= 48×4 = 192，再线性映到 d_model = 3072。
- **token 内部 d_model = 3072，FFN = 14336**（≈ 4.67× 宽），head 维 128。
- **action 通道**：30 维连续向量，单独走 `action_embedder: 30 → 3072` 升到同一 d_model，与视频 token 共享 attention。
- **典型 256×256 输入、49 帧的 token 数**：
  - VAE 输出：T'=13, H'=W'=16，共 `13·16·16 = 3328` 个 latent voxel。
  - patch 后：每帧 `8·8 = 64` token，13 帧 → 832 个视频 token（再加 action token、text token）。

所以 lingbot-va 实际上**只用了一套 latent（Wan2.2 VAE，z=48），没有第二级显式 bottleneck**——视频和动作是直接在 d_model = 3072 的"骨干 latent"里联合建模的。这是综述里说的 **"Unified Stream + Implicit Future Prediction"** 路线（survey 1269 行表里 lingbot-va 被划到 "Multi-Stream: Cross-Attention / MoT"，因为它用 MoT 共享 attention，但 latent 的 substrate 仍是 z=48 的 Wan2.2 VAE）。

---

## 3. 同行对比：V-JEPA 2 / iVideoGPT / DiT4DiT 实际压到什么 shape

### 3.1 V-JEPA 2（implicit / 不重建像素）

- patchify：tubelet **2×16×16**（T×H×W）；
- 256×256 输入 → 每个 2-frame slice 给出 `16×16 = 256` 个 spatial token；
- encoder dim **D**：ViT-L=1024，ViT-H=1280，ViT-g=1408，ViT-G=1536；
- DROID 后训练（论文里写得很明确）：每帧独立编码，得到 feature map `z_k ∈ R^{16×16×1408}`（ViT-g）；
- **预测器宽度 pred_hidden_size = 384**（HF 默认），比 encoder 小 ≈ 4 倍——这是个非常关键的设计：预测器走的是"窄瓶颈"，让动力学在低维流形里学，**避免把容量花在像素级 nuisance 上**。

V-JEPA 2 给我们一个"高分辨率视频最终压到什么 shape"的具体答案：
**256×256 输入 × 16 帧 → 16×16×D 的 latent grid（无时间维度合并，仅按 tubelet=2 折半）**。

### 3.2 iVideoGPT（compressive VQ + GPT）

- ctx_vqgan：第一帧（context）用完整 256+1 token，**后续每帧只用 16 个 dynamic token**；
- 渐近压缩率 ≈ **16× token 削减**（survey 也提到，论文图中描述）；
- 词表大小 = `num_vq_embeddings + num_dyn_embeddings` (+ 2 if special_token)；
- 高分辨率（256×256）tokenizer 自身参数量从 64×64 的 114M 涨到 310M，但 Transformer 骨干仍是 138M / 436M。

iVideoGPT 的特点：**用"前一帧条件下的差分"做 VQ**，绕过了"每帧都要重新编码全部空间内容"的浪费。这是离散路线对应 lingbot-va 那种"连续 z=48"路线的另一种处理。

### 3.3 DiT4DiT（cascaded：video DiT 中间 hidden 作为 action 条件）

paper 2603.10448 的核心 latent 设计：

> "we repurpose the DiT as a feature extractor: a forward hook mechanism intercepts intermediate hidden activations in flow timestep τ_f ... converting the generative process into rich visual tokens for downstream tasks."

- VAE 用 Cosmos-Predict2.5 的 causal VAE（spatial+temporal 下采样，作者没改），与 lingbot-va 同属 "用一个强 VAE 做 substrate"；
- 关键的 latent 不是 VAE z，而是 video DiT 在 **特定 flow timestep τ_f** 的 hidden state h_t^{τ_f}，**维度就是 DiT 的 d_model**；
- 这个 hidden 通过 cross-attention 注入 Action DiT，本身不做额外瓶颈压缩；
- 选什么 τ_f 是"维度大小"之外的另一个 knob：early stage 强调全局结构，late stage 强调细节。

DiT4DiT 没显式做"狭义瓶颈"，它的"维度选择"等价于选择 **DiT 哪一层 + 哪一个 τ_f** 抽 hidden。这一点很重要：在 DiT 这类大模型里，**"latent dim"已经分裂为"特征 dim × 层数 × 噪声水平"三维 knob 空间**。

---

## 4. 衡量"维度大小"的几条可操作准则

把上面这些放在一起，下面列的就是你应该用来**判断一个 WAM latent 维度是不是合理**的几个尺子（综述 2605.12090 的 §4.1.2 / 7 Open Challenges 也提到过这些权衡）：

### 4.1 重建—压缩比（information capacity bound）

对 VAE-based latent（lingbot-va / DiT4DiT / VPP）：

\[
\text{capacity ratio} = \frac{3 \cdot T \cdot H \cdot W \cdot 8\text{bit}}{\text{z\_dim} \cdot T' \cdot H' \cdot W' \cdot 16\text{bit (bf16)}}
\]

lingbot-va：≈ `3·16·16·4·8 / (48·16·2) = 24576/1536 = 16×` 数据率压缩。

如果你把 z_dim 从 48 砍到 16（仍 Wan2.2 类配置），等于 64× 压缩——**通常重建 PSNR 直接掉 3–6 dB**，对手部、夹爪边缘这种细物体识别影响最大。这就是 survey 里说的"pixel-matching shortcut" 反面：**容量不足** → 模型连基本物体边界都建模不准，下游动作精度受损。

### 4.2 任务相关信息密度（task-relevant bits per token）

如果是 implicit / JEPA 路线（V-JEPA 2、DexWorldModel、LDA-1B）：你不重建，所以不能用 PSNR 衡量。要看 **下游 probe 的可分性**：

- 同样 d_model（如 384 vs 768 vs 1024），用一个固定 probe（线性 / 浅层 MLP）跑下游动作 / 物理预测准确率；
- V-JEPA 2 的设计直觉是 "encoder 1408 大、predictor 384 窄"——**预测维度小一点反而泛化更好**，因为强迫动力学忽略 nuisance；
- DexWorldModel / LDA-1B 把 latent 从 VAE z 换成 **DINOv3 ViT-s 特征**（dim 通常 ~384），就是为了拿一个"语义 dense, 物理无关的细节稀疏"的表示，对"未见过物体"的鲁棒性更好。

### 4.3 token 数 × d_model 的有效 FLOPs（compute budget）

骨干 attention 复杂度 ∝ `N²·d`，N 是 token 数，d 是 d_model。
- lingbot-va 256² 输入：`N ≈ 13·8·8 = 832` 视频 token，d=3072，`832²·3072 ≈ 2.1G` 算子级。
- 把空间 patch 从 (1,2,2) 升到 (1,4,4) → N 减 4×，attention 减 16×，**但每个 token 要建模 4× 信息量**。
- 关键 trade-off：**"d_model 升 1.5× 比 N 升 2× 更划算"**（V-JEPA 2 / Wan2.2 等都遵守这个经验），所以瓶颈层放在通道维去做更高效。

### 4.4 模态对齐尺度（cross-modal compatibility）

WAM 一定要把视频 latent 和 action latent 投到同一 d_model 才能一起做 attention。lingbot-va 是 30 → 3072（升维），动作天然稀疏，**这一升维会引入大量空白维度**。要么:
- 用 action chunking 增加 action token 数（lingbot-va `action_per_frame=8`）减小这个 mismatch；
- 要么用 MoT（Mixture of Transformers）让 action 专家有独立 d_model_a < d_model_v（这是 LingBot-VA paper 主张的 "dual-stream MoT"）。

---

## 5. 关于"瓶颈维度对未见过物体泛化"的实验设计

你想做 ablation：**不同 z_dim / bottleneck 维度 → 在 unseen-object 评测上的表现**。综合上面材料，给你一个最小可控的实验骨架：

| 实验组 | 改动 | 预期方向 |
|---|---|---|
| A. baseline | 现在的 z=48, d_model=3072, patch=(1,2,2) | 参考 |
| B. 缩 z 通道 | 重 VAE 训练或用 16-ch / 8-ch / 4-ch VAE | 重建 PSNR ↓；FID ↑；unseen-object **可能反而 ↑**（更抽象） |
| C. 缩 d_model | 3072 → 1536 → 768，layers 不变 | 训练曲线变 noisy；OOD 可能波动 |
| D. 加二级显式瓶颈 | 在第 N 层后面加 `Linear(d_model → d_bot → d_model)`，d_bot ∈ {128, 384, 768} | "信息瓶颈"假说：d_bot 适中时 OOD 最好 |
| E. JEPA-style 替换 | 用 DINOv3-ViT-s（dim 384）替换 Wan VAE 做 latent substrate | 像 LDA-1B / DexWorldModel：unseen-object 泛化往往更好，但视频重建质量大幅 ↓ |
| F. compressive VQ | 像 iVideoGPT 一样让"后续帧"只编 16 token | token 数大幅 ↓；适合长序列 OOD 验证 |

评测维度（按 survey §6 整合）：

- **可见性指标**：PSNR / FVD / LPIPS（看 latent 容量够不够）；
- **语义对齐**：DINO 相似度 / DreamSim（看 latent 是否抓到了对象语义而不是表面纹理）；
- **物理常识**：VideoPhy / PhyGenBench（看 latent 是否丢掉了关键物理动力学）；
- **OOD 动作成功率**：在 LIBERO-Spatial / Object / Goal、RoboCasa-GR1、以及你内部 unseen-object split 上跑成功率；
- **DiT4DiT 提到的 "scaling proxy" 视角**：同等 trainable 参数下，看收敛速度（1× vs 7×）和数据效率（1× vs 10×）的曲线斜率。

操作上，因为 lingbot-va 把 VAE 当成"冻结 substrate"（参考 `LINGBOT_VA_ARCHITECTURE.md` §2.2），最便宜的 ablation 是 **D 组（加显式 bottleneck 层）**，可以直接在 `WanTransformerBlock` 之间塞一个 down-project + up-project，不需要重训 VAE：

参考插入点（仅作 ablation 设计，不要立刻改主干）：

```648:655:wan_va/modules/model.py
self.blocks = nn.ModuleList([
    WanTransformerBlock(inner_dim,
                        ffn_dim,
                        num_attention_heads,
                        cross_attn_norm,
                        eps,
                        attn_mode=attn_mode) for _ in range(num_layers)
])
```

在第 N 层（如 15/30）后面插入一个 `Linear(3072 → d_bot → 3072) + LayerNorm` 当显式 bottleneck，只对这一段做 LoRA-style 微调；d_bot 扫 {128, 256, 384, 768, 1536}，看 OOD 成功率的 U 形或单调形态。

---

## 6. 关于"DiT4DiT 与 lingbot-va 思考的根本不同"

把两者并排看（这是 survey §4.2 / 表 3 给的归类）：

- **lingbot-va**（Unified-ish, MoT, 显式自回归）：
  - latent substrate：Wan2.2 VAE z=48；
  - 视频和动作在**同一个 d_model=3072 的骨干**里联合 denoise，靠 FlexAttn 的 mask 解耦因果；
  - 优点：训练目标"对称"，KV cache 适合长 horizon；
  - 弱点：z=48 这一选择把"上限"钉死在 VAE 重建质量上，"未见过物体"如果 VAE 重建本来就糟，下游一起糟。
- **DiT4DiT**（Cascaded, Hidden-State Coupling）：
  - latent substrate：Cosmos VAE z；
  - 但 **action 只看 DiT 第 X 层、第 τ_f 时刻的 hidden**，不直接吃 VAE z；
  - 优点：hidden 比 z 语义更"形而上"，对外观抖动不敏感，paper 报告 OOD（unseen objects / categories / quantities）零样本泛化更强；
  - 弱点：需要在训练时同时维护两个 DiT，inference 还要走一次 video DiT forward 拿 hidden。

两条线 latent 维度的本质区别其实是：

> **lingbot-va 把 latent 选在"原始数据压缩"的那一层；DiT4DiT 把 latent 选在"生成模型已经做完物理推理"的那一层。**

后者的维度天然更大（d_model 量级，比 z_dim 大 1–2 个数量级），但**信息密度对动作更有用**——这也是为什么 DexWorldModel / LDA-1B 都进一步把 latent 切换到 DINOv3 语义空间。

---

## 7. 给你的"latent dim 思考决策树"

下次评一篇 WAM 的 latent 维度，建议按这个顺序问：

1. **它在哪一层叫 latent？** VAE 通道 / token 网格 / DiT hidden / JEPA embedding——四种不能比。
2. **重建是 hard constraint 还是 soft？** 要重建（lingbot-va、Cosmos Policy、DreamZero）→ z 不能太小；不要重建（V-JEPA 2、DexWorldModel）→ 可以猛压。
3. **token 数 × d_model 的预算账平了吗？** 不要为了"看起来"小 z 而把 d_model 撑到 4096——下游 attention 才是真瓶颈。
4. **下游 action 是怎么吃 latent 的？** Cross-attention（DiT4DiT、CoVAR）对 latent dim 容忍度高；直接 concat 进 token 序列（lingbot-va）需要严格对齐。
5. **泛化目标是什么？** 看 unseen object → 偏好"语义 dense + 像素 sparse"（DINO 类）；看长 horizon → 偏好可压缩、抗误差累积的 latent（V-JEPA predictor 384 的取舍）。
6. **算力预算？** 同样 token 数下，d_model × layers 决定真实 FLOPs。survey §7 "Inference Latency" 提到 DreamZero 7Hz vs 50Hz VLA 的差距主要就出在这。

---

## 8. 直接结论（针对你当前 lingbot-va）

把所有事实落到 lingbot-va 这个代码库上：

- **当前 4 个维度**：z=48 / token 网格 ≈ (T/4, H/32, W/32) / d_model=3072 / FFN=14336 / 30 层；属于"中等偏宽"的 Wan2.2 路线，跟 GigaWorld-Policy、X-WAM、AIM 同档；
- **"未见过物体"泛化弱**多半不出在 z=48 本身（这个容量比 4-ch SD 的 z=4 大太多了），而出在：
  1. 训练数据里 unseen 物体的 VAE 表征分布不在 latents_mean/std 里（看 `vae/config.json` 里的 mean/std 是 robotic 域 fine-tune 过的）；
  2. 30 层 DiT 太容易"记住"训练集的纹理 shortcut。
- **最低风险的实验**：在 DiT 中段加 d_bot ∈ {128, 384, 768} 的显式 bottleneck（D 组），对照 OOD 成功率；
- **如果允许换 substrate**：参考 DexWorldModel / LDA-1B，给 transformer 加一个 DINOv3-ViT-s（384 维）的 **辅助 latent prediction loss**，让骨干在 z=48 之外学一个语义 latent；这是综述里被反复点名"对 unseen object 提升最明显"的做法。

需要的话我可以把 D 组的 bottleneck 层和评测脚本直接落到 `wan_va/modules/model.py` 里——你只要说改不改 / 哪里改即可。

参考过的相关历史会话：[WAM survey 与 V-JEPA / iVideoGPT 比较](f4e170ca-72ec-4a88-9257-c2bc0ab5c248)（如果有）。