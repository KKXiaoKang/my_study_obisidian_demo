

下面是我针对 **VLA 模型 SFT 之后再用 RL 微调（尤其面向实物机器人）** 整理出来的近一年最值得读的论文。考虑到 `flow_matching_action_head.py` 用的是 **flow matching 生成动作**，我把跟 flow matching / diffusion / consistency policy 直接相关的工作单独标出来。

---

## 一、为什么 VLA 需要 RL（动机共识）

近年这几篇都给了非常一致的判断：

- **SFT 是模仿，是"误差累积"的源头**：动作分布外 (OOD) 状态下表现退化；
- **RL 解决补偿/纠错/长程规划**：失败可被显式惩罚，能学到 demo 里没有的"补救动作"；
- **RL 的代价**：实物代价高、不安全、不稳定 → 各论文都在解决"怎么便宜、怎么稳"的问题。

NeurIPS 2025 的 [What Can RL Bring to VLA Generalization?](https://rlvla.github.io)（清华 Yi Wu 组）给了实证结论：
> **PPO 比 DPO/GRPO 更适合 VLA**；RL 提升的主要是**语义理解**和**执行鲁棒性**，而非视觉鲁棒性。

可以作为读其他论文前的"路标"。

---

## 二、与你 GROOT (flow matching action head) 最直接相关

这一类是首选。

### 1. ReinFlow（NeurIPS 2025）—— 专门给 flow matching policy 做在线 RL

- arXiv: [2505.22094](https://arxiv.org/abs/2505.22094)，主页：[reinflow.github.io](https://reinflow.github.io/)
- **核心思路**：在 flow 的确定性 ODE 路径上**注入可学习的高斯噪声**，把 flow 转成离散时间 Markov 过程 → 可以**精确算 log-likelihood** → 可以直接用 PPO 类的策略梯度。
- 微调完后**丢掉噪声网络**，恢复原本的快速 ODE 推理。
- 在 Rectified Flow / Shortcut Model 上，**1~4 步去噪**就能 RL 微调；机械臂任务平均成功率 +40%，腿足 +135%，比 DPPO 省 82% 时间。
- 为什么值得你看：你 `flow_matching_action_head.py` 第 234 行附近的 sampling/inference 路径（ODE 积分），跟 ReinFlow 假设的"确定性 velocity field"是同一类，**最容易直接套用**。

### 2. π* 0.6 / Recap（Physical Intelligence，2025-11）—— flow-matching VLA 真正"从经验中学"

- arXiv: [2511.14759](https://arxiv.org/abs/2511.14759)
- Physical Intelligence (π0/π0.5 团队) 的最新工作，**显式声明支持 flow matching / diffusion 类 action head**。
- 流程：SFT → 自主+人介入采集数据 → 训一个 value function → 计算 advantage → **advantage-conditioned 重新训 π***。
- 工程上很务实：对实物机器人允许"专家偶尔接管纠错"（DAgger 式），并能**同时利用好 demo + bad demo**。
- 开源 RL pipeline 已被人复刻：[Safe-Sentinel-Inc/reinforcement_learning_for_vlas](https://github.com/Safe-Sentinel-Inc/reinforcement_learning_for_vlas)（基于 openpi）。

### 3. ConRFT（RSS 2025）—— 真机 + Consistency Policy + HIL

- arXiv: [2502.05450](https://arxiv.org/abs/2502.05450)，代码：[cccedric/conrft](https://github.com/cccedric/conrft)
- 两阶段：
  - **Cal-ConRFT（离线）**：BC + Q-learning 联合损失，从少量 demo 同时学策略和价值函数
  - **HIL-ConRFT（在线）**：CPQL + **人类干预** (Human-in-the-Loop)，保证安全和样本效率
- **8 个真实机械臂任务，45–90 分钟**就能从 ~40% 上到 **96.3% 平均成功率**，比纯 SFT +144%。
- 一致性策略 (consistency model) 可以视为 flow matching 的近亲，借鉴价值很高。
- 底座是 RAIL-Berkeley 的 **HIL-SERL** ([rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl))，这套基础设施是真机 RL 当下最成熟的开源框架。

---

## 三、纯 RL 后训练 VLA（不一定是 flow matching，但思想可借鉴）

### 4. RIPT-VLA（ICLR 2026 在审）—— "第三阶段：交互式后训练"

- 论文：[OpenReview PDF](https://openreview.net/pdf/f41f95ec4c28c5ad1fb3293ad55e5e6df1a38300.pdf)
- 概念清晰：**Pretrain → SFT → RL Interactive Post-Training**，仅需**稀疏二元成功奖励**。
- 算法：扩展 LOOP 框架（REINFORCE leave-one-out + dynamic rollout sampling），不需要 critic、不需要 reward shaping。
- 数据效率惊人：**1 条 demo + 15 轮 RL 迭代**，把一个 4% 成功率的 SFT 模型推到 **97%**。
- 适合你：奖励工程几乎不用做，只要能判断"成功/失败"。

### 5. SimpleVLA-RL（2025-09）—— 工程友好的大规模 VLA-RL 框架

- arXiv: [2509.09674](https://arxiv.org/abs/2509.09674)
- 基于字节的 **veRL**（LLM RL 基建）改的，专门给 VLA 做 trajectory sampling、并行 rendering、loss 计算优化。
- 在 OpenVLA-OFT 上把 LIBERO-Long 从 17.1% 拉到 91.7%（**仅一条 demo 每任务**）。
- 提出"**pushcut**"现象：RL 学到了 demo 里没有的新策略——这正是 SFT 上限的天花板。
- **sim-to-real 也跑通了**：仿真训完直接迁移到真机，平均 17.5% → 38.5%。
- 推荐场景：你有仿真器，想低成本探索"RL 能不能给我 GROOT 带来增量"。

### 6. iRe-VLA（ICRA 2025）—— RL 与 SL 交替，防止崩坏

- arXiv: [2501.16664](https://arxiv.org/abs/2501.16664)
- 直接观察到的痛点：**大 VLA 直接做在线 RL 会训崩**。
- 解法：**RL 阶段** 用环境奖励探索，**SL 阶段** 把成功 trajectory 加回来回放 SFT，循环往复。
- 在仿真 + 一套真机 manipulation 上验证。
- 适合你：只想要一个稳定可工作的 baseline，不追求 SOTA。

### 7. VLA-RFT（ICLR 2026 在审）—— 用 world model 当模拟器

- arXiv: [2510.00406](https://arxiv.org/abs/2510.00406)，主页：[vla-rft.github.io](https://vla-rft.github.io/)
- 思路：训一个**视觉世界模型**（输入动作 → 预测下一帧）当 controllable simulator，然后用 **GRPO** + **dense trajectory-level verified reward** 微调 VLA。
- **<400 步**就超过强 SFT baseline；不依赖真机交互、也不依赖外置仿真器，完全用真实数据训出来的 world model 兜底。
- 适合你：真机不能开放跑、又没有匹配的仿真，但有大量真机回放数据。

### 8. LifeLong-RFT（2026-02）—— 不依赖环境反馈的多维过程奖励

- arXiv: [2602.10503](https://arxiv.org/abs/2602.10503)
- 解决"持续学习场景下 SFT 灾难性遗忘"。
- 用三种过程奖励（量化动作一致性 + 连续轨迹对齐 + 格式合规）替代环境奖励，纯离线就能跑。
- LIBERO 持续学习上比 SFT +22%，**只用 20% 数据**。
- 适合你：**没有奖励信号、没有仿真**，只想从历史轨迹挖更强信号。

---

## 四、按你"实物机器、SFT 后再 RL"这个具体诉求的推荐路线

| 你的现状                                    | 推荐组合                                    | 理由                                                                                             |
| --------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 已有 GROOT (flow matching) + 真机 + 少量 demo | **ConRFT + ReinFlow**                   | ConRFT 给真机 RL 的工程框架（HIL/HIL-SERL 底座），ReinFlow 提供"flow matching policy 怎么算 likelihood、怎么 PG"的理论 |
| 已有真机 + 想最小工程量                           | **RIPT-VLA 思路 + HIL-SERL 框架**           | 只要写一个"成功/失败"判定，剩下交给 LOOP/PPO                                                                   |
| 真机不安全/贵，但有大量回放数据                        | **VLA-RFT 思路**                          | 训 world model + GRPO，全程不上真机                                                                    |
| 想跟齐工业界最先进                               | **π* 0.6 / Recap** + Safe-Sentinel 开源实现 | flow-matching 原生友好，Physical Intelligence 在真机上已验证规模化可行                                          |
| 仿真够好，想做 sim-to-real                     | **SimpleVLA-RL**                        | 工业级训练基建，已展示 sim-to-real 增益                                                                     |

---

## 五、几条工程上务必注意的事（这些论文里反复强调）

1. **奖励要么二元 (RIPT-VLA)、要么靠 world model verified (VLA-RFT)、要么靠人介入 (ConRFT)**——不要徒手做 reward shaping。
2. **要给探索加噪**：flow/diffusion 是确定性的，不加可学习噪声会"卡住"（ReinFlow 的核心 motivation）。
3. **PPO > GRPO/DPO**（针对 VLA，[NeurIPS 2025 RLVLA](https://rlvla.github.io) 经验结论），但 GRPO 在 verified reward 场景仍很常用 (VLA-RFT)。
4. **必须用 KL 约束/SL 回放**防止 policy collapse（iRe-VLA 给出的对照实验非常清楚）。
5. **真机一定要 HIL-SERL 或等价框架**，不要让模型自己乱探索机械臂——会撞坏。

---

## 一句话总结

最贴你这套 **GROOT + flow matching action head + 实物机器** 的"先 SFT 再 RL"路线，最值得入手的三篇是：

> **ReinFlow（NeurIPS 2025）** ——给 flow matching 提供 RL 微调的理论框架；
> **ConRFT（RSS 2025）+ HIL-SERL** ——给真机提供工程框架（人介入、安全、45 分钟收敛）；
> **π\*0.6 / Recap（Physical Intelligence, 2025-11）** ——给 flow-matching VLA 提供工业级"从经验中学"的范式。

如果你想，我可以基于这三篇画一张更细的"GROOT 现有 flow matching 头 → 改造成可 RL 微调"的改造路线图，标出在 `flow_matching_action_head.py` 中具体哪几个函数（如 `sample_actions` / `compute_loss` / velocity field forward）需要改、改成什么样。要我做吗？