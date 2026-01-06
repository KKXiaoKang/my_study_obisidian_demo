![[Alpamayo-R1_Bridging Reasoning and Action Prediction.pdf]]


# KeyNote
* 训练分为4个阶段
	* ![[Pasted image 20260106141534.png]]
* VLM training :  使用370万个视觉问答（VQA）样本在VLM中后训练，让Cosmos-Reason VLM 模型具备一定的物理常识和具体推理能力（在原本训练好的VLM当中进行post-train， 让其具备更多的物理常识和具体推理能力）
	*  Cosmos-Reason is a VLM specifically designed for Physical AI applications, post-trained on 3.7M Visual Question Answering (VQA) samples to develop physical common sense and embodied reasoning capabilities

* VLM training: 使用24700个专注于驾驶场景的精选视频 VQA样本，包含场景描述，驾驶难度注释以及从DeepSeek-R1 中提取的推理轨迹及预测下一步动作
	* The model incorporates 24.7K curated video VQA samples focused on driving scenarios, including scene descriptions, driving difficulty annotations, and reasoning traces distilled from DeepSeek-R1 (DeepSeek-AI, 2025) to predict the next action

### vision encoder 
* Domain-Specific Adaptations ： 特定领域的小组件升级
	* Vision Encoding ：目标机载部署，所以在vision token中要生成尽可能少的token
		*  Single-Image Tokenization
			* 传统 SigLip2 （2025）
			* ReWiND （2023）
		*  Multi-Camera Tokenization
			* 参考: [Efficient Multi-Camera Tokenization with Triplanes for End-to-End Driving](https://arxiv.org/abs/2506.12251)
		*  Multi-Camera Video Tokenization 多相机视频编码
			*  参考: [Towards Efficient and Effective Multi-Camera Encoding for End-to-End Driving](https://www.arxiv.org/pdf/2512.10947)  

### action decoder
* 对于VLA reasoning traces 来说，
	* 哪些 reasoning traces 是低信息密度的 - 黄色标识
	* 哪些 reasoning traces 是和当前自身行为相反或者冲突的 - 红色标识
	* 哪些 reasoning traces 是思考的的不够深刻 - 蓝色标识
	* ![[Pasted image 20260106161603.png]]
* CoC labeling pipeline :
	* ![[Pasted image 20260106161312.png]]
	* 1 ) Clip 剪辑选择：思维链裁剪
		*  针对刚刚 reasoning traces  的一些情况，选择包含明确驾驶决策的剪辑，过滤掉提供有限并且低信息密度 因果信息
	* 2) Keyframe Labeling：关键帧标签
		* 识别每个视频剪辑中的决策时刻，最大限度地减少潜在的因果混乱
	* 3) - 5) Structured CoC Labeling
		* 结构化 CoC 标签，为了构建最终的 CoC 并进一步减轻因果混乱，我们首先注释观察中的关键组件，同时避免引用未来帧中的因果因素，然后标记相应的驾驶决策。然后，我们用自然语言从驱动决策和因果因素中构建推理轨迹。

## 阶段3：Chain of Causation Dataset: Learning Causally Grounded Reasoning VLAs - 从因果链数据集当中构建VLA
### labeling framework - 因果链标注框架
#### 结构化的标注格式 - Structured Chain of Causation
* the driving decision - 驾驶决策
* the causal factors (critical components) - 因果关系组成部分
* the composed CoC trace : 思维链
*  CoT的最终结果必定是结构化的driving decision ： 包含了纵向选择 和 横向选择
##### driving decision - 驾驶决策
* ![[Pasted image 20260106164626.png]]
#####  因果关系组成部分
* ![[Pasted image 20260106165612.png]]

##### 因果链：确定了因果关系和驾驶决策后，就可以结构化思维链输出了
![[Pasted image 20260106165910.png]]


### Hybrid Labeling Procedure - 混合标签注入
#### 只在决策时刻时介入标注CoT（给定一些元动作，当元动作发生变换时该帧作为keyframe进行标注）
* driving decisions (驾驶决策) - 下图中使用绿色标识  
* critical components (因果组成) - 下图中使用蓝色标识  
* CoT标注示例如下
	* ![[Pasted image 20260106170607.png]]
* Meta action 元动作示例：
	* ![[Pasted image 20260106172836.png]]
#### Labeling Procedure
* **Next, we employ state-of-the-art VLMs such as GPT-5 (OpenAI, 2025) to perform offline auto-labeling through a multi-step reasoning process.** 
* 为了减轻因果混乱，在识别关键组件时，系统会提示 VLM 使用 2 秒历史视频。随后的 6 秒未来视频以及自我的轨迹和元动作，然后用于解决多模态问题并确定相应的驾驶决策。在此过程中，模型对已识别的因果因素进行重要性排序，并在最终推理轨迹中仅保留那些直接影响驾驶决策的因素。


## 训练策略
### 阶段1 pre-training VLM 
*  参考KeyNote上的方法
### 阶段2 pre-training VLA  
* 训练时参考Physical Intelligence的做法，把KV Cache的梯度进行截断，防止梯度传播到VLM当中
	* **![[Pasted image 20260106175640.png]]**
*  动作专家使用普通的flow-matching head loss，本质上确实是MSE loss Flow Matching (L2回归)
	* $$L_{cfm}(\Theta) = \mathbb{E}{t \in P{schedule}, (\mathbf{o}, \text{REASON}) \in \mathcal{D}{data}} || \mathbf{v}{\Theta}(\mathbf{a}t, \mathbf{o}, \text{REASON}) - \mathbf{u}(\mathbf{a}_t | \mathbf{a}) ||$$
	* 核心特点：

		1. 目标：学习速度场（velocity field）

			- 模型预测速度向量 $\mathbf{v}{\Theta}$，而非直接预测动作

			- 通过速度场从噪声逐步生成动作轨迹

		2. 输入结构：

			- $\mathbf{a}t$：加噪后的动作（训练时从噪声到真实动作的中间状态）

			- $\mathbf{o}$：观察（视觉输入）

			- $\text{REASON}$：推理文本（可能来自预训练或简单生成）

		3. 训练方式：

			- 使用时间步采样 $t \in P_{schedule}$（Flow Matching的标准做法）

			- 学习从任意噪声状态 $\mathbf{a}t$ 到真实动作 $\mathbf{a}$ 的速度方向


### 阶段3 CoT SFT 监督微调 - 语言模型损失 数学关系：负对数似然 = 交叉熵（在分类问题中）：
Loss函数：

$$\mathcal{L}{\text{SFT}}(\theta) = -\mathbb{E}{(o, \text{REASON}, a) \sim \mathcal{D}{\text{CoC}}} [\log \pi{\theta}(\text{REASON}, a \mid o)]$$

核心特点：

1. 目标：最大化联合生成概率， 负对数似然 (分类)

- 同时生成结构化的推理 $\text{REASON}$ 和动作 $a$

- 通过最大化对数似然来学习

1. 输入结构：

- $o$：观察（视觉输入）

- 输出：$\text{REASON}$ 和 $a$ 的联合序列

1. 训练方式：

- 使用高质量的CoC数据集 $\mathcal{D}{\text{CoC}}$（包含因果链推理）

- 标准的自回归语言模型训练（token-by-token生成）

1. 损失类型：负对数似然（分类损失，离散token空间）


### 阶段4 - Post-Training Algorithm RL 后训练
*  目标是评估三类
	*  Reasoning 质量（推理本身是否合理）（通过大型推理模型反馈）
	*  Action 质量（行为是否合理、安全）
	*  推理-动作一致性（最核心）
*  使用GRPO作为对齐算法

	在 post-training 阶段，目标是：
	
	- 提升推理质量（reasoning quality）
	
	- 增强推理-动作一致性（reasoning-action consistency）
	
	- 优化轨迹质量（trajectory quality）
	
	传统 RL 方法（如 PPO）的问题：
	
	- 依赖绝对奖励信号，可能不稳定
	
	- 奖励函数可能有噪声或偏差
	
	- 容易过度优化，偏离 SFT 模型学到的先验
	
	GRPO（Grouped Relative Policy Optimization）的核心思想：
	
	- 不依赖绝对奖励，而是优化组内相对优势
	
	- 通过组内对比，更稳定、更鲁棒
	
	- KL 正则化防止过度偏离 SFT 模型
	
	根据论文和图片描述，GRPO 的目标函数为：
	
	$$\mathcal{L}{\text{GRPO}}(\theta) = -\mathbb{E}{\tau_i \sim \pi_\theta} \left[ \frac{\exp(\beta A_i)}{\sum_j \exp(\beta A_j)} \left( \log \pi_\theta(\tau_i) - \lambda_{\text{KL}} \text{KL}[\pi_\theta(\tau_i)||\pi_{\text{ref}}(\tau_i)] \right) \right]$$
	
	其中相对优势定义为：
	
	$$A_i = r_i - \bar{r}$$
	
	### 公式各部分解析
	
	#### 1. 相对优势 $A_i = r_i - \bar{r}$
	
| 符号                                      | 含义           | 作用                     |
| --------------------------------------- | ------------ | ---------------------- |
| $r_i$                                   | 第 $i$ 个轨迹的奖励 | 由奖励模型（如大型推理模型critic）给出 |
| $\bar{r} = \frac{1}{K}\sum_{j=1}^K r_j$ | 组内平均奖励       | 归一化基准                  |
| $A_i$                                   | 相对优势         | 衡量该轨迹在组内的相对表现          |

要点：

- 使用相对优势而非绝对奖励，减少奖励尺度影响

- 组内归一化使训练更稳定

#### 2. 权重项 $\frac{\exp(\beta A_i)}{\sum_j \exp(\beta A_j)}$

这是 Softmax 权重：

- $\beta$：温度参数，控制权重分布的尖锐程度

- $\beta \to 0$：权重趋于均匀

- $\beta \to \infty$：只关注最优轨迹

作用：

- 为表现更好的轨迹分配更高权重

- 通过组内归一化，自动适应奖励尺度

#### 3. 策略项 $\log \pi_\theta(\tau_i)$

- $\pi_\theta(\tau_i)$：当前策略生成轨迹 $\tau_i$ 的概率

- $\log \pi_\theta(\tau_i)$：对数概率，用于策略梯度

#### 4. KL 正则化项 $\lambda_{\text{KL}} \text{KL}[\pi_\theta(\tau_i)||\pi_{\text{ref}}(\tau_i)]$
	
| 符号                                        | 含义               |
| ----------------------------------------- | ---------------- |
| $\pi_{\text{ref}}$                        | 参考策略（通常是 SFT 模型） |
| $\text{KL}[\pi_\theta\|\pi_{\text{ref}}]$ | KL 散度，衡量策略偏离程度   |
| $\lambda_{\text{KL}}$                     | 正则化系数            |

	作用：
	- 防止过度优化有噪声/偏差的奖励
	- 保留 SFT 阶段学到的语言和行为先验
	- 保持模型稳定性和可解释性

-   Grading Reasoning with Large Reasoning Models
	-  使用大型推理模型对推理进行评分。
- Reasoning Critic Design.
	-  推理评论家设计
	-  对于每个训练样本，LRM 批评家将 2 秒历史窗口最后一帧的多摄像机视觉观察图像、数据集中的真实 CoC 推理轨迹 ReasonGT 以及当前策略 πθ 产生的模型生成推理轨迹 Reasonpred 作为输入。评论家从两个维度评估 Reasonpred 与 ReasonGT 的一致性：行为一致性，预测推理是否描述了与地面事实一致的驾驶决策；因果推理质量，是否根据 CoC 原则正确识别场景历史中可观察到的因果因素（第 4.1 节）。批评者根据关注行为一致性和因果推理一致性的结构化评分标准对预测推理进行评分：
	- 参考prompt
		- ![[Pasted image 20260106203134.png]]