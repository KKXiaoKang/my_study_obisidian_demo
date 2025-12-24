
![[Pasted image 20250625165905.png]]
## 数学建模公式
### Stochastic Langevin Dynamics - 随机朗之万动力学
* DDPMs are a class of generative model where the output generation is modeled as a denoising process, often called `Stochastic Langevin Dynamics` Welling and Teh
* 
## pipeline核心思想
#### 🔹 (a) Diffusion Policy General Formulation
* 扩散策略的总体流程
	* `输入`：观察序列（Observation Sequence）
		* 图像序列+机器人状态
		* $\mathcal{O}=(o_{t-T}^{obs}, ..., o_{t}^{obs})$ 
		* 每个观测$o_t$ 通常包含
			* RGB图像
			* 机器人的手臂位姿（末端eef_pose）
	* `Diffusion Policy` 结构
		* 核心模型为扩散网络
		* $\epsilon_\theta(\mathcal{O},\, A,\, k)$ 
		* 试图从带噪动作序列A中`去噪`回真实动作
		* $k$ 是当前扩散迭代步(denoising step)
	* `输出` ：动作序列 
		* 动作序列公式
			*  $A_t = {a_t,a_{t+1}, ..., a_{t+T_{p}}}$ 
		* 表示在未来$T_p$步的期望动作
	* 输出形式通常为末端执行器位置的变换，抓取控制信号等 

#### 🔹 (b) CNN-based Diffusion Policy（用 1D CNN 建模）
 * 这种方式利用 **FiLM（Feature-wise Linear Modulation）特征级线性调制**机制进行条件建模
 * 流程说明
	 * $\mathcal{O}_t$经过编码，调制网络行为
	 * 动作嵌入序列（带噪版本），通过 Conv1D 网络逐层去噪
	 * **FiLM** 用观察来调节每一层中的卷积运算
		 * $x \rightarrow a \cdot x + b$    
		 * $a,b$ 是由`Observation`编码器生成的调制参数
		 * 对动作序列的每一维都进行调制（条件生成）

#### 🔹 (c) Transformer-based Diffusion Policy
* 使用`Transformer`有更强的建模能力，适合高维观测和复杂空间动作
* 流程说明
	* 将观测$\mathcal{O}_t$ 和动作序列$\mathcal{A}$ 分别编码为`Embeddings`
	* 使用`Cross-Attention`模块
		* 动作嵌入作为`Query`
		* 观测嵌入作为`Key/Value`
		* 学习一个跨模态的注意力机制，条件化动作序列
	* 每一步都重复$K$ 次迭代（即扩散过程的步数） 

## 🔷 什么是 E(At)E(A_t)？

在扩散模型（包括 Diffusion Policy）中，$E(A_t)$ 是一个`“能量函数”或目标函数`，代表当前动作序列 $A_t$ 相对于条件观测 $\mathcal{O}_t$ 的“合理程度”或“似然”。

### 从扩散模型角度看：
- 在训练时，我们从 **真实动作序列** 加噪声得到带噪动作 $A_t^{(k)}$
- 网络学习如何预测**噪声 $\epsilon$ 或恢复**去噪动作**
- E(At)E(A_t) 表示当前动作的“质量评分”或“偏离真实数据的程度”
    
## 🔶 在 CNN-based Diffusion Policy 中，E(At)E(A_t) 的函数意义
- **函数表示**：$E(A_t) = \| \epsilon_\theta(\mathcal{O}_t, A_t^{(k)}, k) - \epsilon \|^2$ 
- **物理含义**：预测噪声 $\hat{\epsilon}$与真实噪声 $\epsilon$ 之间的差距
- **实现方式**：
    - 网络输入为 $\mathcal{O}_t$（图像、状态等）和带噪动作 $A_t^{(k)}$ 
    - CNN 对动作序列做一维卷积处理，通过 FiLM 与观测特征条件化
    - 最终输出预测的噪声向量 $\hat{\epsilon}$
- **训练目标**：最小化
    * $\mathcal{L} = \| \hat{\epsilon} - \epsilon \|^2 = E(A_t)$
	* 所以 $E(A_t)$ 本质就是网络的预测误差。

## 🔶 在 Transformer-based Diffusion Policy 中，E(At)E(A_t) 的函数意义

- **函数表示仍为**：$E(A_t) = \| \epsilon_\theta(\mathcal{O}_t, A_t^{(k)}, k) - \epsilon \|^2$
- **结构差异**：
    - 动作序列经过嵌入，作为 Query
    - 观测经过嵌入，作为 Key 和 Value
    - Transformer 使用 cross-attention 机制让动作“理解”观测条件
    - 多层 Transformer 后输出噪声预测 $\hat{\epsilon}$
- **含义保持不变**：
    - 仍然是“当前动作与真实动作之间的偏差”
    - 能量函数 $E(A_t)$ 本质是评估“该动作是否合理”

## ✅ 总结对比表

| 项目              | CNN-based                                                                  | Transformer-based                                                          |
| --------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 能量函数形式 $E(A_t)$ | $E(A_t) = \| \epsilon_\theta(\mathcal{O}_t, A_t^{(k)}, k) - \epsilon \|^2$ | $E(A_t) = \| \epsilon_\theta(\mathcal{O}_t, A_t^{(k)}, k) - \epsilon \|^2$ |
| 实现机制            | Conv1D + FiLM                                                              | Transformer + Cross Attention                                              |
| 网络目标            | 学会从观测中恢复动作的原始形态                                                            | 同左，但可处理更复杂依赖关系                                                             |
| 能量函数作用          | 衡量“当前动作是否合理”                                                               | 衡量“当前动作是否合理”                                                               |

## 🌟 延伸理解：
* 虽然在公式上 E(At)E(A_t) 只是一个平方误差，它的**“策略意义”**更深远：
> 它驱动策略模型从“噪声动作”逐步演化到“专家动作”。
> 从模仿学习角度看，它替代了显式策略网络，是一个“去噪”生成动作的策略优化方式。