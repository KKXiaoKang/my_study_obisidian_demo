# 论文原地址
* ![[Reinforcement Learning with Action Chunking.pdf]]

根据你的`SequenceACTSACActorV2`实现中的`_sample_action_sequence`方法，让我给出详细的数学公式：

## 🔢 联合概率计算的数学公式

### **1. 单个时间步的动作分布**

对于序列中的每个时间步 $t$，动作分布为：

$$a_t \sim \mathcal{N}(\mu_t, \Sigma_t)$$

其中：
- $\mu_t \in \mathbb{R}^d$ 是时间步 $t$ 的动作均值
- $\Sigma_t = \text{diag}(\sigma_t^2)$ 是对角协方差矩阵
- $d$ 是动作维度

### **2. Tanh变换分布（代码中的实际实现）**

由于使用了`TanhMultivariateNormalDiag`，实际的分布是：

$$a_t = \tanh(\tilde{a}_t), \quad \tilde{a}_t \sim \mathcal{N}(\mu_t, \Sigma_t)$$

单个动作的对数概率为：

$$\log p(a_t | s_{1:t}) = \log p(\tilde{a}_t) - \sum_{i=1}^d \log(1 - a_{t,i}^2)$$

其中 $\tilde{a}_t = \tanh^{-1}(a_t)$。

### **3. 动作序列的联合概率（Q-chunking核心）**

对于长度为 $T$ 的动作序列 $\mathbf{a}_{1:T} = \{a_1, a_2, ..., a_T\}$，在给定观测序列 $\mathbf{s}_{1:T}$ 的条件下：

#### **条件独立假设下的联合概率：**

$$p(\mathbf{a}_{1:T} | \mathbf{s}_{1:T}) = \prod_{t=1}^T p(a_t | \mathbf{s}_{1:T})$$

#### **联合对数概率：**

$$\log p(\mathbf{a}_{1:T} | \mathbf{s}_{1:T}) = \sum_{t=1}^T \log p(a_t | \mathbf{s}_{1:T})$$

### **4. 代码实现对应的公式**

你的代码中的实现：

```python
# 第352-356行
log_probs_individual = torch.stack(log_probs_list, dim=1)  # (batch, chunk_size)
log_probs_joint = log_probs_individual.sum(dim=1)  # (batch,)
```

对应的数学表达式：

$$\log p(\mathbf{a}_{1:T} | \mathbf{s}_{1:T}) = \sum_{t=1}^T \log p(a_t | \mathbf{s}_{1:T})$$

其中每个 $\log p(a_t | \mathbf{s}_{1:T})$ 是通过以下步骤计算的：

$$\log p(a_t | \mathbf{s}_{1:T}) = \log p(\tanh^{-1}(a_t); \mu_t, \Sigma_t) - \sum_{i=1}^d \log(1 - a_{t,i}^2)$$

### **5. 完整的Q-chunking损失函数**

在SAC框架中，使用这个联合概率的损失函数为：

$$\mathcal{L}_{\text{actor}} = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \alpha \log p(\mathbf{a}_{1:T} | s_t) - Q(s_t, a_1) \right]$$

其中：
- $\alpha$ 是温度参数
- $Q(s_t, a_1)$ 是只对第一个动作 $a_1$ 计算的Q值
- $\mathbf{a}_{1:T}$ 是完整的动作序列
- $\log p(\mathbf{a}_{1:T} | s_t)$ 是序列的联合对数概率

### **6. 重参数化采样公式**

对于每个时间步的采样：

$$\epsilon_t \sim \mathcal{N}(0, I)$$

$$\tilde{a}_t = \mu_t + \sigma_t \odot \epsilon_t$$

$$a_t = \tanh(\tilde{a}_t)$$

其中 $\odot$ 表示逐元素乘法。

### **7. 序列一致性的体现**

Q-chunking的关键在于联合概率 $p(\mathbf{a}_{1:T} | \mathbf{s}_{1:T})$ 确保了：

1. **时间一致性**：所有动作都基于相同的观测序列进行预测
2. **全局优化**：策略优化考虑整个动作序列的概率，而不仅仅是单步
3. **探索平衡**：熵正则化应用于整个序列，促进更好的探索

这种设计使得策略在学习时会考虑动作序列的整体一致性，避免了单步优化可能导致的短视行为。