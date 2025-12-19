![[Pasted image 20251219155940.png]]
### RT-2

> CE loss：  
> “在这个场景下，我应该说哪个动作？”

### GROOT（VLM 部分）

> CE loss：  
> “我是否正确理解了任务和场景？”

### GROOT（DiT 部分）

> Regression / diffusion loss：  
> “这条动作轨迹是否数值正确？”

# VIMA  (https://github.com/vimalabs/VIMA)
* VIMA = language-conditioned manipulation policy
* 基于Transformer的AI输入 （输入visual tokens+ 语义tokens压缩， 直接通过Transformer 自回归的生成动作）
![[Pasted image 20251219152022.png]]
* 【给我一张当前桌面的图 + 一句自然语言指令，我**直接输出一串机器人低层动作**】
### 👉 重点差异 - 目标信息用结构化tokens
* 视觉 token 不是普通 patch token, VIMA **没有用 ViT patch token**，而是
	* 使用 **object-centric representation** （目标检测）
	* 每个视觉 token 对应
		* 一个 object
		* 带有 **属性 embedding**（颜色、形状、类别）
		* 带有 **空间信息**
* VIMA 使用统一 self-attention 作为信息通道，但模态对齐主要是由 action 预测目标驱动的，而不是 token 交替本身
	* 没有 action loss，VIMA 的 language–vision self-attention 是对不齐的
* action token不进入Transformer
	* 只在最后MLP回归

### 👉关键训练目标 - action regression loss/ action autoregressive
例如：

> _“把红色方块放到蓝色方块左边”_

VIMA 的目标是：理解指令 -> 直接输出动作
- **感知当前场景**
- **理解语言中隐含的空间/操作约束**
- **直接生成一段 action trajectory**
* **(image + language) → action tokens**

模态对齐由 action loss 强约束
- 如果语言 attend 错 object
- 动作立刻失败
- loss 立刻爆
* **这句话在这个桌面场景下意味着哪条动作序列**


# RT-2 
* 没有快慢系统，直接单系统VLM从token中输出action token, 策略选择器
![[Pasted image 20251219155317.png]]

### 👉 重点差异
* RT-2作为策略选择器，不直接参与action生成
* 将action进行token化，  直接混合训练
* 将action token， visual token 和 language token在同一个token空间下完成融合

### 👉关键训练目标 - softmax → cross entropy 分类交叉熵
* token