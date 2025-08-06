* https://arxiv.org/abs/2502.20396
* [https://toruowo.github.io/recipe](https://toruowo.github.io/recipe)
* ![[Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids.pdf]]
###  论文核心 - pipeline

![[Pasted image 20250806145311.png]]

#### B.通用的奖励设计方案 - 用于操作任务抓取
* manipulation的dense reward 参考设计
* 对于灵巧手抓取任务（长时程-接触丰富的操作任务），定义动作序列状态
	* 手 - 物体接触 - 物体状态组合
*  任务状态 - 根据接触状态和物体状态的交错序列
	* 1） 一只手接触物体
	* 2）物体被抬起到另一只手附近的位置
	* 3）另一只手接触物体
	* 4）物体被转移到最终目标位置
* 奖励仅基于“接触目标”和“物体目标”来定义
	*  每个接触目标可以通过惩罚手指到理想接触点或物体质心位置的距离来指定
	*  每个物体目标可以通过惩罚其当前状态到其Target目标状态的距离来指定（distance reward）
* 奖励公式 物体接触公式
	* 机器人要操作一个物体，就得跟物体**产生接触**。但具体**接触哪里、如何接触**，是学习的关键。为了让学习过程知道**哪些接触是“好”的**，作者设计了一个**基于“接触目标点”的奖励函数**
$$r_{contact} = \sum_i \left[ \frac{1}{1 + \alpha d(\mathbf{X}_i^L, \mathbf{F}_i^L)} + \frac{1}{1 + \beta d(\mathbf{X}_i^R, \mathbf{F}_i^R)} \right]$$
* 其中每一项的含义如下：
	* $\mathbf{X}_i^L$ , $\mathbf{X}_i^R$ 
		* 在第$i$个任务中， 左/右手的目标接触点坐标（你希望它解除的物体上的点）


#### A.自动从真实到模拟调优的模块
* 模块使模拟环境更接近真实世界

#### C.一个分而治之的蒸馏过程
* 该过程提高了硬探索问题的样本效率

#### D.稀疏和密集物体表示的混合
* 弥合模拟到真实的感知差距