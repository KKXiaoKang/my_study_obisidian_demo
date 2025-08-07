## SAC recipe experiments log
#### 已知问题
* 已知目前人工干预VR下，教学Kuavo机器人搬箱子，但是机器人只学习到如何接近目标，和抓取箱子，但是不能很好的抓取箱子后抬起来
* 同时存在问题q值loss暂时未收敛，于是乎从如下方法出发，针对这套SAC的架构进行参数对比实验，验证该算法SAC本身是否可以支持多action space的输出能力，以及学习能力

#### 实验
* 实验目标：让agent学会输出一个维度为$N$ 的target action为0
* 实验算法
	* SAC - lerobot
	* SAC - stable-baseline3
	* PPO - stable-baseline3
* 实验变量
	* reward设计包含如下：
		* MSE reward :
			* $\text{reward} = -np.mean( \text{action} - \text{target action}) ^ 2)$
		* shaped reward:
			* $\text{reward}= 1.0  /  (1.0 + np.linalg.norm( \text{action} - \text{target action}))$   
	* obs状态 : 
		* zero_obs ： 启用obs全为0状态
		* real_obs : 使用obs的真实观测状态
	* target_entropy : 目标熵值
		* 设置null ： 可以自动根据 action.shape 算出一个target entropy
		* 设置为固定值
	*  grad_clip_norm : 梯度裁剪范围
		* 将梯度更新时设置最大值
	*  discount：折扣因子
		* 每次critic网络更新时的折扣因子
	*  batch_size： 每次学习的批次大小
		* 64
		* 256
	*  temperature_init：初始熵值
		* 1.0
		* 0.01
#### 实验具体项目如下
* ![[Pasted image 20250807183025.png]]
* ![[Pasted image 20250807183013.png]]

## experiment针对如下参数进行对比测试，挑选一些重点进行记录

### 1. 关于reward type
* 对于SAC，奖励不能一直都是正奖励，或者奖励设计上有偏差，偏差下会出现训练reward一直不涨的情况
* 但是对于PPO，最终肯定会收敛，而不是像SAC那样在一个局部最小reward当中附近震荡
#### 对于 lerobot / stable-baseline3 SAC来说，reward十分敏感 | 同时也对超参十分敏感
* 对于mean reward
	* 当前mean reward下，每一个step的最大奖励为1，eposide最大为200步，则最大奖励为200
	* 当采用shaped mean的reward设计时，只有当action space低于6（实验当中采用了shape为2和6）才可以收敛到reward接近200，可以看到下图，action为2的时候很好学，但是action为6的时候收敛很困难
		* kuavo_debug_19_test | no_action_dim | fps10_action_06_grad_clip_norm_10_reward_shaped_no_batch_size_64
		* kuavo_debug_11_test | zero_obs_yes_dataset_action-0.5or0.5_fps10_seed1000_action_02_reward_type_shaped
	* ![[Pasted image 20250807191020.png]]
* 对于MSE reward
	* MSE reward下，eposide最大为200步，则最大奖励为0
		* kuavo_debug_22_test | no_action_dim | fps10_action_14_grad_clip_norm_10_reward_mse_no_batch_size_64 | lr_1e-4 | target_entropy_null
		* kuavo_debug_20_test | no_action_dim | fps10_action_06_grad_clip_norm_10_reward_mse_no_batch_size_64 | lr_3e-4 | target_entropy_-6.0
	* 对比如上实验，可以看到在MSE reward下，关于动作维度可以扩展到14都可以进行收敛到0
		* ![[Pasted image 20250807191522.png]]
* 对于超参，可以通过修改学习率lr从3e-4到1e-3，同时调节折扣因子，让sac学习输出14维度的action输出为0的学习效率大大加快
	* kuavo_debug_23_test | discount 0.95 | fps10_action_14_grad_clip_norm_10_reward_mse_no_batch_size_256 | lr_1e-3 | target_entropy_null
	* kuavo_debug_22_test | no_action_dim | fps10_action_14_grad_clip_norm_10_reward_mse_no_batch_size_64 | lr_1e-4 | target_entropy_null
	* ![[Pasted image 20250807191908.png]]


### 2. 关于reward scale
* 在实验当中，对比于