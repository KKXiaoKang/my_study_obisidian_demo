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
* 对于超参，可以通过修改学习率lr从3e-4到1e-3，同时调节折扣因子，让sac学习action dim 为14的target action输出为0的学习效率大大加快
	* kuavo_debug_23_test | discount 0.95 | fps10_action_14_grad_clip_norm_10_reward_mse_no_batch_size_256 | lr_1e-3 | target_entropy_null
	* kuavo_debug_22_test | no_action_dim | fps10_action_14_grad_clip_norm_10_reward_mse_no_batch_size_64 | lr_1e-4 | target_entropy_null
	* ![[Pasted image 20250807193505.png]]
#### 对于stable-baseline3 PPO | stable-baseline3 SAC
* 在action dim为6的情况下，SAC的reward会抖动，但是PPO会一直稳定上涨
* ![[img_v3_02ou_29bac0a8-563e-431c-9ba9-24bdf419e94g.jpg]]

### 2. 关于reward shift
#### 在stable-baseline3 SAC 当中进行对比
* 可以看到在 mean reward（倒数reward下），倒数reward和倒数reward-1
	* 当前说明下的action dim默认为6
	* 所以这个reward scale不对，一个是奖励一个是惩罚，奖励下可能会出现震荡，但是惩罚下SAC可以稳步学习到reward上涨
	* ？提出问题：正向action下的输出是否可以作为一个奖励？具体的reward设计必须是要鼓励这种行为，而不是随机行为也给一个正奖励。
* ![[img_v3_02ou_ef8dcdeb-6976-484b-862a-c73f06f5c77g.jpg]]
* 对于奖励函数形态：倒数reward和 倒数reward-1 和 mse
	* ![[img_v3_02ou_a73e1b6b-d075-47d5-931d-7afa865cd09g.jpg]]

#### 在stable-baseline3 PPO 当中进行对比
* PPO mse vs 倒数 mse vs 倒数mse-1 (惩罚形态下收敛较快)
	* ![[img_v3_02ou_de5f4d9c-3a17-4404-882c-de545566694g.jpg]]
### 3. 关于reward scale
#### 在stable-baseline3 SAC 当中进行对比
* 可以看到在reward scale越大的时候，SAC居然反向收敛的越快
* sac action dim 为14。 reward分别 scale 1， 100， 1000，reward=mse
* ![[img_v3_02ou_cbb2fe0f-1b6a-45bd-9e63-78c190c128bg.jpg]]

#### 在stable-baseline3 PPO 当中进行对比
* reward scale对于ppo来说，其实是有一丢丢负面影响，因为导致ppo收敛变慢了，但是最终ppo还是可以收敛的
* ppo 14 dims; scale = 1 vs scale = 1000
* ![[img_v3_02ou_9434b41e-782c-496c-a7a2-4bee03e1905g.jpg]]
## 观察总结
* [x] SAC对于reward的设计极为敏感，从某种程度上说，reward当中不能随意设计，对于具体的reward设计必须是要鼓励这种行为，而不是随机行为也给一个正奖励
* [x] 在reward设计正确的情况下，SAC的训练收敛 确实 会比PPO快的多
* [x] ppo关于actor policy下，始终会回归到收敛程度，没有说对于简单的输出target action为0的时候 崩掉的情况
* [x] RLPD-Lerobot当中的SAC，对于梯度进行裁剪，防止训练爆炸，这部分和stable-baseline3 SAC官方的实现有所不同，所以在reward scale上的表现 lerobotSAC 确实无法做到像 stable-baseline3 SAC那样可以快速收敛

## 后续更改
* 可以定位到其实原本的RLPD的MetaVR遥操作的人工干预的buffer和transition当中并没有问题
* 问题来源于reward type上的一些设计技巧，以及关于超参的一些设置   