## SAC recipe experiments log
#### 已知问题
* 已知目前人工干预VR下，教学Kuavo机器人搬箱子，但是机器人只学习到如何接近目标，和抓取箱子，但是不能很好的抓取箱子后抬起来
* 同时存在问题q值loss暂时未收敛，于是乎从如下方法出发，针对这套SAC的架构进行参数对比实验

#### 实验
* 实验目标：让agent学会输出一个维度为$N$ 的target action为0
* 实验变量
	* reward设计包含如下：
		* MSE reward :
			* $\text{reward} = -np.mean( \text{action} - \text{target action}) ^ 2)$
		* shaped reward:
			* $\text{reward}= 1.0  /  (1.0 + np.linalg.norm( \text{action} - \text{target action}))$   
	* obs状态 : 
		* zero_obs ： 启用obs全为0状态
		* real_obs : 使用obs的真实观测状态
	* target_entropy : 
		* 
#### 实验具体项目如下
* ![[Pasted image 20250807183025.png]]
* ![[Pasted image 20250807183013.png]]

## experiment针对如下参数进行对比测试
