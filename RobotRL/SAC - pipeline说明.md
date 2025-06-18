## `tain` 过程详解 
#### 1) 从buffer中采样历史数据
```python
obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

...
def sample(self, batch_size):
    idx = random.sample(range(self.size), batch_size)
    return (
        self.obs[idx],
        self.actions[idx],
        self.rewards[idx],
        self.next_obs[idx],
        self.dones[idx]
    )

```
* `random` 从buffer当中不放回的均匀随机采样`batch_size`个索引,  一般设置的`batch_size`为128
* 随机抽取`128个idx` ，范围从`0到 buffer_size`


#### 2) 归一化观测
```python
# 归一化观测
## 累计得到的滑动均值和方差
mean = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=self.device)
var = torch.as_tensor(self.obs_rms.var, dtype=torch.float32, device=self.device)
# 确保var非负
var = torch.clamp(var, min=1e-8) # 确保var ≥ 1e-8
## 通过减去均值将分布中心化，通过除以标准差（即 sqrt 方差）将其缩放到单位方差
normalized_obs = (obs - mean) / torch.sqrt(var)
normalized_next_obs = (next_obs - mean) / torch.sqrt(var)
# 裁剪观测值 裁剪以避免极端数值
clipped_obs = torch.clamp(normalized_obs, -10.0, 10.0)
clipped_next_obs = torch.clamp(normalized_next_obs, -10.0, 10.0)
```
```bash
obs = torch.tensor([11.0, 18.0, 30.0])
next_obs = torch.tensor([12.0, 20.0, 33.0])
mean = [10.0, 15.0, 25.0]  # 均值
var = [4.0, 9.0, 25.0]

normalized_obs = (obs - mean) / torch.sqrt(var) 
               = ([11, 18, 30] - [10, 15, 25]) / [2, 3, 5]
               = [0.5, 1.0, 1.0]

normalized_next_obs = (next_obs - mean) / torch.sqrt(var)
					= ([12, 20, 33] - [10, 15, 25]) / [2, 3, 5]
                    ≈ [1.0, 1.667, 1.6]

# 裁剪到-10到10，其实没有变换
clipped_obs = torch.clamp(normalized_obs, -10.0, 10.0)
clipped_next_obs = torch.clamp(normalized_next_obs, -10.0, 10.0)
```

#### 3) 更新Critic评论家
* ***calculate_loss_q** 函数就是计算Q值损失(双目标Q网络，防止过高估计)
    *  `最大化预期回报 + 保持策略多样性（熵项）`
    * `目标Q值函数公式如下`
$$
\text{target}_Q = r + \gamma \cdot (1 - \text{done}) \cdot \left[ \min_i Q_{\text{target}}(s', a') - \alpha \cdot \log \pi(a'|s') \right]
$$
* 任意`函数Q的损失函数`定义如下
![[Pasted image 20250618183018.png]]
```python
# 计算Q值损失
q_loss = self.policy.calculate_loss_q(clipped_obs, actions, rewards, clipped_next_obs, dones, self.gamma)

# 评论家
self.policy.critic_optimizer.zero_grad()

# 反向传播
q_loss.backward()

# 评论家step
self.policy.critic_optimizer.step()
```
```python
def calculate_loss_q(self, obs, actions, rewards, next_obs, dones, gamma):
"""
计算Q值损失
:param obs: 当前状态
:param actions: 当前动作
:param rewards: 奖励
:param next_obs: 下一个状态
:param dones: 是否结束
:param gamma: 折扣因子
"""
	# 目标网络无梯度计算Q值
	with torch.no_grad():
		next_actions, log_pi_next, _ = self.actor(next_obs)
		target_q_values = self.critic_target(next_obs, next_actions)
		target_q_min = target_q_values.min(1)[0]
		target_q = rewards + (1 - dones) * gamma * (target_q_min - \
		           self.get_alpha().detach() * log_pi_next)
	# 当前网络计算Q值
	current_q = self.critic(obs, actions) # [batch_size, n_critics]
	# 计算Q值损失
	q_loss = 0.5 * (current_q - target_q.unsqueeze(1)).pow(2).sum(dim=1).mean()
	return q_loss
```


#### 4) 更新策略网络Actor
* 策略函数的损失函数如下：

$$
\mathcal{L}_\pi = \mathbb{E}_{s \sim D,\, a \sim \pi(\cdot|s)} \left[ \alpha \cdot \log \pi(a|s) - Q(s, a) + \text{action penalty} \right]
$$
```python
alpha_loss = self.policy.calculate_loss_alpha(log_pi)
self.policy.alpha_optimizer.zero_grad()
alpha_loss.backward()
self.policy.alpha_optimizer.step()
```
```python
def calculate_loss_pi(self, obs):
	"""
	计算策略损失
	:param obs: 当前状态
	"""
	actions_pi, log_pi, action_penalty = self.actor(obs) # 当前网络计算动作
	q_values_pi = self.critic(obs, actions_pi)
	min_qf_pi = q_values_pi.min(1)[0]
	policy_loss = (self.get_alpha().detach() * log_pi - min_qf_pi + action_penalty).mean()
	
	return policy_loss, log_pi
```

#### 5) 更新alpha - 更新熵参数
```python
def calculate_loss_alpha(self, log_pi):
	alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
	return alpha_loss
```
🧠 对应的数学公式
这个损失函数对应的是下面这个目标的负梯度方向：
$$
\mathcal{L}_\alpha = \mathbb{E}_{a \sim \pi} \left[ \alpha \cdot \left( -\log \pi(a|s) - \mathcal{H}_{\text{target}} \right) \right]
$$
令 $\alpha = \exp(\log \alpha)$，实际上优化的是：
$$
\mathcal{L}_{\log \alpha} = -\log \alpha \cdot \left( \log \pi(a|s) + \mathcal{H}_{\text{target}} \right)
$$
#### 6) 更新目标网络
```python
def update_target_network(self, tau=0.005):
	# 软更新目标网络
	for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
		target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```
*（如 θ′←τθ+(1−τ)θ′θ′←τθ+(1−τ)θ′）