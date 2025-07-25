### load加载如下参数
```python
def load(self, path: str)
"""
Load model parameters and attributes from a file.
:param path: path to the saved model
"""
	data = torch.load(path, map_location=self.device)
	self.policy.load_state_dict(data['policy_state_dict'])
	self.obs_rms.mean = data['obs_rms_mean']
	self.obs_rms.var = data['obs_rms_var']
	self.obs_rms.count = data['obs_rms_count']
	self.policy.update_target_network(tau=1.0) # Hard update
```

### `save`  模型保存参数如下
##### RunningMeanStd状态
**恢复 `RunningMeanStd` 的状态**，让新环境下来的观测能使用跟训练一致的归一化
* ***obs_rms_mean**
	- 表示：训练期间所有观测的滑动 **均值**
	- 类型：`np.ndarray`，维度等于 `observation_space.shape`
	- 用途：用于观测归一化的中心化：`obs - mean`
* ***obs_rms_var**
	- 表示：训练期间所有观测的滑动 **方差**
	- 类型：`np.ndarray`    
	- 用途：用于标准化观测：`(obs - mean) / sqrt(var)` 
* ***obs_rms_count** 
	- 表示：累计更新的观测数量（用于增量计算 mean/var）
	- 类型：`float`
	- 用途：保持归一化过程的一致性和数值稳定性

##### 模型组相关参数
*  ***policy_state_dict** 
     * 保存`actor`网络的`weight`和`bias`
     * 保存`critic`当前网络的`weight` 和 `bias`
     * 保存`critic_target`目标网络的权重
     * 保存超参数$log({\alpha})$   ：用于控制策略的探索程度（熵）和期望回报之间的权衡
         * 关于为什么要取$\log$ 对数，为了始终保证$\alpha = exp(x)$ 为正数，其中的$x=log(\alpha)$
         * $exp(log(\alpha))$ 可以将参数还原为真实值
![[Pasted image 20250619111249.png]]
> `Tips`:在这里训练$\alpha$的时候，实际上在训练$log(\alpha)$ 这个参数
```bash
{
  'actor.fc1.weight': tensor(...),
  'actor.fc1.bias': tensor(...),
  'actor.fc2.weight': tensor(...),
  'critic.q1.fc1.weight': tensor(...),
  'critic.q2.fc1.bias': tensor(...),
  'critic_target.q1.fc1.weight': tensor(...),
  'log_alpha': tensor(0.4),   # 对应 log(α)
  ...
}
```