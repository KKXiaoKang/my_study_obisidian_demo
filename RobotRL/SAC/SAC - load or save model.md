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
*  ***policy_state_dict** 
     * 保存`actor`网络的`weight`和`bias`
     * 保存`critic`当前网络的`weight` 和 `bias`
     * 保存`critic_target`目标网络的权重
     * 保存超参数$log({\alpha})$   ：用于控制策略的探索程度（熵）和期望回报之间的权衡
         * 关于为什么要取$\log$ 对数，为了始终保证$\alpha$ 为正数 
         * $exp(log(\alpha))$ 可以将参数还原为真实值
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