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

#### 3) 更新Critic