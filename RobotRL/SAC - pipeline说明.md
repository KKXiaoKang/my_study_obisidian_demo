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


#### 2) 