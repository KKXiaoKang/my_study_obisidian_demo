![[Pasted image 20250703140939.png]]
## 关于SAC当中离散Q值网络的实现指南
### 机器人控制的实际需求
#### 连续动作：
- 关节速度控制
- 末端执行器位置控制
- 力控制
#### 离散动作：
- 夹爪开关控制
- 工具选择
- 操作模式切换
- 任务阶段选择

## 关于SAC当中的Buffer经验回放池的设计pipeline及区别
### 两种回放缓冲区的区别

| 特性   | replay_buffer (在线)                             | offline_replay_buffer (离线)                                            |
| ---- | ---------------------------------------------- | --------------------------------------------------------------------- |
| 数据来源 | Actor实时收集的环境交互数据                               | 预定义的数据集（如人类演示数据）                                                      |
| 更新方式 | 动态添加新的转换数据<br><br>数据构成：policy探索的数据 + 人工在线干预的数据 | 静态加载，训练过程中正常不更新,但是有一些方法除外，比如Hil-Serl当中<br><br>数据构成：专家演示数据 + 人工在线干预的数据 |
| 容量   | cfg.policy.online_buffer_capacity              | cfg.policy.offline_buffer_capacity                                    |
| 用途   | 存储在线探索的经验                                      | 存储高质量的人类演示数据                                                          |
|      |                                                |                                                                       |

### 关于如何从两个Buffer当中采样 并且 完成融合
* 核心构建pipeline ： `生产者` or `消费者`

*  只有在队列用完的情况下，需要重新再次创建`iterator`的时候，才会调用`offline_replay_buffer`当中的`sample`重新采样新的数据
```python
online_iterator = replay_buffer.get_iterator(
	batch_size=batch_size, # 批量大小
	async_prefetch=async_prefetch, # 是否异步预取
	queue_size=2 # 队列大小
)

offline_iterator = offline_replay_buffer.get_iterator(
	batch_size=batch_size, # 批量大小
	async_prefetch=async_prefetch, # 是否异步预取
	queue_size=2 # 队列大小
)

# **get_iterator** 会重新调用self.sample(batch_size) 获取新的batch
```


*  `replay_buffer`
```python
# 迭代获取状态
batch = next(online_iterator)
batch_offline = next(offline_iterator)

# 将在线和离线批次数据连接起来 torch.cat
batch = concatenate_batch_transitions(
	left_batch_transitions=batch, # 在线批次
	right_batch_transition=batch_offline # 离线批次
)
```