## 2025/8/13 - rewACT 添加reward model
* 开发分支：KangKK/beta/reward_model_ACT_fix_key
	* 9ef81ee08b1818606bbcd7fe7abfe06ea94d365f
* 在现有的act当中添加一个MLP结构的reward head，用于根据输入obs时，自动计算当前的reward为多少
* input -> 512转256 -> 256转128 -> 128转1 -> output
```python
            self.reward_head = nn.Sequential(
                nn.Linear(config.dim_model, config.dim_model // 2),
                nn.ReLU(),
                nn.Linear(config.dim_model // 2, config.dim_model // 4),
                nn.ReLU(),
                nn.Linear(config.dim_model // 4, 1),
                nn.Sigmoid(),
            )
```
* 使用监督学习的方式，不断降低reward loss，将reward loss将入到训练过程当中，同时输出的reward可以用于后期RL微调的dense reward model（原作： Hil-Serl也训练了一个对应的二分类reward model）
* 现有的ACT架构
```bash
                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
```

## 训练过程当中reward_loss也作为学习目标
```python
loss = l1_loss + self.config.reward_loss_weight * reward_loss + mean_kld * self.config.kl_weight
```
* ![[Pasted image 20250813165725.png]]


## 单独放一个训练集当中的episode输出![[reward_visualization_train.mp4]]
* 

## 最终实时推理 | 找一个测试集的episode实时推理reward | 原子策略用于运行
* ![[0_A_rewACT_仿真演示测试_原子策略跑_reward拿obs去推测reward.mp4]]