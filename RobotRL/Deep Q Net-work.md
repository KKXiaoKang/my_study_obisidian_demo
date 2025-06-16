Q网络(Deep Q-Network, DQN)是深度强化学习中的一个重要概念，让我来详细解释一下：

1. 基本概念：
- Q网络是一个深度神经网络，用于近似Q函数(Q-function)
- Q函数表示在状态s下采取动作a的预期累积奖励
- 输入是状态s，输出是所有可能动作的Q值

2. 主要特点：
- 使用神经网络替代传统的Q表格
- 可以处理连续或高维的状态空间
- 能够泛化到未见过的状态

3. 网络结构：
```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

4. 训练过程：
- 使用经验回放(Experience Replay)存储转换样本
- 使用目标网络(Target Network)稳定训练
- 通过最小化TD误差来更新网络参数

5. 主要优势：
- 可以处理高维输入
- 不需要显式存储所有状态-动作对
- 能够学习复杂的策略

4. 改进版本：
- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- Noisy Networks

Q网络是深度强化学习中最基础和重要的算法之一，为后续的算法发展奠定了基础。