### QN 讲解
* 这是一个[Flappy Bird](https://zhida.zhihu.com/search?content_id=120377040&content_type=Article&match_order=1&q=Flappy+Bird&zhida_source=entity)小游戏
![[Pasted image 20250616153000.png]]
* **状态State**：将每一帧作为一个状态，取小鸟离下一个地面上柱子在水平和竖直方向上的距离作为状态的观测值，即下图中的(△x, △y)；
- **行为Action**：对每一个状态（每一帧），只有两种选择：跳，不跳；
- **奖励Reward**：小鸟活着时给每帧奖励1，死亡时奖励-1000。
  在该游戏中，程序是如何选择该跳还是不该跳呢？按照前面说的Q学习算法，那么它应该是需要有一个`Q(S, A)`函数的，可以知道在什么状态时采取什么样的行为能得到最大的Reward之和。在这个游戏中，很显然状态和动作的组合都是有限的，因此可以维护一个`S-A`表，其记录了在每个状态下，采用什么动作时能得到什么样的`Q`值。表格形式如下，只要程序在运行中不断更新这个表格，使其最终能收敛，那么程序就能拿得到的`state`通过查表的方式来判断它该选择什么样的行为，才能获得最大的Q值。
![[Pasted image 20250616153015.png]]
### Q值更新
**Q值更新方法**  
Q值大体上有两种更新方式，一种是类似上面`小鸟游戏`例子中的情况，状态和行为的组合是可以穷尽的情况，这时候往往采用的是`S-A`表格的形式记录Q值，而如果状态和行为的组合不可穷尽，比如自动驾驶中输入的外界环境照片与车速之间的组合是有无穷种的，那么前一种方法显然就不适用了，这时候常用的方式为将深度学习与Q学习结合起来，也就是本文的重点，DQN，这个我们将在后面重点讲解。


### DQN讲解
Q网络(Deep Q-Network, DQN)是深度强化学习中的一个重要概念，让我来详细解释一下：

1. 基本概念：
- Q网络是一个深度神经网络，用于近似Q函数(Q-function)
- Q函数表示在状态s下采取动作a的预期累积奖励
-  在状态`S`下，采取动作`A`后，**未来**将得到的奖励`Reward`值之和

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