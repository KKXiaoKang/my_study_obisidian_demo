![[Pasted image 20250616151927.png]]
* 强化学习执行流程如上图所示，`Agent`是我们的程序，它观察`Environment`并获得`state`，依据它的`Policy`对`state`做出`action`，此时能得到一个`reward`，且`Environment`改变了，因此`Agent`会得到一个新的`state`，并继续执行下去。

## 关于obs当中环境状态 vs 状态的区别
#### observation.state（状态）
- 含义：通常指机器人自身的状态信息
- 内容：可能包括：
- 关节角度（joint angles）
- 关节速度（joint velocities）
- 末端执行器位置（end-effector position）
- 机器人内部传感器数据
- 机器人配置信息

#### observation.environment_state（环境状态）
- 含义：通常指外部环境的状态信息
- 内容：可能包括：
- 目标物体位置
- 环境中的障碍物信息
- 任务相关的环境变量
- 外部传感器数据（如力传感器、距离传感器等）
- 环境配置参数

```python
# 输入观测数据示例
obs = {
    # 图像观测
    "observation.images.front": torch.tensor([...]),  # 前视相机图像
    "observation.images.wrist": torch.tensor([...]),  # 手腕相机图像
    
    # 🔥 机器人状态（observation.state）
    "observation.state": torch.tensor([
        0.1, 0.2, 0.3,  # 关节1角度、速度、力矩
        0.4, 0.5, 0.6,  # 关节2角度、速度、力矩
        0.7, 0.8, 0.9,  # 关节3角度、速度、力矩
        0.1, 0.2, 0.3,  # 末端执行器位置 (x, y, z)
        0.4, 0.5, 0.6,  # 末端执行器姿态 (roll, pitch, yaw)
        0.7, 0.8, 0.9,  # 夹爪状态
        0.1, 0.2, 0.3,  # 其他机器人内部状态
    ]),
    
    # 🔥 环境状态（observation.environment_state）
    "observation.environment_state": torch.tensor([
        0.5, 0.6, 0.7,  # 目标物体位置 (x, y, z)
        0.8, 0.9, 1.0,  # 目标物体姿态
        0.1, 0.2, 0.3,  # 障碍物1位置
        0.4, 0.5, 0.6,  # 障碍物2位置
        0.7, 0.8, 0.9,  # 环境温度、湿度等
        0.1, 0.2, 0.3,  # 任务进度、时间等
    ])
}
```

#### 机器人操作任务：
- observation.state：机器人关节状态、末端位置、夹爪状态
- observation.environment_state：目标物体位置、桌面高度、障碍物信息
#### 自动驾驶任务：
- observation.state：车辆速度、转向角、加速度
- observation.environment_state：其他车辆位置、交通信号、道路状况
#### 无人机任务：
- observation.state：无人机位置、姿态、电池状态
- observation.environment_state：目标点位置、风速、障碍物分布

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

| 特性   | replay_buffer (在线)                | offline_replay_buffer (离线)         |
| ---- | --------------------------------- | ---------------------------------- |
| 数据来源 | Actor实时收集的环境交互数据                  | 预定义的数据集（如人类演示数据）                   |
| 更新方式 | 动态添加新的转换数据                        | 静态加载，训练过程中不更新                      |
| 容量   | cfg.policy.online_buffer_capacity | cfg.policy.offline_buffer_capacity |
| 用途   | 存储在线探索的经验                         | 存储高质量的人类演示数据                       |
|      |                                   |                                    |

## 拓展：具体可以查看[[Agent - explore - SMDP半马尔可夫决策过程]]