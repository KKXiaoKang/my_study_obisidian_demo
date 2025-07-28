分析`TEST_DEMO_ONLY_DEFAULT_JOINT_REWARD`为True时的奖励设计机制，这是一个专门用于关节空间学习的简化奖励系统。以下是详细的奖励组件：

## 1. **核心奖励组件**

### 1.1 默认关节位置奖励 (Default Joint Reward)
```python
# 目标关节角度：左臂(20,0,0,-50,0,0,0)度，右臂(20,0,0,-50,0,0,0)度
default_joint_angles_deg = np.array([
    20, 0, 0, -50, 0, 0, 0,  # Left arm
    20, 0, 0, -50, 0, 0, 0   # Right arm
])

# 基于平均偏差的指数衰减奖励
default_joint_reward = np.exp(-0.02 * mean_joint_deviation_deg) * 2.0
```
- **设计目的**：鼓励机器人手臂保持特定的舒适姿态
- **奖励范围**：0 到 2.0
- **特点**：使用指数衰减，偏差越小奖励越高

### 1.2 密集关节奖励 (Dense Joint Reward)
```python
# 基于每个关节的个体准确度
individual_joint_rewards = np.exp(-0.03 * np.rad2deg(joint_deviation)) * 0.1
dense_joint_reward = np.sum(individual_joint_rewards)
```
- **设计目的**：为每个关节的准确度提供细粒度反馈
- **奖励范围**：0 到 1.4 (14个关节 × 0.1最大奖励)
- **特点**：每个关节独立计算，提供更密集的学习信号

## 2. **惩罚机制**

### 2.1 效率惩罚 (Step Penalty)
```python
reward -= 0.001  # 每步小幅惩罚
```
- **设计目的**：鼓励快速收敛到目标姿态
- **影响**：防止智能体无意义地延长episode

## 3. **成功与终止机制**

### 3.1 成功条件
```python
success_threshold_deg = 5.0  # 5度容差
all_joints_close = np.all(np.rad2deg(joint_deviation) < success_threshold_deg)

if all_joints_close:
    reward += 10.0  # 成功奖励
    terminated = True
```
- **成功标准**：所有14个关节都在目标角度的5度范围内
- **成功奖励**：一次性10.0奖励bonus

### 3.2 超时终止
```python
elif self.episode_step_count >= 200:
    terminated = True
    info["success"] = False
```
- **最大步数**：200步
- **超时处理**：无额外惩罚，但标记为失败

## 4. **进步跟踪奖励 (Improvement Reward)**

```python
if mean_joint_deviation_deg < self.best_joint_deviation:
    improvement = self.best_joint_deviation - mean_joint_deviation_deg
    improvement_reward = improvement * 0.5
    reward += improvement_reward
    self.best_joint_deviation = mean_joint_deviation_deg
```
- **设计目的**：奖励在episode过程中的持续改进
- **奖励计算**：改进幅度 × 0.5
- **防止停滞**：只有当达到新的最佳状态时才给奖励

## 5. **奖励范围控制**

```python
reward = np.clip(reward, -5.0, 15.0)
```
- **下限**：-5.0（防止过度惩罚）
- **上限**：15.0（理论最大值：2.0 + 1.4 + 10.0 + 进步奖励）

## 6. **与正常模式的对比**

### DEMO模式特点：
- **专注性**：只关注关节空间控制，忽略任务相关奖励
- **密集性**：提供连续的、细粒度的反馈
- **简化性**：没有复杂的阶段性奖励或约束

### 正常模式差异：
- 包含靠近箱子、抓取、举起等任务相关奖励
- 有分阶段的奖励设计（接近阶段 vs 抓取阶段）
- 包含更多约束和惩罚机制

## 7. **设计理念**

这个DEMO模式的奖励设计遵循以下原则：

1. **单一目标**：专注于学习特定的关节配置
2. **密集反馈**：每个关节都有独立的奖励信号
3. **进步导向**：持续奖励改进而非维持状态
4. **简单明确**：避免复杂的多目标权衡

这种设计特别适合：
- 调试关节控制策略
- 验证基础的关节空间学习能力
- 作为更复杂任务的预训练基础


```bash
[DEMO CONSTRAINTS] Applied relaxed safety constraints:
  Left pos: [0.338, -0.060, -0.035]
  Right pos: [0.062, 0.091, 0.481]
[INCREMENTAL DEBUG] Left increment: [-0.01400664  0.00501108  0.00214253], Right increment: [-0.00834038 -0.00618724  0.01377134]
[INCREMENTAL DEBUG] Left absolute pos: [ 0.33794409 -0.05986074 -0.03498751], Right absolute pos: [0.06227585 0.09066066 0.48058272]
[INCREMENTAL DEBUG] Current cumulative - Left: [ 0.02014145 -0.4602788  -0.01557023], Right: [-0.25552678  0.49107873  0.5       ]
[DEMO MODE] Publishing action-based arm poses for joint space learning
[DEMO MODE] Step 199: Mean deviation: 36.35°
  Default joint reward: 0.967, Dense reward: 0.668
  Total reward: 1.634, Best so far: 12.53°
  All joints close: False, Terminated: False
  Worst joint: R1 (139.01°)
[VR INCREMENTAL DEBUG] VR positions - Left: [ 0.33794409 -0.05986074 -0.03498751], Right: [0.06227585 0.09066066 0.48058272]
[VR INCREMENTAL DEBUG] Fixed positions - Left: [ 0.31780263  0.40041807 -0.01941728], Right: [ 0.31780263 -0.40041807 -0.01941728]
[VR INCREMENTAL DEBUG] Calculated increments - Left: [ 0.02014145 -0.2        -0.01557023], Right: [-0.2  0.2  0.2]
[DEMO CONSTRAINT] Relaxed constraints - only linear z disabled:
    action[2] (linear z): 0.000
  Left hand increment: x=-0.060, y=-0.060, z=0.035
  Right hand increment: x=0.060, y=0.002, z=0.060
  Max increment limit: ±0.060
[DEMO CONSTRAINTS] Applied relaxed safety constraints:
  Left pos: [0.323, -0.060, -0.029]
  Right pos: [0.060, 0.086, 0.481]
[INCREMENTAL DEBUG] Left increment: [-1.5205311e-02  8.8629313e-06  5.7140258e-03], Right increment: [-0.0026723  -0.00451872  0.01501707]
[INCREMENTAL DEBUG] Left absolute pos: [ 0.32273877 -0.05985189 -0.02927348], Right absolute pos: [0.05960354 0.08614195 0.48058272]
[INCREMENTAL DEBUG] Current cumulative - Left: [ 0.00493614 -0.46026996 -0.00985621], Right: [-0.2581991   0.48656002  0.5       ]
[DEMO MODE] Publishing action-based arm poses for joint space learning
[DEMO MODE] Step 200: Mean deviation: 35.98°
  Default joint reward: 0.974, Dense reward: 0.678
  Total reward: 1.650, Best so far: 12.53°
  All joints close: False, Terminated: True
  Worst joint: R1 (138.71°)

```