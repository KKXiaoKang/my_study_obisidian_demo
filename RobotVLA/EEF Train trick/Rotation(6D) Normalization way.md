分析使用 absolute position 统计值归一化 relative position 的利弊：


分析使用 absolute position 统计值归一化 relative position 的利弊：

## 当前实现分析

从代码第 533-544 行可以看到，当前实现：
1. 使用 absolute position 的 `min/max` 统计值
2. 计算 `abs_range = max(|min|, |max|)`
3. 使用 `rel_range = abs_range * 1.5` 作为归一化范围
4. 使用 `[-rel_range, rel_range]` 进行归一化（以 0 为中心）

## 好处

### 1. 实现简单，无需额外预处理
- 直接复用 dataset 中已有的 absolute position 统计值
- 无需遍历整个 dataset 计算 relative position 的统计值
- 代码实现简单，维护成本低

### 2. 向后兼容
- 可以复用现有的统计值存储机制（`state_dict`/`load_state_dict`）
- 不需要修改 dataset 格式或统计值计算流程

### 3. 启发式调整（1.5x 倍数）
- 假设 relative position 的分布范围约为 absolute position 的 1.5 倍
- 对于大多数任务，这个倍数可能足够

## 坏处

### 1. 统计不准确
- Relative position 的分布与 absolute position 不同：
  - Absolute position：可能集中在工作空间的特定区域
  - Relative position：以 0 为中心，分布更对称，范围可能不同
- 使用 `1.5x` 是启发式，可能不反映真实分布

### 2. 可能影响训练效果

#### 情况 A：范围估计过大（`rel_range` 太大）
```python
# 如果实际 relative position 范围是 ±0.5m，但使用了 ±1.5m 的范围
# 归一化后：0.5m -> 0.33，大部分数据被压缩到 [-0.33, 0.33]
# 问题：模型学习到的信号范围较小，可能影响精度
```

#### 情况 B：范围估计过小（`rel_range` 太小）
```python
# 如果实际 relative position 范围是 ±1.5m，但使用了 ±0.5m 的范围
# 归一化后：1.5m -> 3.0（超出 [-1, 1] 范围）
# 问题：数据被裁剪，导致梯度问题，训练不稳定
```

### 3. 任务/数据集依赖
- 不同任务中 relative position 的分布差异可能很大：
  - 精细操作：relative position 范围小（±0.1m）
  - 大范围移动：relative position 范围大（±1.0m）
- 固定的 `1.5x` 倍数无法适应所有情况

### 4. 与 6D rotation 处理不一致
- 6D rotation：不归一化（保持几何约束）
- Position：使用启发式归一化（可能不准确）
- 这种不一致可能导致不同组件的学习难度不同

## 改进建议

### 方案 1：计算真实的 relative position 统计值（推荐）
```python
# 在训练时，遍历 dataset 计算 relative position 的统计值
# 保存到 state_dict 中，推理时直接使用
# 优点：准确，符合数据分布
# 缺点：需要额外的预处理步骤
```

### 方案 2：自适应范围估计
```python
# 在训练过程中，动态调整归一化范围
# 使用滑动窗口或指数移动平均来估计 relative position 的范围
# 优点：可以适应不同的数据分布
# 缺点：实现复杂，需要额外的状态管理
```

### 方案 3：使用更保守的范围（当前方案的改进）
```python
# 使用更大的倍数（如 2.0x 或 2.5x）来确保覆盖所有情况
# 优点：简单，避免裁剪问题
# 缺点：可能压缩有效信号范围
```

## 总结

当前方法是一种实用的折中：
- 优点：实现简单、向后兼容、对多数情况可用
- 缺点：统计不准确、可能影响训练、任务依赖性强

建议：
1. 短期：如果训练效果可接受，保持现状
2. 长期：实现方案 1（计算真实的 relative position 统计值），以获得更准确的归一化

需要我实现计算真实 relative position 统计值的方案吗？