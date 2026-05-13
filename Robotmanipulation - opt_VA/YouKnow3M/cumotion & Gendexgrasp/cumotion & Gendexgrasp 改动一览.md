![[Pasted image 20260506201538.png]]
1. 基于 NVIDIA cuRobo 的 GPU 并行运动规划栈，将 6/7 自由度机械臂规划耗时压至 ~50ms 级，相较 OMPL/RRT 提速一个数量级，支持在线重规划。
2. 打通 nvblox ESDF → cuRobo Voxel 的实时感知–规划闭环，包含距离取反、未观测体素重映射、voxel 中心→表面对齐三处关键修正，保证感知数据语义与论文规范一致。
3. 设计 Go/Back 抓取语义状态机：去程实时刷新 ESDF 避障；回程冻结上一帧地图，避免被抓物体被误判为障碍——解决了纯感知规划在抓取闭环中"自己挡自己"的常见痛点。
4. 以 MoveIt2 MoveGroup Action 形式封装 cuRobo，做到对上层零侵入替换；同时通过双线程 + `MultiThreadedExecutor` 解耦感知刷新与规划触发，保证 action 不被 ESDF 服务阻塞。
5. 完整的可观测性与调参面板：实时发布 cuRobo 内部体素到 RViz、暴露 `time_dilation_factor / voxel_size / max_attempts` 等关键参数，便于不同场景（高架、抓取、转运）现场调优。


#  图片详细的理解 如何理解Minimum-Jerk C^2轨迹
这里的 **“Minimum-Jerk C² 轨迹”**，本质上是在说：

> cuRobo 在找到一条“可行路径（path）”之后，  
> 不只是简单地做直线插值，  
> 而是会进一步优化成一条 **平滑、连续、适合真实机器人执行** 的轨迹。

它对应的是机器人运动学/轨迹规划里的一个经典思想：

---

# 1. 什么是 Jerk？

机器人里：

- 位置：(x)
    
- 速度：(\dot{x})
    
- 加速度：(\ddot{x})
    
- Jerk（加加速度）：
    

[  
\dddot{x}  
]

也就是：

> 加速度变化的速度

---

例如：

|轨迹类型|现象|
|---|---|
|jerk 很大|“抽搐”“顿挫”|
|jerk 小|丝滑、像人一样|

---

# 2. Minimum-Jerk 是什么意思？

意思是：

> 在满足目标点、碰撞约束、速度约束的情况下，  
> 让整个轨迹的 jerk 尽可能小。

数学上通常是：

\min \int_0^T \left|\dddot{q}(t)\right|^2 dt

其中：

- (q(t))：关节轨迹
    
- (\dddot{q}(t))：jerk
    
- 整个积分：  
    表示“整段轨迹总体有多抖”
    

---

# 3. 为什么机器人特别重视 Minimum-Jerk？

因为真实机器人：

---

## （1）降低机械冲击

如果 jerk 很大：

- 电机瞬间冲击
    
- 谐波减速器受力剧烈
    
- 机械振动明显
    

尤其你做人形/双臂时：

- 长链条
    
- 柔顺结构
    
- 大惯量
    

非常怕 jerk spike。

---

## （2）减少末端抖动

比如你现在做：

- depalletize
    
- pick
    
- 双臂搬箱子
    

如果：

- 加速度突然变化
    
- 轨迹不平滑
    

那么：

- 箱子会晃
    
- 吸盘容易掉
    
- 双臂同步会错位
    

---

## （3）更像“人类动作”

人类手臂运动：

天然近似 minimum-jerk。

这是运动神经科学里的经典结论。

因此很多 imitation learning：

- ACT
    
- Diffusion Policy
    
- VLA
    

本质都在隐式学习：

> “低 jerk 的自然运动分布”

---

# 4. 什么是 C² 轨迹？

这里的：

> C²

意思是：

轨迹至少二阶连续：

|连续阶数|含义|
|---|---|
|C⁰|位置连续|
|C¹|速度连续|
|C²|加速度连续|

---

也就是说：

轨迹不能：

- 位置突然跳
    
- 速度突然跳
    
- 加速度突然跳
    

否则：

机器人会：

- 抖
    
- 爆 torque
    
- 电流尖峰
    

---

# 5. 为什么 Minimum-Jerk 常和 C² 一起出现？

因为：

如果你想让 jerk 有意义：

[  
\dddot{q}(t)  
]

那至少：

[  
\ddot{q}(t)  
]

必须连续。

所以：

Minimum-Jerk 通常意味着：

- C² 连续
    
- 甚至 C³ 连续
    

---

# 6. cuRobo 里的这一层到底是什么？

你图里的：

```python
motion_gen_result.optimized_plan
```

其实已经不是：

“离散路径点”

而是：

> 经过 trajectory optimization 的动态可执行轨迹。

里面通常包含：

- pos
    
- vel
    
- acc
    

也就是：

[  
q(t), \dot q(t), \ddot q(t)  
]

这意味着：

cuRobo 后面有一层：

trajectory parameterization / smoothing

在做：

- 时间参数化
    
- 动力学约束
    
- jerk smoothing
    

---

# 7. 为什么这是论文里的“技术深度点”？

因为很多人误以为：

Motion Planning 只是：

```text
A点 -> B点
```

实际上真正难的是：

```text
可执行
平滑
动态稳定
低振动
不碰撞
满足关节速度加速度限制
```

而 Minimum-Jerk + C²：

就是：

> “从几何路径”升级到“真实机器人运动轨迹”的关键一步。

这也是：

- MoveIt
    
- cuRobo
    
- CHOMP
    
- TrajOpt
    
- STOMP
    
- Drake
    

这些系统的核心。

---

# 8. 你可以怎么在汇报里讲（推荐）

你可以把它总结成一句非常专业的话：

---

### 建议表述（论文/汇报风格）

> 在路径搜索得到无碰撞几何路径后，  
> 系统进一步进行 Minimum-Jerk 的 C² 连续轨迹优化，  
> 生成满足位置、速度、加速度连续性的动态可执行轨迹，  
> 从而降低执行过程中的机械冲击与末端振动，  
> 提高真实机器人系统的稳定性与可部署性。

---

# 9. 更进一步（非常关键）

实际上：

你现在做的：

- delta EEF action
    
- flow matching
    
- chunk trajectory learning
    

也在隐式追求：

Minimum-Jerk。

因为：

如果 action chunk：

- 不平滑
    
- orientation jitter
    
- chunk boundary discontinuity
    

本质上：

就是 jerk 太大。

你最近碰到：

- 6D rotation jitter
    
- chunk 边界抖动
    
- RTC/BID 平滑
    
- action chunk overlap
    

其实都和：

> “轨迹高阶连续性”

直接相关。