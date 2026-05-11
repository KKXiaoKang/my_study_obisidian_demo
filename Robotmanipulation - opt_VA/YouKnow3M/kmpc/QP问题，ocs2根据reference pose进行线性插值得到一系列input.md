```bash
KMPC 上肢：
  reference anchor → SQP runImpl 中 setupQuadraticSubproblem
    → cost 项内部 LinearInterpolation::interpolate 拿目标
    → HPIPM 解 QP → optimizedState/Input → 直接下发
  (单层 QP，没有 WBC)

下肢 MPC+WBC：
  reference anchor → SQP runImpl 中 setupQuadraticSubproblem
    → cost 项内部 LinearInterpolation::interpolate 拿目标
    → HPIPM 解 QP → stateDesired/inputDesired
  → WBC: 对 inputDesired 做有限差分得到 q̈_des
    → qpOASES 解第二个 QP → τ → 下发
  (双层 QP)
```

## 把两条线路的“QP 数学”并排写出来

### KMPC（移动机械臂）单层 QP

```text
变量：     z = [δX, δU]           δX, δU 是整段轨迹增量
horizon：  T ≈ 1.0s, dt = 15ms
目标：
  min Σ_k 1/2 ||x_k - x_ref(t_k)||_{Q_k}^2
           + 1/2 ||u_k - u_ref(t_k)||_{R_k}^2
           + 1/2 ||ee_pose(x_k) - ee_ref(t_k)||_{W_ee}^2
约束：
  动力学:  x_{k+1} = f(x_k, u_k)         (kinematic model, 速度级)
  关节/底盘位置限位                       (relaxed barrier soft constraint)
  关节/底盘速度限位
  自碰撞距离 ≥ d_min
求解器：  HPIPM (OCS2 SQP 内部 QP)
输出：    nextState, optimizedInput → 关节级参考
```

### 下肢主体两层 QP

第一层：OCS2 SQP MPC

```text
变量：     z = [δX, δU]
horizon：  T ≈ 1.0s
模型：     centroidal momentum 模型
目标：
  min Σ_k 1/2 ||x_k - x_ref(t_k)||_Q^2
           + 1/2 ||u_k - u_ref(t_k)||_R^2
约束：
  centroidal dynamics
  接触一致性、摩擦锥
  关节/速度限位、CoM 限位、足端 swing trajectory
求解器：  HPIPM
输出：    stateDesired(t), inputDesired(t) 给 WBC
```

第二层：WBC QP（单步、每 1ms 一次）

```text
变量：     y = [q̈, F_c, τ]
horizon：  无（单步）
目标：（WeightedWbc）
  min  w_base   ||J_base q̈ + J̇_base v - a_base_des||^2
      + w_com   ||J_com  q̈ + J̇_com  v - a_com_des||^2
      + w_swing ||J_sw   q̈ + J̇_sw   v - a_sw_des||^2
      + w_joint ||q̈ - q̈_joint_des||^2
      + w_arm   ||q̈_arm - q̈_arm_des||^2
      + w_F     ||F_c - F_c_des||^2
约束：
  浮基方程 M q̈ + h = S^T τ + J_c^T F_c
  摩擦锥
  力矩限位 τ_min ≤ τ ≤ τ_max
  站立腿无运动 J_c q̈ + J̇_c v = 0
求解器：  qpOASES
输出：    τ → 力矩控制器
```

参考量怎么从 MPC 到 WBC：

```text
q_des, v_des, F_c_des  ← MPC interpolation (TargetTrajectories.getDesiredState/Input)
q̈_des                   ← (v_des(t) - v_des(t-dt_wbc)) / dt_wbc   有限差分
a_base_des, a_com_des  ← 由 q̈_des 和雅可比正向计算
```



```text
项                       KMPC                       下肢 humanoid
================================================================================
被控对象                 上肢 + 躯干 + 底盘          全身（下肢主导 + 上肢辅助）
模型                     kinematic                   centroidal momentum + 关节
状态 x                   关节角 + base pose          centroidal momentum + base + 关节
输入 u                   速度级                       接触力 + 关节速度
MPC 求解器               OCS2 SQP (SqpMpc)           OCS2 SQP (SqpMpc)
MPC 内部 QP              HPIPM                       HPIPM
MPC 频率                 100 Hz                      100 Hz
MPC 输出                 nextState、optimizedInput   stateDesired、inputDesired
是否接 WBC               否                           是
WBC 求解器               无                           qpOASES（WeightedWbc）
WBC 频率                 无                           ~500-1000 Hz
关节级参考               直接由 MPC 给               q̈, F_c, τ 由 WBC 给
约束处理                 soft constraint + barrier    硬等式 / 不等式 + 加权 task
末端目标                 EndEffectorConstraint 6D    arm task 通常是关节级
接触力建模               无                           显式 F_c + 摩擦锥
线性插值发生在            MPC reference 取值          MPC reference 取值 + 输入有限差分
================================================================================
```


```text
KMPC：
  每 100Hz：HPIPM QP，规模 ≈ (N × (stateDim + inputDim))
            stateDim 大约 6 + 14~20，N ≈ 70 (dt=15ms)
            总变量 ≈ 几千

下肢：
  每 100Hz：HPIPM QP，规模更大（centroidal 模型 + 接触一致）
  每 1ms：  qpOASES QP，变量 ≈ generalizedCoordinatesNum + 3*numContacts + actuatedDofNum
            约束包含浮基方程、摩擦锥、力矩限位
            ≈ 几十~一百多个变量，几十个约束
            单次求解 cpu_time 上限 2ms
```

所以人形下肢路线总的实时压力更大，因为有两层 QP；KMPC 只有一层 QP，但 SQP horizon 长、维度高，靠 MPC+MRT 解耦保证频率。


> 不是“线性插值得到 input”，而是：reference anchor 在 MPC 网格上做线性插值（姿态 slerp）得到目标值，SQP 在此基础上把非线性 OCP 局部化成 QP 求解，得到 state/input。KMPC 到这一步就结束；人形下肢还会再把 MPC 的 (state, input) 经过一次有限差分/雅可比映射成 q̈、F_c 目标，由 WBC 的 qpOASES QP 解出最终关节力矩。


# QP项都有哪些？QP问题构建
# 为什么“泰勒展开 + QP”可以指导优化？

---

# 1. 先看一个简单函数

$$
f(x)=x^2+4x+1
$$

这是一个标准抛物线。

它的最优点其实在：

$$
x^*=-2
$$

因为：

$$
f'(x)=2x+4=0
\Rightarrow x=-2
$$

但优化器一开始并不知道这个答案。

---

# 2. 假设当前站在：

$$
x_k=0
$$

优化器现在真正的问题是：

> “从当前位置往哪里走，会下降最快？”

---

# 3. Gradient（梯度）是什么？

先求导：

$$
f'(x)=2x+4
$$

代入当前位置：

$$
f'(0)=4
$$

于是：

$$
g_k=4
$$

---

## Gradient 的意义

gradient 本质：

> 当前的“坡度”。

这里：

$$
g_k=4>0
$$

说明：

- 往右走，函数会上升
- 往左走，函数会下降

所以优化器知道：

> 应该往左移动。

---

# 4. Hessian（海森矩阵）是什么？

再求一次导：

$$
f''(x)=2
$$

于是：

$$
H_k=2
$$

---

## Hessian 的意义

Hessian 本质：

> 当前函数的“曲率”。

也就是：

> 这个碗弯得有多厉害。

---

### 曲率小

地形平：

- 可以大胆走
- 步子可以大

---

### 曲率大

地形陡：

- 必须小步走
- 否则容易冲过头

---

# 5. 泰勒展开真正干了什么？

二阶泰勒展开：

$$
f(x_k+\Delta x)
\approx
f(x_k)
+
g_k\Delta x
+
\frac12 H_k(\Delta x)^2
$$

现在代入：

---

## 当前函数值

$$
f(0)=1
$$

---

## 当前梯度

$$
g_k=4
$$

---

## 当前 Hessian

$$
H_k=2
$$

---

得到：

$$
f(0+\Delta x)
\approx
1+4\Delta x+\frac12\cdot2(\Delta x)^2
$$

整理：

$$
=
1+4\Delta x+(\Delta x)^2
$$

---

# 6. 这个式子到底意味着什么？

注意：

优化器现在：

## 已经不直接优化原函数了

而是：

## 在当前位置附近

构造了一个：

# “局部抛物面模型”

这个模型告诉优化器：

> “如果你移动 $\Delta x$，函数大概会怎么变化。”

---

# 7. 为什么这会变成 QP？

因为：

现在目标函数已经变成：

$$
\min_{\Delta x}
\quad
1+4\Delta x+(\Delta x)^2
$$

这是一个：

# 标准二次函数

而：

# QP（Quadratic Programming）

本质：

就是：

# 优化二次函数。

---

# 8. 求这个局部模型的最低点

对：

$$
1+4\Delta x+(\Delta x)^2
$$

求导：

$$
2\Delta x+4=0
$$

得到：

$$
\Delta x=-2
$$

---

# 9. 这个 $\Delta x=-2$ 是什么？

它不是：

> “最终答案”

而是：

> “下一步建议移动方向”

意思是：

- 当前在 $x_k=0$
- 优化器建议：
  - 向左移动 2

---

# 10. 更新位置

更新规则：

$$
x_{k+1}=x_k+\Delta x
$$

代入：

$$
x_{k+1}=0+(-2)
$$

得到：

$$
x_{k+1}=-2
$$

---

# 11. 下一步会发生什么？

优化器会：

## 重新站在新位置

$$
x=-2
$$

然后：

## 再做一次泰勒展开。

---

# 12. 在新位置重新计算

---

## Gradient

$$
f'(-2)=2(-2)+4=0
$$

---

## Hessian

$$
f''(-2)=2
$$

---

于是：

新的局部模型：

$$
f(-2+\Delta x)
\approx
-3
+
0\cdot\Delta x
+
\frac12\cdot2(\Delta x)^2
$$

变成：

$$
=-3+(\Delta x)^2
$$

---

# 13. 为什么现在不动了？

因为：

$$
g_k=0
$$

意思：

> 坡度已经为0。

也就是说：

# 已经到谷底了。

---

# 14. 泰勒展开真正目的

泰勒展开不是：

> “把函数展开好看”

而是：

# “在当前位置附近，构造一个容易优化的局部模型”

---

# 15. 为什么机器人控制里特别重要？

真实机器人里的函数可能是：

- $\sin(q)$
- $\cos(q)$
- 动力学
- 接触约束
- 摩擦锥
- 复杂非线性

根本没法直接全局优化。

---

于是：

优化器每一步：

---

## Step1：当前位置

$$
q_k
$$

---

## Step2：局部二次化

$$
f(q_k+\Delta q)
\approx
f(q_k)
+
g_k^T\Delta q
+
\frac12\Delta q^TH_k\Delta q
$$

---

## Step3：构造QP

$$
\min_{\Delta q}
\quad
\frac12\Delta q^TH_k\Delta q
+
g_k^T\Delta q
$$

---

## Step4：解QP

得到：

$$
\Delta q
$$

---

## Step5：更新状态

$$
q_{k+1}=q_k+\Delta q
$$

---

## Step6：重新线性化

继续迭代。

---

# 16. 最终核心总结

---

## Gradient

告诉优化器：

> “往哪边走”

---

## Hessian

告诉优化器：

> “应该走多远”

---

## 泰勒展开

告诉优化器：

> “当前位置附近的世界长什么样”

---

## QP

负责：

> “快速找到这个局部世界里的最低点”
