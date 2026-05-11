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

### QP 标准形式

```text
min_x  1/2 x^T H x + g^T x

s.t.   A_eq x = b_eq
       A_ineq x ≤ b_ineq
```

约束：`H` 半正定（凸）；`x` 是决策变量。

### 例子：两个二次跟踪目标 + 一个线性等式

假设有两个目标，写成最小二乘形式：

```text
min   1/2 ||A1 x - b1||^2  +  1/2 ||A2 x - b2||^2
s.t.  C x = d
```

**第一步：展开成 H, g**

```text
||Ai x - bi||^2
= (Ai x - bi)^T (Ai x - bi)
= x^T Ai^T Ai x  -  2 bi^T Ai x  +  bi^T bi
```

把 1/2 加上、把所有 task 相加：

```text
1/2 Σ ||Ai x - bi||^2
= 1/2 x^T (Σ Ai^T Ai) x  +  (- Σ Ai^T bi)^T x  +  const
```

对照 QP 标准形式：

```text
H = Σ Ai^T Ai
g = - Σ Ai^T bi
```

这正是 WBC 代码里的写法：

```86:90:/home/lab/kuavo-ros-control-amp/src/humanoid-control/humanoid_wbc/src/WeightedWbc.cpp
Task weighedTask = formulateWeightedTasks(stateDesired, inputDesired, period);
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> H =
    weighedTask.a_.transpose() * weighedTask.a_;
vector_t g = -weighedTask.a_.transpose() * weighedTask.b_;
```

每个 task 给一个 `(A_i, b_i)`，加权时直接 `(w_i * A_i, w_i * b_i)`（代码里是 `task * weight`），最后竖着堆起来再做 `Aᵀ A` 就行。

**第二步：把等式 / 不等式约束塞进 lbA, ubA**

WBC 里所有约束 `(a, b)` 等式 + `(d, f)` 不等式（`d x ≤ f`）拼起来：

```text
A = [ a;
      d ]

lbA = [ b;
        -inf ]

ubA = [ b;
         f ]
```

代码：

```76:84:/home/lab/kuavo-ros-control-amp/src/humanoid-control/humanoid_wbc/src/WeightedWbc.cpp
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(numConstraints, getNumDecisionVars());
vector_t lbA(numConstraints), ubA(numConstraints); // clang-format off
A << constraints.a_,
    constraints.d_;

lbA << constraints.b_,
      -qpOASES::INFTY * vector_t::Ones(constraints.f_.size());
ubA << constraints.b_,
      constraints.f_;
```

`lbA ≤ A x ≤ ubA` 就同时表达了等式（`lbA = ubA = b`）和不等式（`lbA = -inf`）。

**第三步：调求解器**

```text
qpProblem.init(H, g, A, nullptr, nullptr, lbA, ubA, nWsr, &cpu_time);
```

具体到 KMPC 的 SQP，OCS2 会把每个 shooting node 的 cost、动力学线性化、约束线性化拼成一个超大的 sparse QP，交给 HPIPM 解。流程一样，只是规模大、并且每次 MPC 迭代只解一个 QP（因为 task.info 里 `sqpIteration = 1`）。

### 一个具体小数字例子

让 x ∈ R²。两个 task：

```text
task1: 让 x 接近 [1, 0]    →  A1 = I,  b1 = [1, 0]^T
task2: 让 x1 + x2 接近 0   →  A2 = [1, 1],  b2 = 0
```

一个等式约束：

```text
x1 - x2 = 0
```

写成 QP：

```text
H = A1^T A1 + A2^T A2
  = I + [1; 1][1, 1]
  = [[2, 1],
     [1, 2]]

g = -A1^T b1 - A2^T b2
  = -[1, 0]^T
  = [-1, 0]^T

A_eq = [1, -1],   b_eq = 0
```

最优化结果：把 `x1 = x2 = a` 带入，目标变成 `1/2 (a-1)^2 + (1/2)(0)^2 + 1/2 (2a)^2` 对 a 求导得 `a = 1/5`，即 `x* = [0.2, 0.2]`。这就是 QP 给出的解。

### 构建指南（按顺序）

```text
1. 先确定决策变量 x：
   - WBC: [q̈, F_c, τ]
   - MPC: 所有 node 上的 [δx_k, δu_k] 拼起来

2. 把所有要追踪的目标写成 ||A_i x - b_i||^2：
   - 跟踪目标值 b_i, 雅可比矩阵作 A_i
   - 比如 CoM 跟踪: A = J_com, b = a_com_des - J̇_com v

3. 加权: A_i ← sqrt(w_i) A_i, b_i ← sqrt(w_i) b_i

4. 竖着堆: A_total = [A_1; A_2; ...], b_total = [b_1; b_2; ...]
   H = A_total^T A_total,  g = -A_total^T b_total

5. 写硬约束:
   - 等式（动力学方程、接触零速等）→ a x = b
   - 不等式（摩擦锥、力矩限、关节限位等）→ d x ≤ f
   合成 lbA ≤ A_c x ≤ ubA

6. 检查 H 是否半正定（每个 task 自带 A^T A 一定半正定，正常情况下不用调）

7. 给 QP 求解器: qpOASES / HPIPM / OSQP 都行
```

### 为什么 SQP MPC 也是这套

非线性的 OCP 在当前轨迹上做一次泰勒展开：

```text
cost:        L(x_k + δx, u_k + δu)
            ≈ L_k + ∇L_k [δx, δu] + 1/2 [δx, δu]^T H_k [δx, δu]

dynamics:    x_{k+1} + δx_{k+1} = f(x_k + δx, u_k + δu)
            ≈ f(x_k, u_k) + A_k δx_k + B_k δu_k
            → δx_{k+1} = A_k δx_k + B_k δu_k + b_k

constraint:  g(x_k + δx, u_k + δu) ≤ 0
            ≈ g_k + C_k [δx, δu] ≤ 0
```

代进 QP 模板，决策变量是所有 `[δx_k, δu_k]` 拼成的大向量，约束包括所有 `δx_{k+1} - A_k δx_k - B_k δu_k = b_k` 和线性化后的不等式，目标就是局部二次的 cost。这就是 `setupQuadraticSubproblem → getOCPSolution (HPIPM) → takeStep` 里在做的事情。

如果想验证，可以打开 task.info 把 `sqp.sqpIteration` 改成 1，就能看清 SQP 单次只迭代一个 QP 这件事；如果改大它会循环“构造 QP → 解 QP → line search → 重新线性化”直到收敛。