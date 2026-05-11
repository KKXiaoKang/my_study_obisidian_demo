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