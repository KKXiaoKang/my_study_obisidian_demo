已经把两条链路看清楚了：humanoid 主体里 MPC（OCS2 SQP）出来的 nextState/Input 会先“线性插值/求差”，然后再送给 WBC 形成第二个 QP；KMPC 不接 WBC，MPC 出来的关节状态直接走 arm/waist 控制下发。下面先纠正你那句话的几个细节，再把两条路线的“QP 数学差异”讲清楚。

## 1. 先把那句话修一下

你说的：

> QP 问题，OCS2 根据 reference pose 进行线性插值得到一系列 input。

更准确的表述是这样：

```text
- OCS2 接收的 reference 是稀疏 anchor，不是稠密参考。
- 在 MPC 求解前，cost 项调用 getDesiredState(t_k) / getDesiredInput(t_k)，
  对 reference 做线性插值（姿态用 slerp）。
- 这是“拿到每个 shooting node 上的目标值 x_ref(t_k), u_ref(t_k)”。
- 然后 SQP 在当前轨迹 (x_k, u_k) 附近做二阶近似 + 动力学/约束线性化，
  形成一个 QP 子问题。
- QP 解的是 δx_k, δu_k，最优 input 是 QP 优化的结果，
  不是“插值出来”的。
```

也就是说插值只是“把稀疏目标拉到 MPC 网格上”，真正的 input 是 QP 求出来的。

## 2. 两条链路概览

```text
KMPC 路线（上肢/移动机械臂）：
VR retarget → reference anchor
  → OCS2 SQP（kinematic MPC，速度级 input）
  → MRT (evaluatePolicy) 得到 nextState、optimizedInput
  → 直接发布 arm_traj、waist_traj、cmd_pose
  → 关节级位置/速度控制器执行

人形主体路线（下肢 + 全身）：
joystick/gait/footPose → reference anchor
  → OCS2 SQP（centroidal model MPC）
  → MRT (evaluatePolicy) 得到 stateDesired、inputDesired
  → WBC（qpOASES QP）形成 q̈、F_c、τ
  → 关节力矩控制器执行
```

注意一个核心区别：**KMPC 是单层 QP（MPC 内部）；下肢是双层 QP（MPC 内部一个 QP + WBC 一个 QP）**。

## 3. KMPC 这层 QP 在算什么

`MobileManipulator` 用的是 kinematic 模型，state 是关节位置 + base 位姿，input 是速度。

数学上 OCS2 内部 SQP 构造的 QP 是：

```text
min_{δX, δU}
  Σ_k  1/2 δz_k^T H_k δz_k + h_k^T δz_k

s.t.
  δx_{k+1} = A_k δx_k + B_k δu_k + b_k       (动力学线性化)
  C_k δz_k + c_k ≤ 0                          (软/硬约束线性化)
  δx_0 = x0_measured - x_0_nominal
```

其中 δz_k = [δx_k, δu_k]。具体到你的代码：

```text
- H_k 来自 cost 项 Hessian：
    QuadraticInputCost           → R 对应 input
    QuadraticBaseStateCost       → Q 对应 base tracking
    BaseArmRegularizationCost    → Q_reg
    EndEffectorConstraint 二次惩罚 → muPosition*, muOrientation*
- A_k, B_k 来自 dynamics 在 (x_k, u_k) 的雅可比
- C_k 来自：
    self-collision relaxed barrier
    joint position/velocity limits
    base position limits
- x_ref(t_k), u_ref(t_k) 来自 TargetTrajectories 线性插值/slerp
```

求出来 δX, δU 后做 line search，把整段 (X, U) 更新。MRT 再在最新 policy 上插值给当前 t 的 nextState 和 optimizedInput。这些直接当成 arm/waist 的关节级参考下发。

```text
KMPC 链路里 QP 只有一个：OCS2 内部 SQP QP。
没有 WBC。
没有动力学逆解。
没有接触力。
```

## 4. 下肢 humanoid 这层 QP 在算什么

下肢主体走的是“centroidal model MPC + WBC”。MPC 那一层和上面差不多，也是 SQP 把 OCP 变 QP，但状态/输入和约束完全不一样：

```text
state:  [centroidal momentum (6), base pose (6), joints (jointNum)]
input:  [contact forces (3 × numContacts), joint velocities]
cost:   tracking + 输入正则 + arm tracking + ...
constr: 摩擦锥、零速接触、关节限位、足端切换 ...
```

MPC 出 stateDesired, inputDesired 后，喂给 WBC。WBC 是一个**单步 QP**，不是 MPC：

```text
决策变量 x = [q̈ (generalizedCoordinatesNum),
              F_c (3*numContacts),
              τ  (actuatedDofNum)]

目标（WeightedWbc）：
  min 1/2 ||A_task x - b_task||^2

约束：
  浮基方程       M q̈ + h = S^T τ + J_c^T F_c
  摩擦锥          μ |F_x|, |F_y| ≤ F_z
  关节力矩限     τ_min ≤ τ ≤ τ_max
  无接触运动     J_c q̈ + J̇_c v = 0 (站立脚)
```

正是这里：

```86:108:/home/lab/kuavo-ros-control-amp/src/humanoid-control/humanoid_wbc/src/WeightedWbc.cpp
// Cost
Task weighedTask = formulateWeightedTasks(stateDesired, inputDesired, period);
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> H =
    weighedTask.a_.transpose() * weighedTask.a_;
vector_t g = -weighedTask.a_.transpose() * weighedTask.b_;

const auto t3 = Clock::now();
// Solve
auto qpProblem = qpOASES::QProblem(getNumDecisionVars(), numConstraints);
qpOASES::Options options;
options.setToMPC();
options.printLevel = qpOASES::PL_LOW;
options.enableEqualities = qpOASES::BT_TRUE;
qpProblem.setOptions(options);
int nWsr = 200;
qpOASES::real_t cpu_time = 0.002;
const auto t4 = Clock::now();
// minimize 0.5 * x^T * H * x + g^T * x
// subject to lbA <= A * x <= ubA
qpProblem.init(H.data(), g.data(), A.data(), nullptr, nullptr, lbA.data(), ubA.data(), nWsr, &cpu_time);
```

WBC QP 的 cost 是 task 形式 ||Ax - b||²，再展开成 H, g。具体 task 你能在 WbcBase 里看到一堆，每个对应一个跟踪目标：

```text
formulateFloatingBaseEomTask     -> 浮基方程（约束）
formulateTorqueLimitsTask        -> 力矩限位（不等式约束）
formulateFrictionConeTask        -> 摩擦锥（不等式约束）
formulateNoContactMotionTask     -> 站立脚零速（用作约束）
formulateBaseXYLinearAccelTask   -> 跟踪 base XY 加速度
formulateBaseHeightMotionTask    -> 跟踪 base 高度
formulateBaseAngularMotionTask   -> 跟踪 base 角运动
formulateCenterOfMassTask        -> 跟踪 CoM 加速度
formulateSwingLegTask            -> 跟踪摆动腿
formulateContactForceTask        -> 接触力跟踪 (来自 MPC inputDesired)
formulateJointAccelTask          -> 跟踪 MPC 期望关节加速度
formulateArmJointAccelTask       -> 跟踪手臂关节加速度
```

这里特别重要的一个点：**WBC 跟踪的不是 MPC 的“位置”，而是“加速度 / 接触力”**。你能在 `WbcBase::update` 里看到：

```text
// 上一拍 input -> 当前 input 做有限差分，作为 q̈_des / Ḟ_c 参考
vd_des_fd_ = (qd_des_ - last_v_des_) / period;
```

也就是说从 MPC 给的 (stateDesired, inputDesired) 到 WBC 的 task：

```text
q_des, v_des    ←  MPC stateDesired
F_c_des         ←  MPC inputDesired (前 3*numContacts 项)
qd_des          ←  MPC inputDesired (后 actuatedDofNum 项, 关节速度)
q̈_des           ←  (qd_des - last_v_des_) / dt          ← 一阶差分 (有限差)
                  这就是“线性插值/有限差分得到加速度参考”
```

这一步才是你直觉里那个“线性插值得到一系列 input”——它确实出现，但出现在 **WBC 之前的有限差分**，目的是把 MPC 给的速度序列转成 q̈ 期望值，作为 WBC QP 的 b。

## 5. 把两条线路的“QP 数学”并排写出来

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

## 6. 把两条链路放一起的差异表

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

## 7. 计算量上的对比

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

## 8. 一句话总结你那句直觉

> 不是“线性插值得到 input”，而是：reference anchor 在 MPC 网格上做线性插值（姿态 slerp）得到目标值，SQP 在此基础上把非线性 OCP 局部化成 QP 求解，得到 state/input。KMPC 到这一步就结束；人形下肢还会再把 MPC 的 (state, input) 经过一次有限差分/雅可比映射成 q̈、F_c 目标，由 WBC 的 qpOASES QP 解出最终关节力矩。