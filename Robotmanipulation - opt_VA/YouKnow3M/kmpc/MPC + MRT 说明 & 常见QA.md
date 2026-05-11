下面按你的项目代码把链路理清楚，后面用 QA 形式整理可讲点。全程不写“面试”两个字。

## 1. 异步 MPC + MRT 架构

你的链路可以这样理解：

```text
Quest3 / 手柄 / 骨骼追踪
        |
        v
`monitor_quest3.py`
发布 `/leju_quest_bone_poses`、`/quest_joystick_data`
        |
        v
`quest3_node_incremental.py`
- 读取 launch 参数：`incremental_control`、`hand_reference_mode`
- 调用 `Quest3ArmInfoTransformer`
- 做 VR -> 机器人坐标系 / 手臂比例 / 手腕姿态 / 手指映射
- 发布 `/mm/two_arm_hand_pose_cmd`
- 通过 `/mobile_manipulator_mpc_control` 切控制模式
        |
        v
移动机械臂 IK / target manager
生成 `TargetTrajectories`
        |
        v
`MobileManipulatorControllerBase`
        |
        +---------------- 后台线程 ----------------+
        | `mpcThread_`                              |
        | 固定 MPC 频率调用 `advanceMpc()`           |
        | 求解一段未来最优策略 policy                |
        +-------------------------------------------+
        |
        +---------------- 前台控制循环 --------------+
        | `MobileManipulatorController::update()`    |
        | 读取当前 observation                       |
        | 调用 `evaluatePolicy()` 插值当前时刻策略    |
        | 得到 `nextState` / `optimizedInput`         |
        | 发布给 humanoid arm / waist / base         |
        +-------------------------------------------+
```

核心点在 `MPC_MRT_Interface`：

```text
MPC: Model Predictive Control
- 慢线程
- 做非线性优化
- 一次可能 30~50ms
- 产出未来一段时间的最优轨迹 policy

MRT: Model Reference Tracking
- 快线程
- 不重新求解优化问题
- 只在已有 policy 里按当前时间和状态插值 / rollout
- 每 1ms / 10ms 都能输出当前控制参考
```

你的代码中对应关系是：

- `setupMpc()`：创建 `SqpMpc` 或 `GaussNewtonDDP_MPC`，并绑定 `RosReferenceManager`。
- `setupMrt()`：创建 `MPC_MRT_Interface`，启动 `mpcThread_`，后台循环调用 `advanceMpc()`。
- `starting()`：先同步等待第一条 policy，避免前台没有策略可插。
- `update()`：前台每个控制周期读取当前状态，调用 `setCurrentObservation()`，然后 `evaluatePolicy()` 得到 `optimizedStateMrt` 和 `optimizedInputMrt`。
- `MobileManipulatorController::controlHumanoid()`：把 MM 状态/输入转回 humanoid 的 arm、waist、base 控制目标。

### 一个具体例子

假设 VR 是 90Hz，控制前台是 100Hz，MPC 单次求解 40ms：

```text
t = 0ms
MPC 后台拿到当前 observation + target，开始求 P1，预计 40ms 后完成。

t = 10ms
VR 又来了一帧手部目标。
前台不等 MPC。
MRT 使用当前已有 policy P0，在 t=10ms 插值得到 nextState，发给下游。

t = 20ms
前台继续 evaluatePolicy(P0)，输出连续 nextState。
机器人动作不会停住。

t = 40ms
后台 MPC 完成 P1。
`updatePolicy()` 把 P1 换成最新策略。

t = 50ms
前台开始在 P1 上插值。
轨迹自然过渡，不是突然跳到 VR 新目标。
```

所以你可以这样讲：

> 这个系统不是 VR 来一帧就阻塞等一次 MPC 求解，而是把“高耗时优化”和“高频控制输出”解耦。MPC 后台持续优化未来轨迹，MRT 前台在最新可用策略上按当前时间插值输出控制量。这样即使 MPC 一次求解 30~50ms，前台仍然能 100Hz 输出连续 `next_state`，避免遥操作卡顿。

## 2. 控制延迟高、动作抖动怎么解决

我会分四层处理。

第一层是输入侧抗抖。VR 骨骼数据天然有噪声，尤其手腕、手指、肘部会跳。你的 `quest3_utils.py` 已经做了不少处理，比如 `hand_reference_mode=thumb_index` 可以用拇指和食指中点作为更稳定的末端参考；`scale_arm_positions()` 会按机器人臂长缩放；`limit_arm_vector_rotation()` 对上臂向量做角速度限制；`constrain_upper_arm_vector()` 把上臂限制在合理工作空间内。进一步可以加时间戳对齐、低通滤波、四元数 slerp、异常帧丢弃和死区。

第二层是控制模式上使用增量遥操作。直接把 VR 绝对位姿映射到机器人末端，容易因为佩戴姿态、坐标漂移、人体尺度差异造成跳变。增量模式更适合高动态遥操作：按下触发时记录 VR anchor 和机器人当前末端，后续只发送相对位移/相对旋转。这样能减少初始对齐误差，也能避免 VR 世界坐标漂移直接打到机器人上。

第三层是 MPC/MRT 解耦。MPC 求解耗时不可避免，所以不能让控制主循环等待 `advanceMpc()`。你的架构用后台 MPC + 前台 MRT，前台只做 `evaluatePolicy()`，保证固定周期输出。这个是解决“大延迟/卡顿”的关键。

第四层是下游限幅和异常保护。你的控制器里有 `basePoseDeltaLimit_` 限制 base 单次 delta；`limitArmPosition()` 可做关节限位；`anomaly_check()` 会检查 policy 和 target 的误差；launch 里也有 `quest3/enable_safety`、`max_pos_diff`、`max_quat_diff` 参数。实际系统里我会把这些保护打开，并把 service 调用只放在模式切换，不放在高频控制环里。

## 3. QA

**Q：为什么不用 VR 每来一帧就直接求一次 MPC？**  
A：因为 MPC 是优化问题，求解时间不稳定，可能 30~50ms，VR 和控制环不能被它阻塞。我的做法是让 MPC 后台持续求未来一段轨迹，前台 MRT 在最新策略上插值输出当前控制量。这样控制频率由前台决定，MPC 求解慢一点也不会让机器人停顿。

**Q：`MPC_MRT_Interface` 在这里解决了什么问题？**  
A：它把“优化”和“执行”分开。MPC 负责产生完整策略，MRT 负责把策略变成当前时刻的状态和输入。在代码里，后台线程调用 `advanceMpc()`，前台 `update()` 调用 `evaluatePolicy()` 得到 `nextState` 和 `optimizedInput`。

**Q：如果 MPC 新策略还没算完，前台输出什么？**  
A：前台继续使用上一条可用 policy。只要第一条 policy 已经初始化，MRT 就可以在旧策略上按当前时刻插值/rollout，保持输出连续。新 policy 算完后通过 `updatePolicy()` 切换进去。

**Q：为什么第一条 policy 要同步等待？**  
A：因为系统刚启动时前台没有任何可插值的策略。如果直接进入控制循环，`evaluatePolicy()` 没有基础策略，容易输出无效控制。你的 `starting()` 里会循环 `advanceMpc()`，直到 `initialPolicyReceived()` 为真。

**Q：100Hz 控制和 30~50ms MPC 求解是否矛盾？**  
A：不矛盾。100Hz 指的是前台 MRT 输出频率，每 10ms 输出一次控制参考；30~50ms 是后台 MPC 更新完整策略的频率。前台不用每次都等新策略，只需要使用最新可用策略。

**Q：动作抖动主要来自哪里？**  
A：主要有三类：VR 骨骼追踪噪声、人体到机器人 retarget 映射不连续、MPC target 高频跳变。对应处理是输入滤波/异常帧剔除、增量控制和人体尺度映射、MPC/MRT 平滑输出以及下游限速限幅。

**Q：为什么要用 `thumb_index` 作为手部参考点？**  
A：手掌中心在抓取或手腕旋转时可能和实际操作意图不一致。`thumb_index` 用拇指和食指中点作为末端参考，更接近捏取/操作点，遥操作时更自然，也能减少手掌姿态变化带来的末端跳动。

**Q：retarget 映射做了哪些事情？**  
A：主要做坐标系转换、人体胸腔坐标到机器人 base 坐标转换、人体臂长到机器人臂长缩放、肩肘腕几何约束、手腕四元数转换、手指关节映射。`Quest3ArmInfoTransformer` 里这些逻辑比较集中。

**Q：为什么需要限制上臂角速度和工作空间？**  
A：VR 肘部点很容易抖，直接映射会导致机器人肩肘快速摆动。限制上臂向量旋转角速度可以减少高频抖动；限制上臂在合理锥形空间内，可以避免不可达人姿态或自碰撞风险。

**Q：为什么用增量控制会降低延迟感？**  
A：增量控制不是降低物理通信延迟，而是降低“感知上的跳变和拖拽”。它让机器人从当前真实末端开始跟随 VR 相对运动，避免绝对坐标误差导致突然大位移，操作者会感觉响应更直接、更稳。

**Q：服务调用为什么不适合放在高频控制环？**  
A：ROS service 是同步调用，可能阻塞。你的系统里 `/mobile_manipulator_mpc_control` 更适合做模式切换，比如 NoControl、ArmOnly、BaseArm。高频目标应该走 topic / target trajectory，避免控制环被服务延迟卡住。

**Q：`nextState` 为什么不是直接 VR target？**  
A：VR target 是期望末端目标，可能不满足机器人动力学、关节限位、速度限制和当前状态连续性。`nextState` 是 MPC/MRT 根据当前 observation、模型和约束计算出来的下一步可执行状态，更适合下发给 WBC/arm/base 控制器。

**Q：你这个项目最核心的技术点怎么概括？**  
A：基于 OCS2 构建实时 Kinematic MPC 遥操作框架，用 VR retarget 生成移动机械臂目标，通过异步 MPC + MRT 保证高频、低延迟、连续输出，并在输入映射、增量控制、限速限幅和异常检测上处理抖动与安全问题。