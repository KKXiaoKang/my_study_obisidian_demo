# 1. quest3 socket机制设计
* protobuf 信息格式定义
![[Pasted image 20260506175456.png]]

# 2.  OCS2 SQP-MPC + CppAD codegen + 100Hz 实时回路
- 求解器选 SQP（默认）而非 DDP，对这种带不等式约束的 IK-MPC 收敛更快
- CppAD 自动生成解析雅可比（`auto_generated/kuavo`），第一次启动慢、之后每次解每个 SQP 子问题都是 ms 级
- 100 Hz 主回路 + MPC 异步线程（`mpcThread_`，见 base.h 第 181 行），真正做到"VR 帧到来时立即采样最新 MPC 解"

# 3. 异步 MPC + MRT 双线程：VR 帧不等 MPC 求解
* `MPC_MRT_Interface` 是 OCS2 经典的 MPC + Model Reference Tracking 模式：MPC 在自己线程里慢慢求一个完整最优策略，主线程按 1ms / 10ms 在策略里插值出当前参考送下去 → 即便 MPC 求解一次要 30~50ms，VR 端拿到的"下一帧 next_state"始终是平滑的、低延迟的。

![[Pasted image 20260506183040.png]]![[Pasted image 20260506183049.png]]

# demo
![[Pasted image 20260507164023.png]]