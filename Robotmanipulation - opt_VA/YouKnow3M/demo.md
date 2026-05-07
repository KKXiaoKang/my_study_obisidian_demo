#### curobo 的双臂灵巧手6D姿态估计与避障抓取系统
* 基于cuRobo实现GPU并行运动规划，将机械臂规划耗时压缩至~50ms并支持在线重规划，相比传统采样方法提速一个数量级。构建ESDF Voxel感知-规划闭环并完成语义对齐，解决感知与规划不一致问题。
* 提出抓取阶段感知冻结机制(Go/Back状态机)，避免"自遮挡"导致的规划失败，提升抓取成功率。完成cuRobo在Movelt2中的工程级落地(无侵入替换+多线程架构+可视化调参).

#### kmpc的VR高动态高响应遥操系统开发
* Quest3构建VR遥操作通信链路，设计socket+protobuf数据协议，实现低延迟姿态与控制指令传输。构建100Hz实时控制回路，采用 OCS2 SQP-MPC(结合CppAD自动生成解析雅可比)在复杂约束场景下相比 DDP具有更快收敛与更稳定求解性能
* 设计异步MPC+MRT双线程控制架构，MPC后台持续优化全局策略，主线程高频插值输出控制指令，实现"VR帧无需等待MPC求解"的低延迟响应。在MPC单次求解耗时30-50ms条件下，仍可保证控制输出平滑连续，显著降低遥操作抖动并提升系统实时性与稳定性。

#### MPC+WBC机器人控制器，适配lsaacSim4.5.0平台搭建具身benchmark
* 基于OCS2最优控制框架，搭建统一机器人控制接口，适配Isaac Sim，构建可扩展的具身智能仿真Benchmark支持多任务评测(抓取，操作等)与策略对比实验，
* 提供统一数据接口与评测指标，加速算法迭代。
* 基于3D-GS生成高保真的3D数字资产，可直接导入到lsaacSim作为usd文件

#### 基于数据驱动的密堆料框搬运工业落地场景下的具身模型VLA架构创新 
* Task-decomposed Multi-HeadAction Decoder(按运动学语义解耦的多头动作解码器)
* Homoscedastic Uncertainty-Weighted Multi-Task Loss(基于同方差不确定性的可学习多任务损失加权).
* Bimanual Cross-Attention with Geometry-Preserving Decoupling(带几何约束保护的双臂跨注意力解码器)

#### HIL-SERL在IsaacSim平台的仿真验证与落地.Env适配(HIL-Serl接到IsaacLab+ROS+Kuavo)
SAC PPO奖励消融实验，在某些奖励项上MSE会更适合SAC算法收敛 Q-chunking探索(核心：动作序列联合概率+n-step backup) n-step Obs探索(观测历史窗口)
.Warm-up BCActor探索(用模仿学习给SACactor热身)
亍