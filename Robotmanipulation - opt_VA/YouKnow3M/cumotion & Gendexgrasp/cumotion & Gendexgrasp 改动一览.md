![[Pasted image 20260506201538.png]]
1. 基于 NVIDIA cuRobo 的 GPU 并行运动规划栈，将 6/7 自由度机械臂规划耗时压至 ~50ms 级，相较 OMPL/RRT 提速一个数量级，支持在线重规划。
2. 打通 nvblox ESDF → cuRobo Voxel 的实时感知–规划闭环，包含距离取反、未观测体素重映射、voxel 中心→表面对齐三处关键修正，保证感知数据语义与论文规范一致。
3. 设计 Go/Back 抓取语义状态机：去程实时刷新 ESDF 避障；回程冻结上一帧地图，避免被抓物体被误判为障碍——解决了纯感知规划在抓取闭环中"自己挡自己"的常见痛点。
4. 以 MoveIt2 MoveGroup Action 形式封装 cuRobo，做到对上层零侵入替换；同时通过双线程 + `MultiThreadedExecutor` 解耦感知刷新与规划触发，保证 action 不被 ESDF 服务阻塞。
5. 完整的可观测性与调参面板：实时发布 cuRobo 内部体素到 RViz、暴露 `time_dilation_factor / voxel_size / max_attempts` 等关键参数，便于不同场景（高架、抓取、转运）现场调优。