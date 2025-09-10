* Deep Reactive Policy: Learning Reactive Manipulator  Motion Planning for Dynamic Environments
* 


---
## 1. Dynamic Closest Point RMP (DCP-RMP) 的思想

RMP（Riemannian Motion Policy）本身是用几何和动力学一致性来构造运动策略的框架。传统 RMPs 通常依赖于静态障碍物或者已知的几何信息。但在真实世界环境中，障碍物往往是**动态的、快速移动的**，这就需要额外的反应式机制。

**DCP-RMP 的关键思想**：

- **动态最近点原则**：实时计算机器人末端执行器或身体与障碍物之间的最近点对 (xr,xo)(x_r, x_o)。
    
- **动态反应力**：
    
    - 如果障碍物正在接近机器人（即速度方向朝向机器人），则产生一个“排斥力”项 fr(xr,xo)f_r(x_r, x_o)，推动关节目标远离障碍物。
        
    - 如果障碍物正在远离机器人（即速度方向背离机器人），则该力不激活（零贡献）。
        
- **作用目标**：这个力不会直接修改机器人轨迹，而是通过 **修改 joint goal qs→qmgq_s \to q_{mg}** 的方式，让机器人能**快速局部避障**。
    

这样，DCP-RMP 相当于一个 **动态障碍局部修正模块**，它保证机器人在执行高层次策略时不会撞上突然出现或快速靠近的障碍物。

一句话总结：  
👉 **DCP-RMP 是一个基于最近点和速度方向的动态避障力场，用于实时修正机器人关节目标，增强对快速移动障碍物的反应能力。**

---

## 2. 论文整体工作流 (Deep Reactive Policy, DRP)

从图和文字描述来看，整体 pipeline 可以分为两个部分：

### (a) 局部反应式避障 (DCP-RMP)

- 输入：Joint goal qsq_s、当前关节位置 qcq_c、环境点云。
    
- 输出：修正后的 joint goal qmgq_{mg}。
    
- 功能：利用 DCP-RMP 模块对 joint goal 动态调整，确保不会直冲障碍物。
    

### (b) 全局运动规划模仿 (IMPACT)

- 基于 ACT (Action Chunking Transformer) 的模仿学习框架。
    
- 输入：
    
    - 当前环境的点云 Ps,PrP_s, P_r（通过 PointSet Abstraction 编码），
        
    - 当前 joint position qcq_c，
        
    - 修正后的 joint goal qmgq_{mg}。
        
- 网络结构：
    
    - Transformer encoder–decoder 结构
        
    - Learnable action tokens
        
    - MLP 预测 action sequence
        
- 输出：一系列机器人动作（关节控制信号），用于实时执行。
    

### (c) 完整 DRP 工作流

1. **感知输入**（点云、关节状态）。
    
2. **DCP-RMP 局部修正**（动态障碍物 → 修正 joint goal）。
    
3. **IMPACT Transformer**（将修正后的目标与场景点云一起输入 → 预测动作序列）。
    
4. **机器人实时执行**。
    

---

## 3. 对比已有方法的贡献

- 传统 **Geometric Fabrics** 依赖精确的障碍物信息（privileged knowledge），但实际点云输入不具备这些信息。
    
- 本文通过 **IMPACT**（基于点云的 Transformer 模仿学习）+ **DCP-RMP**（实时避障修正） → 在真实动态环境中实现快速、安全的反应式运动策略。
    

---

要不要我帮你画一个简化的 **流程图**，用文字框把 DCP-RMP 和 IMPACT 的关系画出来？这样你能更直观地看到论文 pipeline。