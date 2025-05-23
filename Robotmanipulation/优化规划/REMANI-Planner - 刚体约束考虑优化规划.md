*  移动底盘+机械臂带刚体约束考虑优化规划
### System Overview - 系统架构
![[Pasted image 20250331154641.png]]

## A 优化公式
* 为了生成一个平滑、安全、可行的轨迹，让移动机械臂完成特定的任务，我们制定了轨迹生成问题的目标函数如下：
$$
J = \sum_{i=1}^{N} J_{s},_{i} + J_{T},_{i} + J_{p},_{i}
$$
* $J_{s},_{i}$  是轨迹的控制力度, 
  *  $J_{s,i} = \int_0^T q_i^{(s)}(t)\tilde{q}_i^{(s)}(t)dt$
* $J_{T},_{i}$ 是时间调节项，可以使轨迹的执行时间尽可能小
  * $J_{T_1} = \omega_T T_i$ 
* $J_{p},_{i}$ 表示不等式约束$G_d(q_{i,j})<=0$ 的惩罚 , $I_{\Sigma}$ 表示稀疏和密集采样
  * $J_{p},_{i} = \sum_{i=1}^{M} \sum_{d \in D} \omega_d I_{\Sigma} (G_d(q_{i,j}))$  
  


## B 安全约束
####  1. 避障：根据障碍物构建ESDF地图，同时确保碰撞球体中心的ESDF值大于它们各自的半径r
$$ G_{*l}(q, \dot{q}) = r_l - D_{ESDF}(W_{pl:m}(q, \dot{q})), \forall l \in \{R,1,2,\cdots,L\}, \forall m \in \{1,\cdots,m_l\} $$
#### 2.自冲突避免：针对每个刚体的链接之间的冲突，计算每个球体中心的欧式几何距离
$$ G_{*l}(q_m) = r_l + r_i - \|^lp_{l:j} - ^lT_i^ip_{l:j}\|,$$
$$ \forall l' \in \{0,\cdots,l-1\}, \forall l \in \{1,\cdots,L\}, $$
$$ \forall i \in \{1,\cdots,m_l\}, \forall j \in \{1,\cdots,m_l\} $$

## C 可行性限制：重点可行性限制
*  为了确保轨迹不会超过移动机械手的跟踪能力，在机器人的运动规划框架中纳入可行性约束
#### 1. 车轮约束
* 针对左右车轮的角速度和加速度施加限制，分别将它们限制为最大值 $\omega_{w},_{max}$  和 $\alpha_{w},_{max}$


$$ G_{\omega(r)}(\dot{q}, \ddot{q}) = \omega^2_{(r)}(\dot{q}, \ddot{q}) - \omega^2_{r,max},$$
$$ G_{\alpha(r)}(\dot{q}, \ddot{q}, \dddot{q}) = \alpha^2_{(r)}(\dot{q}, \ddot{q}, \dddot{q}) - \alpha^2_{r,max} $$

如果车轮速度为0，则车轮的角速度和加速度变得无穷大，所以需要对最小速度施加约束

$$ G_v(\dot{q}) = v^2_{lim} - \dot{q}^T\dot{q},$$
其中 $v_{min}$ 是一个非常小的值，暂且定义为0.05   

#### 2. 关节约束
* 对于机械臂的每个关节需要施加约束考虑，主体考虑关节角度、速度和加速度的限制

$$ G_{q_l}(q_l) = q_l - q_{l,max}, $$ $$ G_{q_l}(q_l) = q_{l,min} - q_l, $$ $$ G_{\omega_l}(q_l) = \dot{q}_l^2 - \omega_{l,max}^2, $$ $$ G_{\alpha_l}(q_l) = \ddot{q}_l^2 - \alpha_{l,max}^2 $$ 

## D 任务约束
* 末端执行器末端空间下的position与目标位置p之间的欧几里得距离限制
####  位置约束公式如下
$$ G_{lr}(q) = \|^Wp_e - ^Wp_t\|_2 - d_r, $$
* $d_{r}$ 公差距离
* $^Wp_e$  末端执行器position
* $^Wp_t$ 任务目标target-position 

#### 旋转角度约束公式如下
$$ G_{lr}(q) = \arccos\left(\frac{1}{2}\left(\text{tr}\left(^WR_l^T\,^WR_r\right) - 1\right)\right) - d_o $$ 

