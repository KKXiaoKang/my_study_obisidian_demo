# Trigger
* ![[Pasted image 20251222211643.png]]
* 正常的模型定义Trigger，不是在监测joint limit， self-collision的情况，而是在监测自己**模型自己的不确定性** 

# 处理不确定性(第一层trigger)
* 对于同一组观测$o_t$ ，多次采样动作序列，计算方差， 给出一定安全的方差阈值
* ![[Pasted image 20251222212634.png]]

# Ensemble 不确定性（第二层 trigger）
* ![[Pasted image 20251222212738.png]]

# Dynamics consistency（第三层，最重要）
* ![[Pasted image 20251222212847.png]]

![[Pasted image 20251222212901.png]]

# Trigger触发后：
![[Pasted image 20251222212953.png]]