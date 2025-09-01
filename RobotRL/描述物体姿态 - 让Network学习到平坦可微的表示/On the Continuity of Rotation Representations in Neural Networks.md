### 目标 Target
* 传统描述物体姿态，使用欧拉角或者四元数，在数学证明上有歧义性，且部分区域出现梯度过渡不平滑的原因，对于 Network来说 使用一个连续的 $f(x)$ 去拟合这些梯度错乱陡峭的地方会让神经网络过难收敛
* 作者提出思路：使用旋转矩阵（3x3）的前两列，拼接为一个完整的【1x6】维的向量，可以在数学证明上是梯度平坦的
![[On the Continuity of Rotation Representations in Neural Networks.pdf]]

下面基于论文 **“On the Continuity of Rotation Representations in Neural Networks” (arXiv:1812.07035)** 的结论来回答你的两问。核心关键词是“**连续表示（continuous representation）**”与**可微性/梯度友好**。([arXiv](https://arxiv.org/pdf/1812.07035?utm_source=chatgpt.com "arXiv:1812.07035v4 [cs.LG] 8 Jun 2020"), [CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf?utm_source=chatgpt.com "On the Continuity of Rotation Representations in Neural ..."), [ar5iv](https://ar5iv.labs.arxiv.org/html/1812.07035?utm_source=chatgpt.com "On the Continuity of Rotation Representations in Neural ..."))

好的，我们把论文 **On the Continuity of Rotation Representations in Neural Networks (arXiv:1812.07035)** 的关键点结合公式整理一下，并逐步回答你的问题。

---

## 1. 为什么欧拉角（Euler angles）在梯度上不友好？

欧拉角表示旋转时存在以下问题：

- **周期性与折返**：例如绕 $z$ 轴的角度 $\theta$，$\theta$ 与 $\theta+2\pi$ 表示相同旋转。网络如果直接回归角度值，损失函数会在边界处产生跳变。
    
- **奇异性（Gimbal lock）**：当某一角度取特殊值时（如 pitch = $\pm 90^\circ$），三个旋转轴会发生退化，雅可比矩阵病态，导致梯度不稳定。
    

因此，欧拉角 $\alpha, \beta, \gamma$ 不可能作为一个**全局连续可微的映射**来描述 $SO(3)$。

---

## 2. 为什么四元数（Quaternions）不完全友好？

四元数 $q = (w, x, y, z)$ 满足单位约束 $|q|=1$，其旋转矩阵为：


$R(q) = \begin{bmatrix} 1 - 2(y^2+z^2) & 2(xy - wz) & 2(xz + wy) \\ 2(xy + wz) & 1 - 2(x^2+z^2) & 2(yz - wx) \\ 2(xz - wy) & 2(yz + wx) & 1 - 2(x^2+y^2) \end{bmatrix}$

问题有两个：

1. **双覆盖性**：$q$ 与 $-q$ 表示同一个旋转，网络在优化时会被“两个不同点对应同一标签”干扰，梯度不连续。
    
2. **单位球约束**：需要始终保持 $|q|=1$，这会导致额外的归一化步骤与数值不稳定。
    

---

## 3. 论文提出的解决方案：6D 连续表示

论文证明了：任何低于 5 维的表示都不可能对 $SO(3)$ 给出全局连续嵌入。因此，作者提出了**5D 和 6D 的连续表示**。

### 6D 表示的构造方法

- 从旋转矩阵 $R \in SO(3)$ 取前两列向量：
    
    $r_1 = R[:,0]$, $r_2 = R[:,1]$

    拼接成 6D 向量：
    
    $\text{repr}_{6d} = \begin{bmatrix} r_1 \\ r_2 \end{bmatrix} \in \mathbb{R}^6.$

- 从 6D 表示 $\text{repr}_{6d} = (a_1, a_2)$ 重建 $R$ 时，用 Gram–Schmidt 正交化：

    $b_1 = \frac{a_1}{\|a_1\|},$
    $a_2' = a_2 - (b_1^\top a_2)b_1,$
    $b_2 = \frac{a_2'}{\|a_2'\|},$
    $b_3 = b_1 \times b_2,$
    $R = [\, b_1 \ \ b_2 \ \ b_3 \,].$

这个过程是连续的，因此训练时梯度不会遇到不连续点。

---

## 4. 欧拉角 / 四元数 → Rotation Matrix → 6D 表示

- **欧拉角 → Rotation Matrix**  
    假设欧拉角为 $(\alpha, \beta, \gamma)$，按 Z-Y-X 旋转顺序：
    $R = R_z(\gamma) R_y(\beta) R_x(\alpha).$
    
- **四元数 → Rotation Matrix**  
    使用上面的 $R(q)$ 公式。
    
- **Rotation Matrix → 6D**  
    取前两列向量 $r_1, r_2$ 拼成 $\mathbb{R}^6$。
    
- **6D → Rotation Matrix**  
    使用 Gram–Schmidt 正交化恢复合法的旋转矩阵。
    

---
✅ 总结：

- 欧拉角存在周期性和奇异点；四元数有双覆盖和归一化约束 → 导致梯度优化不稳定。
    
- 论文提出了 **6D 连续表示**（取旋转矩阵前两列），并用 Gram–Schmidt 保证正交。
    
- 如果你已有欧拉角或四元数，可以先转换成 $R$，再提取 6D 表示。
    