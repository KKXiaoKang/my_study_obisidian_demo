## Task Target
*给定一系列图片和对应的视角，对三维场景进行建模，从而实现任意视角的图片渲染*

## 3D - GS 构型/与NeRF的计算对比
#### 隐式辐射场
* NeRF 的核心思想是使用神经网络来建模场景中每个点的辐射强度和深度，通过训练一个 MLP 实现从<三维坐标和观测方向> 到 <颜色和密度> 的映射 
* 渲染过程如下
	- 1. 对于给定像素，NeRF首先根据相机参数确定了射线（Ray）的方向和起点。
	- 2. 射线采样。然后，它在这个射线上均匀采样多个点。
	- 3. 射线积分。对于每个采样点，NeRF通过将采样点过 MLP（多层感知机）的结果（颜色和密度）结合起来，来计算射线通过该点的积分。这个积分结果就是该像素最终的颜色值。
	- *数据升维*
	- 4. 合成图像。通过对所有像素执行相同的步骤，NeRF 可以合成一幅 2D 图像。
![[Pasted image 20250424113031.png]]图 1 NeRF 示意图，图源于《NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis》。左侧是渲染得到像素值的示意图。右侧表示在渲染结束后可以直接和 GT 像素值算 loss。

#### 显式辐射场和 3D GS
* 相比于 NeRF 等隐式场景表示方法，显式辐射场通常用体素、点云等方式建模三维场景。3D GS 属于显示辐射场，用 3D 高斯建模三维场景，同时利用了神经网络的特性进行参数优化。它利用 3D 高斯作为灵活高效的表示方法，充分利用了两种方法的优势。这些高斯经过优化，能准确地表示场景，将基于神经网络的优化和显式结构化数据存储的优势结合在一起。这种融合方法旨在以更快的训练速度和实时性能实现高质量的渲染，尤其适用于复杂场景和高分辨率输出。

## 3D - GS 架构/数据流
#### 3D GS球的构建 - 椭球的构建
![[Pasted image 20250427110800.png]]
##### 3D高斯球的属性
* 中心（位置） μ 、不透明度 α 、3D协方差矩阵 Σ 和颜色$c$ 
*  $c$ 由球谐函数表示，以实现视角依赖的外观。所有属性都是可学习的，并通过反向传播进行优化  [对应代码](https://link.zhihu.com/?target=https%3A//github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py%23L64C1-L69C27)
##### 每个3D高斯球体由以下参数构成³ ⁷：
* **位置 (μ)**：三维坐标 $(x,y,z)$，通常从SfM（运动恢复结构）点云初始化。
    * 对于深度相机RGB-D来说，对于像素坐标系$(u,v)$ 以及$d$ ，其中K为相机内参矩阵，坐标$\boldsymbol{\mu}$ 表示公式如下
    * $$ \mu = d \cdot K^{-1} \cdot [\boldsymbol{u}, \boldsymbol{v},\boldsymbol{1}]^T$$
* **协方差矩阵 (Σ)**：描述高斯椭球的形状和方向，通过缩放矩阵 $S$ 和旋转矩阵 $R$ 构建：
  $$\Sigma = R \cdot S \cdot S^T \cdot R^T$$
  其中:
  ○ $S$ 是缩放矩阵，由三维向量 $(s_x, s_y, s_z)$ 构成对角距阵。
  ○ $R$ 是旋转矩阵，由四元数（quaternion）转换而来，确保优化过程中协方差矩阵的半正定性。
* **不透明度 (α)**：标量值，控制高斯球体的透明度。
* **颜色 (c)**：通过球谐函数（Spherical Harmonics, SH）表示视角相关的颜色，公式为：
  $$c = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{l,m} \cdot Y_{l,m}(\theta, \phi)$$
  其中 $Y_{l,m}$ 是球谐基函数，$(\theta, \phi)$ 是观察方向的角度，$c_{l,m}$ 是待优化的系数。
#### 3D GS 渲染过程
* 3D GS首先将空间中的3D高斯投影到基于像素的图像平面上，这个过程被称为泼溅（splatting）。随后，3D GS对这些高斯进行排序并计算每个像素的值。不难看出，**NeRF 和3D GS的渲染可以被视为彼此的逆过程**
* 接下来使用3D高斯进行可微分渲染，3D高斯作为场景表示中的最小元素
![[Pasted image 20250425175500.png]]

* **[视锥剔除](https://zhida.zhihu.com/search?content_id=238991668&content_type=Article&match_order=1&q=%E8%A7%86%E9%94%A5%E5%89%94%E9%99%A4&zhida_source=entity)**。给定指定的相机姿势，此步骤确定哪些3D高斯位于相机的视锥体之外。这样做可以确保在后续计算中不涉及给定视图之外的3D高斯，从而节省计算资源 。[对应代码](https://link.zhihu.com/?target=https%3A//github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu%23L193C1-L194C10)。

* **泼溅（splatting）**。在这一步骤中，3D高斯（椭球）被投影到2D图像空间（椭圆）中进行渲染，如图 2a 所示。给定视图变换 $W$  和3D协方差矩阵 $\sum$  ，投影的2D协方差矩阵 $\sum'$     可以使用以下公式计算： $\sum'=JW\sum W^{T}J^{T}$   , 其中 $J$ 是射影变换的仿射近似的雅可比矩阵。 [对应代码](https://link.zhihu.com/?target=https%3A//github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu%23L156)。

* **像素渲染方式 - 以像素为单位进行渲染**
    * 给定像素位置$x$ ，可以通过视图变换$W$计算出它与所有重叠高斯的距离，即这些高斯的深度，形成一个排序的高斯列表$N$  然后采用alpha合成来计算这个像素的最终颜色
    * 这个像素的最终颜色
        * $$C=\sum_{i \in N}C_i \alpha_i' \prod_{j=1^{i-1}}{1-\alpha_j'}$$
    * 其中$C_i$ 是学习到的颜色，最终的不透明度$\alpha_i'$ 是学习到的不透明度$\alpha_i$ 和高斯分布的乘积结果（[对应代码](https://link.zhihu.com/?target=https%3A//github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu%23L343)）
        *  不透明度计算公式 $\boldsymbol{\alpha}$
            * $$\alpha_i' = \alpha_i \times \exp \left(-\frac{1}{2}(\boldsymbol{x}' - \boldsymbol{\mu}_i')^T \boldsymbol{\Sigma}_i'^{-1} (\boldsymbol{x}' - \boldsymbol{\mu}_i') \right)$$
            *  其中$\boldsymbol{x}'$ 和$\boldsymbol{\mu}'$ 是投影空间中的坐标
    *  值得担忧的是，与NeRFs相比，上述的渲染过程可能更慢，因为生成所需的有序列表难以并行化。实际上，这种担忧是合理的；当使用这种简单的逐像素方法时，渲染速度可能会受到显著影响。为了实现实时渲染，3D GS做出了几个妥协，以适应并行计算
* 适应并向计算的方法
    * **瓦片（图像块）** 
    * ![[Pasted image 20250427101838.png]]
        * 3D GS最初将图像划分为多个不重叠的图像块，这些图像块在原始论文中被称为瓦片（tiles）。每个块包含16 × 16像素。3D GS进一步确定哪些图像块与这些投影的高斯（椭圆）相交
        * [代码实现](https://link.zhihu.com/?target=https%3A//github.com/guikunchen/3DGS_NOTES/blob/main/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu%23L232)中不是用椭圆而是用圆作近似。考虑到一个投影的高斯可能覆盖多个块，一种合理的方法是复制高斯，为每个副本分配一个标识符（即块ID），如图 2c 所示

	 * **并行化渲染**。复制后，3D GS 会将各自的块 ID 与每个高斯视图变换得到的深度值结合起来。这样就得到了一个未排序的字节列表，其中高位代表块 ID，低位表示深度。这样，排序后的列表就可以直接用于渲染。图 2cd 提供了该概念的直观演示。值得强调的是，每个块和像素的渲染都是独立进行的，因此这一过程非常适合并行计算。另外一个好处是，每个块的像素都可以访问一个公共的共享内存，并保持一个统一的读取序列，从而提高渲染的并行执行效率。在原论文的官方实现中，该框架将块和像素的处理分别视为类似于 CUDA 编程架构中的block和thread



#### 3D GS 优化过程
* 3D GS的核心是它的优化程序。通过构建大量 3D 高斯集合， 3D GS 可精确捕捉场景的细节，从而促进新视图合成。一方面，3D 高斯的属性可通过可微渲染进行优化，以适应特定场景的纹理。另一方面，能够很好地表现给定场景的 3D 高斯的数量是未知的。一种可行的方法是让神经网络自动学习 3D 高斯的密度。接下来，我们先介绍如何优化每个高斯的属性，然后介绍如何控制场景中 3D 高斯的密度。这两个过程在优化工作流程中交错进行。由于优化过程中有许多手动设置的超参数，为了清晰起见，我们省略了大部分超参数的符号，并给出其在代码中的位置。
##### 参数优化
* **损失函数**：定义3D GS的损失函数为渲染图像与给定图像之间的差异（图像级别的差异）
    * $$ \mathcal{L} = (1 - \lambda)\mathcal{L}^1 + \lambda\mathcal{L}{D-SSIM} $$
    *  其中$\lambda$ 是一个权重因子
    * $\mathcal{L}_1$ 损失和D-SSIM项目是标准度量   
    * 3D GS 的损失函数与 NeRF 的略有不同。由于射线步进计算成本高昂，NeRFs通常在像素级别而不是图像级别进行 loss 的计算
* **参数更新**： 3D高斯的大多数属性可以通过反向传播直接优化
    * 直接优化协方差矩阵$\sum$ 可以得到非半正定矩阵， 这不符合通常与协方差矩阵相关联的物理解释
    * 为了避免这个问题，3D GS 选择优化一个四元数$\boldsymbol{q}$ 和一个三维向量$\boldsymbol{s}$ 
        *  $\boldsymbol{q}$ 表示旋转
        *  $\boldsymbol{s}$ 表示缩放
        *  该方法允许协方差矩阵$\sum$ 重建为  $\Sigma = R \cdot S \cdot S^T \cdot R^T$
            * $\boldsymbol{R}$ 表示从 $\boldsymbol{q}$ 得到的旋转矩阵
            * $\boldsymbol{S}$ 表示从 $\boldsymbol{s}$ 得到的缩放矩阵
    *  为了避免自动微分的成本，3D GS 推到了$\boldsymbol{q}$ 和$\boldsymbol{s}$的梯度，以便在优化过程当中直接计算它们
    * 推导过程可参考 3D GS 原论文的附录A。[对应代码](https://link.zhihu.com/?target=https%3A//github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/backward.cu)

#### 后续优化及展望