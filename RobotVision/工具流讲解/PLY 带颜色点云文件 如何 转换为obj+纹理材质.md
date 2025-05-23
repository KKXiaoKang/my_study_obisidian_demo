## pipeline
*  使用单目SLAM`MASt3R-SLAM` 生成带颜色的ply点云文件
*  通过open3d泊松重建为mesh面，带颜色的obj文件
*  使用meshlab软件提取obj当中的颜色纹理特征

### 构建带颜色的ply点云文件
*  带RGB的点云文件
![[Pasted image 20250514171543.png]]

### 构建泊松曲面
* RGB颜色不直接参与几何重建，但可在重建后映射到网格顶点。若需颜色影响几何，需扩展方法（如调整法向量权重或引入颜色一致性约束），但非标准泊松公式内容
* 参考：原始方法见 Kazhdan et al. (2006), "Poisson Surface Reconstruction"
##### 1）输入数据
* 假定点云。 每个点的位置为 $\boldsymbol{P}_i \in R^3$ 和 法向量$\boldsymbol{n}_i \in R^3$
##### 2) 构建向量场
*  通过法向量的核卷积生成平滑向量场$V(x)$ 
* $$ V(x)=\sum_{i=1}^{N} \boldsymbol{n}_i \cdot g(x-\boldsymbol{p}_i)$$
* 其中g为平滑核函数（如高斯核）, N为点数
##### 3）构建泊松方程
* 求解隐式函数$\boldsymbol{X}(x)$ ，使其梯度逼近$\boldsymbol{V}$，转换为泊松方程
* $$\nabla^2 x = \nabla \cdot \boldsymbol{V} $$
* 其中$\nabla^2$ 为拉普拉斯算子，$\nabla$为散度算子

##### 4）数值求解
* 离散化方程，通常在八叉树结构上使用有限元法，构建线性系统
* $$\boldsymbol{L}_\boldsymbol{X} = \boldsymbol{b}$$
* $\boldsymbol{L}$ 为离散拉普拉斯矩阵
* $\boldsymbol{b}$ 为离散散度场（由$\nabla \cdot \boldsymbol{V}$计算） 
* 解$\boldsymbol{X}$ 通过共轭梯度法迭代求得

##### 5) 提取等值面
* 对解$\boldsymbol{X}$ 应用移动立方体算法，提取等值面（如$\boldsymbol{X} = 0$）作为重建曲面


# 如何制作物体模型的UV纹理图

## Step 1：打开文件
使用meshlab打开一个具有颜色信息的物体模型。这里我们选择的物体模型是物体位姿估计领域里基准数据集Linemod（LM）里的Cat物体（obj_000006.ply）。可以发现，meshlab里显示物体模型文件中包含顶点（vertex）坐标数量和面片（face）数量。
![[Pasted image 20250514175234.png]]

在VsCode里，我们可以查看obj_000006.ply里包含的数据，包括顶点坐标、顶点颜色、顶点法向量和面片。
![[Pasted image 20250514175249.png]]

## Step 2：制作UV map

在Filters中选择Texture -> 在Texture里选择Parametrization：Trivial Per-Triangle。这一步是为了生成一个与物体表面顶点坐标对应的UV map。
![[Pasted image 20250514175300.png]]

一般物体的模型用默认设置即可，如果模型的顶点数量过多，需要增大Texture Dimension。
![[Pasted image 20250514175310.png]]

## Step 3：顶点颜色 -> UV颜色

在Texture中选择Transfer：Vertex Color to Texture，这能够将物体的顶点颜色转换到Step 2中生成的UV纹理图。
![[Pasted image 20250514175318.png]]


在转换时，需要设置UV纹理图的长宽，最好设置成与Texture Dimension一样的数字。
![[Pasted image 20250514175332.png]]


转换后的物体纹理模型如下：
![[Pasted image 20250514175436.png]]
## Step 4：保存文件

由于上述的操作制作的是wedge的UV纹理图，接下来我们使用Convert PerWedge UV into PerVertex UV来制作Vertex的UV纹理图。
![[Pasted image 20250514175513.png]]

## Step 5：导出模型

这一步，我们将物体模型导出。
![[Pasted image 20250514175529.png]]

导出时，按照下图中的勾选。
![[Pasted image 20250514175555.png]]

导出后，你可以在物体模型文件的同级地址下看到尾缀为tex.png的模型纹理文件。
![[Pasted image 20250514175607.png]]

这里我们可视化了png文件如下：
![[Pasted image 20250514175618.png]]

用Vscode打开的物体模型文件如下：
![[Pasted image 20250514175629.png]]