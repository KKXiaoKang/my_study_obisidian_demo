* 论文地址 [MASt3R-SLAM](https://arxiv.org/abs/2412.12392)
#### 核心Target
*  提出了一种基于 MASt3R（一种双视图三维重建和匹配先验）自下而上设计的实时单目稠密 SLAM 系统
* 首个使用双视图三维重建先验 MASt3R [20] 作为基础的实时 SLAM 系统
* 点云图匹配、跟踪和局部融合、图构建和回环检测以及二阶全局优化的高效技术
* 一种最先进的密集 SLAM 系统，能够处理通用的、时变的相机模型

#### 稠密点云构图问题
* 从仅有的2D图像进行稠密SLAM需要如下数据【经典高维逆向求解问题】
    * 1) 推理时间变化的姿态
    * 2) 相机模型
    * 3) 3D场景几何
* 通过场景先验模型（人工标注/模型标注）
    * 单视图先验知识priors，如单目深度和法线，试图从单个图像中预测几何形状
    * 光流追踪多视图先验知识，求解图像中的几何形状

#### 论文提到的MASt3R模块用于`双视图三维重建和匹配先验`合几何形状
![[Pasted image 20250513141611.png]]   图 1：我们在 Burghers 序列[55]上从我们的稠密单目 SLAM 系统重建。使用 MASt3R 在左侧显示的两个视图预测，我们的系统即使没有已知的相机模型，也能在实时中实现全局一致的姿态和几何形状。
#### MASt3R网络
* 直接从两张图像在公共坐标系中输出点云图
* 双视图架构以双视图几何为 SfM 的基本构建块
#### pipeline
##### 1) MASt3R 预测和点图匹配 
##### 2) 跟踪和局部融合 
##### 3)回环检测和全局优化
![[Pasted image 20250513143728.png]]    图 3：MASt3R-SLAM 的系统图。新图像通过预测 MASt3R 的点云图并使用我们高效的迭代投影点云图匹配来找到像素匹配，与当前关键帧进行跟踪。跟踪估计当前姿态并执行局部点云图融合。当新关键帧添加到后端时，通过使用编码的 MASt3R 特征查询检索数据库来选择回环关闭候选者。候选者由 MASt3R 解码，如果找到足够数量的匹配，则将边添加到后端图中。大规模二阶优化实现姿态和密集几何的全局一致性
![[Pasted image 20250513153805.png]]


#### 构建过程
##### （1） 准备MP4视频，同时将其分割为png
```bash
ffmpeg -i VID_20250611_165341.mp4 -qscale:v 1 -qmin 1 -vf fps=4 %04d.png
```

##### （2）实时重建ply
 *  无需相机内参，直接使用图片
```bash
python3 main.py --dataset datasets/our_data/new_huawei_scene_dp_train --config config/base.yaml
```
* realsense相机使用相机内参
```bash
python3 main.py --dataset datasets/our_data/realsense_DP_train --config config/base.yaml --calib config/intrinsics_realsense.yaml
```
![[Pasted image 20250611174007.png]]
##### (3) 重建mesh 
###### 先重建为obj文件
* 自用泊松重建
```bash
# lab @ lab in ~/SLAM/MASt3R-SLAM/logs on git:main x [17:38:39] 
$ python3 tool_ply2mesh_only_ply.py new_huawei_scene_act_train.ply --depth 11 --visualize --use_gpu
:: 检测到GPU支持! 可用GPU设备数: 1
:: 加载点云: new_huawei_scene_act_train.ply
:: 原始点云包含 8837722 个点
:: 计算法线...
```
* 商用软件可使用CloudCompare进行泊松重建

###### 使用meshlab烘烤纹理
* #构建带颜色的ply点云文件 