### 基于自己的视频流构建自己的3DGS参数模型,构建自己的3DGS流程如下
# 一、数据预处理
## 克隆仓库/带子仓库
```bash

# 克隆仓库

git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive

  

# 更新submodules

pip install submodules/diff-gaussian-rasterization

pip install submodules/fused-ssim

pip install submodules/simple-knn

```
## 准备一个视频
## 从ffmpeg当中将视频转换为图片

```bash

ffmpeg -i demo_test.MP4 -qscale:v 1 -qmin 1 -vf fps=2 %04d.jpg

```
## 将图片转换到所需的训练集

* 该使用方法依赖于`colmap`开源视觉软件，支持SFM方法重建（SfM部分负责生成稀疏点云和相机姿态）

* `COLMAP`是一款开源的三维重建工具，结合了​​结构从运动（Structure-from-Motion, SfM）​​和​​多视图立体视觉（Multi-View Stereo, MVS）​​两大核心技术，能够从多视角图像中恢复三维场景结构和相机姿态，并生成稠密的三维模型

* [GPU版本的COLMAP安装ubuntu](https://colmap.github.io/install.html)

* glog版本:0.4.0版本`默认ubuntu20.04安装`

```bash

python3 convert.py -s data/crane/

```

* 稠密点云场景构建如下

```bash

0428 14:23:28.802250 523157 feature_extraction.cc:258] Processed file [37/38]

I0428 14:23:28.802270 523157 feature_extraction.cc:261] Name: 0037.jpg

I0428 14:23:28.802273 523157 feature_extraction.cc:270] Dimensions: 1920 x 1080

I0428 14:23:28.802275 523157 feature_extraction.cc:273] Camera: #1 - OPENCV

I0428 14:23:28.802277 523157 feature_extraction.cc:276] Focal Length: 2304.00px

I0428 14:23:28.802282 523157 feature_extraction.cc:280] Features: 9351

I0428 14:23:28.818295 523157 feature_extraction.cc:258] Processed file [38/38]

I0428 14:23:28.818312 523157 feature_extraction.cc:261] Name: 0038.jpg

I0428 14:23:28.818315 523157 feature_extraction.cc:270] Dimensions: 1920 x 1080

I0428 14:23:28.818316 523157 feature_extraction.cc:273] Camera: #1 - OPENCV

I0428 14:23:28.818318 523157 feature_extraction.cc:276] Focal Length: 2304.00px

I0428 14:23:28.818322 523157 feature_extraction.cc:280] Features: 9059

I0428 14:23:28.847848 523123 timer.cc:91] Elapsed time: 0.019 [minutes]

I0428 14:23:28.905055 523194 misc.cc:44]

==============================================================================

Feature matching

==============================================================================

I0428 14:23:28.905278 523195 sift.cc:1426] Creating SIFT GPU feature matcher

I0428 14:23:28.928860 523194 pairing.cc:168] Generating exhaustive image pairs...

I0428 14:23:28.928874 523194 pairing.cc:201] Matching block [1/1, 1/1]

I0428 14:23:30.137749 523194 feature_matching.cc:46] in 1.209s

I0428 14:23:30.150434 523194 timer.cc:91] Elapsed time: 0.021 [minutes]

I0428 14:23:30.182341 523255 incremental_pipeline.cc:251] Loading database

I0428 14:23:30.182845 523255 database_cache.cc:66] Loading cameras...

I0428 14:23:30.182858 523255 database_cache.cc:76] 1 in 0.000s

I0428 14:23:30.182861 523255 database_cache.cc:84] Loading matches...

I0428 14:23:30.185938 523255 database_cache.cc:89] 703 in 0.003s

I0428 14:23:30.185945 523255 database_cache.cc:105] Loading images...

I0428 14:23:30.191421 523255 database_cache.cc:153] 38 in 0.005s (connected 38)

I0428 14:23:30.191428 523255 database_cache.cc:164] Loading pose priors...

I0428 14:23:30.191462 523255 database_cache.cc:175] 0 in 0.000s

I0428 14:23:30.191463 523255 database_cache.cc:184] Building correspondence graph...

I0428 14:23:30.225800 523255 database_cache.cc:210] in 0.034s (ignored 0)

I0428 14:23:30.225873 523255 timer.cc:91] Elapsed time: 0.001 [minutes]

I0428 14:23:30.226913 523255 incremental_pipeline.cc:297] Finding good initial image pair

I0428 14:23:30.434450 523255 incremental_pipeline.cc:321] Initializing with image pair #3 and #36

I0428 14:23:30.436705 523255 incremental_pipeline.cc:326] Global bundle adjustment

I0428 14:23:30.628808 523255 incremental_pipeline.cc:405] Registering image #35 (3)

I0428 14:23:30.628822 523255 incremental_pipeline.cc:408] => Image sees 438 / 5733 points

```
# 二、开始训练
## 开始进行高斯球的参数优化 - 参数优化
```bash

python3 train.py -s data/crane/

```
# （可选）三、可视化渲染（可以使用其他工具进行运行）
* 第三方渲染器[superspl.at](https://superspl.at/editor)
     *  ![[Pasted image 20250429151316.png]]
* 官方工具因为4090D的版本cuda12以上太高，在ubuntu上容易部署编译报错
* 依赖于官方工具`SIBR_viewers`对于生成的高斯网络进行实时渲染器
```bash

# checkout

git checkout fossa_compatibility

  

# Dependencies

sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev

# Project setup

cd SIBR_viewers

cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build -j4 --target install

```