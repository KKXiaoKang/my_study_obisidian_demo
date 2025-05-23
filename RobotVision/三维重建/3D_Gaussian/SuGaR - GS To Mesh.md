###  SuGaR - GS To Mesh
![[Pasted image 20250430173604.png]]
>  该库强行依赖于cuda11.8，需要手动安装cuda11.8
* 更改配置，将`12.2`更改为`11.8`
```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_PATH=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/home/lab/GenDexGrasp/Gendexgrasp_ros/devel/lib:/opt/ros/noetic/lib:/opt/ros/noetic/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64
```
* 运行代码进行优化和Mesh导出
```bash
python3 train_full_pipeline.py -s /home/lab/3D_Gaussian_Splatting/gaussian-splatting/data/crane -r dn_consistency --high_poly True --export_obj True --gs_output_dir /home/lab/3D_Gaussian_Splatting/gaussian-splatting/output/d9a68247-f
```
*  优化过程如下
    * 参数讲解
    * 每套参数包含了`最小值` ,`最大值`,`平均值`,`标准差`
        *  `Points` ：高斯体的3D中心坐标
        * `Scaling factors` ： 控制每个高斯体的大小/尺寸
        * `Quaternions` ： 控制每个高斯体的旋转
        * `Sh coordinates dc` ：球谐函数DC分量/常数分量
        * `Sh coordinates rest` ： 球谐函数其他分量
        * `Opacities` ： 每个高斯体的透明度/不透明度值
        * `采样高斯体数量` : 用于SDF正则化采样的高斯体数量
```bash
-------------------  
Iteration: 14000  
loss: 0.064584 [14000/15000] computed in 0.45747623046239216 minutes.  
------Stats-----  
---Min, Max, Mean, Std  
Points: -16.064926147460938 40.806396484375 2.9408371448516846 5.874120235443115  
Scaling factors: 6.927535878276103e-09 15.766653060913086 0.0482836589217186 0.14460621774196625  
Quaternions: -0.9993107318878174 0.9999227523803711 0.1838420033454895 0.46497684717178345  
Sh coordinates dc: -2.473360538482666 9.624624252319336 0.10749077796936035 1.415495753288269  
Sh coordinates rest: -0.6686505079269409 0.5529451966285706 0.002502029063180089 0.05996257811784744  
Opacities: 5.563734521274455e-05 1.0 0.5804076790809631 0.4402596354484558  
Number of gaussians used for sampling in SDF regularization: tensor(29991, device='cuda:0')  
-------------------
```

*  可视化
```bash
meshlab output/refined_ply/crane/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.ply 
```
![[Pasted image 20250430172645.png]]

### IsaacLab - converMesh
* 包含如下三种转换插件
- **URDF 导入器** - 从 URDF 文件导入资产。
- **MJCF 导入器** - 从 MJCF 文件导入资产。
- **网格导入器** - 从各种文件格式（包括 OBJ、FBX、STL 和 glTF）导入资产。
#### 下面为converMesh插件
```bash
# run the converter
./isaaclab.sh -p scripts/tools/convert_mesh.py scripts/tools/refined_mesh/facility/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.obj source/isaaclab_assets/data/facility/facility.usd \
  --make-instanceable \
  --collision-approximation convexDecomposition \
  --mass 1.0
```
![[411909508ee54e2a0aa6161ae844a8cf.jpg]]![[e2d751ec94623ead6aa704c3d82af96b.jpg]]

