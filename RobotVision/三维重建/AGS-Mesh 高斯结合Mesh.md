*   Target
    *  用于深度和法线监督的高斯 splatting 模型的研究论文 DN-Splatter 和 AGS-Mesh，以使用智能手机数据（iPhone）和网格重建进行改进的新视角合成

![[pipeline_ags_mesh.png]]
#### 生成估计法线
```bash
cd /home/lab/3D_Gaussian_Splatting/dn-splatter

# 使用omnidata模型权重生成法线 | 需要提前下载好模型
python dn_splatter/scripts/normals_from_pretrain.py --data-dir dataset/room_datasets/vr_room/iphone/long_capture/ 
```

#### 生成深度掩码
```bash
cd /home/lab/3D_Gaussian_Splatting/AGS_Mesh

# 生成深度掩码
python3 depth_normal_consistency.py --data-dir dataset/room_datasets/vr_room/iphone/long_capture 
```

#### 训练模型
```bash
cd /home/lab/3D_Gaussian_Splatting/AGS_Mesh

# 模型训练（model_path指向生成的深度掩码output的位置）
python train.py -s dataset/room_datasets/vr_room/iphone/ --model_path output/mushroom/vr_room --depth_supervision --normal_supervision 
```

#### 获取深度图，提取mesh网格
```bash
# use isooctree-based mesh extraction
## first get rendered training image
python render.py -m output/mushroom/vr_room -s dataset/room_datasets/vr_room/iphone/ --iteration 30000 --skip_mesh --skip_test


## get mesh with IsoOctree-based method 
### mushroom
python isooctree.py output/mushroom/vr_room/train/ours_30000 --transformation_path dataset/room_datasets/vr_room/iphone/long_capture/transformations_colmap.json --tsdf_rel 0.03 --output_mesh_file output.ply --subdivision_threshold=100
```

#### 生成的网格没有进行强uv纹理强绑定 顶点没有颜色信息
![[Pasted image 20250523143602.png]]
![[Pasted image 20250523143539.png]]