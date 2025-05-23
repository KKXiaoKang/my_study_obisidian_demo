*   Target
    *  用于深度和法线监督的高斯 splatting 模型的研究论文 DN-Splatter 和 AGS-Mesh，以使用智能手机数据（iPhone）和网格重建进行改进的新视角合成

![[pipeline_ags_mesh.png]]
#### 生成估计法线
```bash
cd /home/lab/3D_Gaussian_Splatting/dn-splatter

# 使用omnidata模型权重生成法线
python dn_splatter/scripts/normals_from_pretrain.py --data-dir dataset/room_datasets/vr_room/iphone/long_capture/ 
```