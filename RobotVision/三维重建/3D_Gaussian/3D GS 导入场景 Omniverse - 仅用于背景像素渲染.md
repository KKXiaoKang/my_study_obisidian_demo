### Omnivese可视化
基于导入omnivese平台的工作，可以参考Omniverse 3D 高斯分层扩展
- [Omniverse 3D 高斯分层扩展](https://github.com/j3soon/omni-3dgs-extension/tree/master?tab=readme-ov-file#)
- 主体架构
    - （1）VanillaGS - 作为后端渲染器
    - （2）Isaac-Lab - 作为前端可视化插件
* ![[Pasted image 20250430113713.png]]
> 注意，该工作基于Docker镜像工作，IsaacLab仅仅作为可视化RGB前端 
*  优点：作为静态场景可以很快的引入到IsaacLab的场景当中，具有高拟真的RGB信息
 * 缺点：不具备Depth深度信息，在场景当中无法拥有碰撞属性进行交互（在IsaacLab当中需要使用mesh文件作为碰撞的交互信息）
### GS-To-PC
> [3DGS-to-PC: Convert a 3D Gaussian Splatting Scene into a Dense Point Cloud or Mesh](https://arxiv.org/abs/2501.07478)
* 高斯喷溅可以生成场景的极高质量 3D 表示。然而，要正确查看这种重建，需要专门的 Gaussian 渲染。此外，许多 3D 处理软件与 3D 高斯不兼容...但大多数与点云兼容。
* 3D-GS高斯散点ply
    *  ![[Pasted image 20250430113300.png]]
* 运行示例，具体可查看arg表
```bash
python3 gauss_to_pc.py --input_path data/input/point_cloud.ply --no_render_colours --visibility_threshold 0.5 --poisson_depth 50 --colour_quality ultra

python3 gauss_to_pc.py --input_path data/input/point_cloud.ply --transform_path data/input/transforms.json --colour_quality ultra

# ok
python3 gauss_to_pc.py --input_path data/input/point_cloud.ply --no_render_colours
```
* 转换为PC之后的，带颜色的ply点云效果
    *  ![[Pasted image 20250430113427.png]]

* 但是后续该官方作者提供的重建mesh的质量十分不好，推荐还是使用`SuGaR`进行优化

