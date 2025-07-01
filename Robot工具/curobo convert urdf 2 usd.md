* 测试issac sim 4.2的是否可以正常跑起来curobo
```bash
omni_python_42 examples/isaac_sim/mpc_nvblox_example.py
```
* 测试直接转换
```bash
omni_python_42 examples/isaac_sim/util/convert_urdf_to_usd.py --robot biped_s45.yml --save_usd
```
* 可以看到目录下出现了一个130M的超大usd文件，该文件可以直接使用
```bash
lab@lab:~/GenDexGrasp/curobot_ros_ws/curobo/src/curobo/content/assets/robot/biped_s45$ du -sh *
32K     biped_s45_fix_head.urdf
44K     biped_s45_gazebo.urdf
4.0K    biped_s45_gazebo.xacro
32K     biped_s45.urdf
130M    biped_s45.usd
4.0K    CMakeLists.txt
8.0K    config
564K    kuavo4.5.png
12K     launch
30M     meshes
4.0K    package.xml
16K     rviz
32K     xml
```