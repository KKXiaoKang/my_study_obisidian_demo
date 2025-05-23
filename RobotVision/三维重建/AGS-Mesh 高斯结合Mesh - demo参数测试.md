*   Target
    *  用于深度和法线监督的高斯 splatting 模型的研究论文 DN-Splatter 和 AGS-Mesh，以使用智能手机数据（iPhone）和网格重建进行改进的新视角合成
```bash
# 启动lab仿真
cd /home/lab/kuavo-ros-control-Merge

roslaunch humanoid_controllers load_kuavo_isaac_lab.launch joystick_type:=bt2pro

# 启动运动学mpc及tag
cd /home/lab/kuavo-ros-control

roslaunch ar_control robot_strategies.launch

# 启动代码
cd /home/lab/kuavo-ros-control-Merge/src/kuavo-isaac-sim/Robot_data_collect/IsaacLab_data_collect

python3 tool_collect_grasp_box_traj.py
```