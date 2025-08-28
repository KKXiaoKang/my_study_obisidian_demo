1. 在miniforge环境外：系统可以正常访问 /lib/terminfo/x/xterm-256color 文件
2. 在conda环境内：conda环境隔离了一些系统路径，导致无法找到terminfo文件
```bash

### 方法1：在conda环境中安装ncurses
conda activate lerobot_rl
conda install -c conda-forge ncurses

### 方法2：设置TERMINFO环境变量
# 临时设置
export TERMINFO=/usr/share/terminfo:/lib/terminfo
# 或者添加到你的shell配置文件中（~/.zshrc）
echo 'export TERMINFO=/usr/share/terminfo:/lib/terminfo' >> ~/.zshrc

### 方法3：创建软链接
# 激活conda环境后
conda activate lerobot_rl
# 创建软链接
ln -s /lib/terminfo $CONDA_PREFIX/share/terminfo
```

* 楼主方法2完成设置，成功可以继续使用clear命令

## 创建软链接指向 - x11可视化
```bash
ls -la /home/lab/ | grep -E "(anaconda|miniforge)"

ls -la /home/lab/ | grep -E "(anaconda|miniforge)"

ls -la /home/lab/anaconda3/envs/lerobot_rl/share/X11/xkb
```
