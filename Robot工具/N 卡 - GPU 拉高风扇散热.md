![[Pasted image 20250919151544.png]]
## 控制命令如下，下次重启后失效
```bash
# 调整显卡风扇速度 
## 调整显卡为手动控制风扇 
sudo nvidia-settings -a "[gpu:0]/GPUFanControlState=1" 

## 接下来将两个风扇的转速都调到100% 
sudo nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=100" 
sudo nvidia-settings -a "[fan:1]/GPUTargetFanSpeed=100" 

## 恢复为自动风扇控制 
sudo nvidia-settings -a "[gpu:0]/GPUFanControlState=0
```