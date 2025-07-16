### 硬件状态
* bt2pro手柄本体 - 自然关机状态
* bt2pro手柄接收器 - 插入到电脑中
* 端口检查  - 显示`/dev/input/js0`一直存在
```bash
# lab @ lab in ~/kuavo-ros-control on git:KangKK/fix/merge_kmpc_for_rlpd x [10:12:25] 
$ ls -la /dev/input/js*  
crw-rw-r--+ 1 root input 13, 0 7月  16 09:35 /dev/input/js0
```
* 检查端口数据
![[Pasted image 20250716101436.png]]
* 检查设备地址
```bash
# lab @ lab in ~/kuavo-ros-control on git:KangKK/fix/merge_kmpc_for_rlpd x [10:14:12] C:130
$ cat /proc/bus/input/devices | grep -A 5 -B 5 js  
I: Bus=0003 Vendor=20bc Product=507f Version=0111
N: Name="BEITONG  BEITONG A1N3 BFM DONGLE "
P: Phys=usb-0000:00:14.0-8/input0
S: Sysfs=/devices/pci0000:00/0000:00:14.0/usb1/1-8/1-8:1.0/0003:20BC:507F.0003/input/input7
U: Uniq=
H: Handlers=event7 js0 
B: PROP=0
B: EV=1b
B: KEY=4000 0 7fff000000000000 0 0 0 0
B: ABS=30627
B: MSC=10
```
### 软件状态
*  `确认/cmd_vel一直在发布`
```bash
# lab @ lab in ~/kuavo-ros-control on git:KangKK/fix/merge_kmpc_for_rlpd x [10:01:28] 
$ rostopic info /cmd_vel 
Type: geometry_msgs/Twist

Publishers: 
 * /humanoid_quest_control_with_arm (http://127.0.0.1:36309/)
 * /humanoid_joy_control_auto_gait_with_vel (http://127.0.0.1:37639/)

Subscribers: 
 * /humanoid_sqp_mpc (http://127.0.0.1:44707/)
 * /nodelet_manager (http://127.0.0.1:35471/)

```

* `确认/joy 数据一直在发布`
```bash
# lab @ lab in ~/kuavo-ros-control on git:KangKK/fix/merge_kmpc_for_rlpd x [10:01:23] 
$ rostopic info /joy
Type: sensor_msgs/Joy

Publishers: 
 * /joy_node (http://127.0.0.1:35847/)

Subscribers: 
 * /nodelet_manager (http://127.0.0.1:43525/)
 * /humanoid_joy_control_auto_gait_with_vel (http://127.0.0.1:35175/)
 * /humanoid_sqp_mpc (http://127.0.0.1:36099/)

```
### 原因分析
* 原因来自于`src/humanoid-control/joystick_drivers/joy/src/joy_node.cpp`当中
	* 手柄设备存在但未开启：你的 jstest 显示所有轴都是0，所有按钮都是off状态，说明手柄确实没有开启
*  `/joy` 话题发布异常数据的原因
	* 在 `joy_node.cpp` 中，当手柄设备打开时，会读取手柄的初始状态事件`JS_EVENT_INIT`
	* - 这些初始事件可能包含手柄关闭时的最后状态，而不是当前状态
	* - 你看到的 axes: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.0, -0.0] 是手柄关闭时的最后状态
	- 这些值被 joy_node 读取并发布到 /joy 话题
```c
// There seems to be a bug in the driver or something where the=
// initial events that are to define the initial state of the
// joystick are not the values of the joystick when it was opened
// but rather the values of the joystick when it was last closed.
```
* `/cmd_vel` 数据来源
	* 在 `joyCallback`函数中，这些异常的轴数据被处理
```c
     joystickOriginAxisTemp_.head(4) << joy_msg->axes[joyAxisMap["AXIS_LEFT_STICK_X"]], 
                                        joy_msg->axes[joyAxisMap["AXIS_LEFT_STICK_Y"]], 
                                        joy_msg->axes[joyAxisMap["AXIS_RIGHT_STICK_Z"]], 
                                        joy_msg->axes[joyAxisMap["AXIS_RIGHT_STICK_YAW"]];
```
	* 然后在 commandLineToTargetTrajectories 函数中，这些值被乘以速度限制参数
```c
     commad_line_target_.head(4) = joystick_origin_axis.head(4).cwiseProduct(limit_vector);
```
- 其中 limit_vector 是 c_relative_base_limit_{0.4, 0.2, 0.3, 0.4}
- 所以最终发布的 cmd_vel 就是这些固定值

### 解决方案

1. **重启手柄设备**：
   ```bash
   # 重新插拔手柄或重启手柄
   sudo rmmod joydev
   sudo modprobe joydev
   ```

2. **检查手柄映射**：
   - 确认手柄的轴映射是否正确
   - 检查 `bt2.json` 配置文件中的映射关系

3. **添加数据验证**：
   - 在 `joyCallback` 中添加数据有效性检查
   - 当检测到异常数据时，不处理或重置为0

4. **修改死区处理**：
   - 增加死区检测的阈值
   - 确保只有真正的手柄输入才会被处理

这个问题主要是由于手柄驱动和ROS joy_node的交互导致的，手柄关闭时的最后状态被错误地当作当前状态处理了。