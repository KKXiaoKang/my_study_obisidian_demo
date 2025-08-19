* torchcodec表格来源链接：（https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec）
![[img_v3_02p3_1aca9b99-bb1c-4a8f-9b30-acdb223f032g.jpg]]
```bash
# 更新lerobot
cd lerobot/
pip3 install -e .

# 先确认对cuda下的torchcodec版本
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 根据上表去找torchcodec对应的torch版本
pip install torchcodec --index-url=https://download.pytorch.org/whl/cu124

# 补全一下ffmpeg
conda install ffmpeg -c conda-forge
```

根据代码中的奖励设计，我可以将其总结为以下的LaTeX公式：

$$R = \alpha \cdot \text{scale} \cdot \left( w_{\text{hand}} \cdot \text{MSE}_{\text{hand}} + w_{\text{box}} \cdot \text{MSE}_{\text{box}} + \text{bonus} \right)$$

其中各个组件的详细定义为：

$$\text{MSE}_{\text{hand}} = -\left( \text{MSE}_{\text{left}} + \text{MSE}_{\text{right}} \right)$$

$$\text{MSE}_{\text{left}} = \frac{1}{3}\sum_{i=0}^{2}(p_{\text{left},i}^{\text{current}} - p_{\text{left},i}^{\text{target}})^2$$

$$\text{MSE}_{\text{right}} = \frac{1}{3}\sum_{i=0}^{2}(p_{\text{right},i}^{\text{current}} - p_{\text{right},i}^{\text{target}})^2$$

$$\text{MSE}_{\text{box}} = -\frac{1}{3}\sum_{i=0}^{2}(p_{\text{box},i}^{\text{current}} - p_{\text{box},i}^{\text{target}})^2$$

$$\text{bonus} = \begin{cases} 
+3.0 & \text{if } z_{\text{box}}^{\text{current}} > z_{\text{box}}^{\text{target}} \text{ (success)} \\
-1.0 & \text{if } z_{\text{box}}^{\text{current}} < 0.20 \text{ (failure)} \\
0 & \text{otherwise}
\end{cases}$$

参数说明：
- $w_{\text{hand}}$ = `hand_reward_weight`（手部位置权重）
- $w_{\text{box}}$ = `box_reward_weight`（箱子位置权重）
- $\alpha$ = `reward_scale` = 10.0（奖励缩放因子）
- $p^{\text{current}}$：当前位置
- $p^{\text{target}}$：目标位置

设计原理：通过设置 $w_{\text{box}} > w_{\text{hand}}$，使得智能体优先关注箱子的移动而非手部精确定位。