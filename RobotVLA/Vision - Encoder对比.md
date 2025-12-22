# MAE
## 关键loss训练信息
### MAE 的训练目标 ：和Clip很像，本意上都是在学习一个latent， 这个latent其实可以被下游任务使用
* ![[Pasted image 20251222194022.png]]
* ![[Pasted image 20251222194850.png]]
MAE 目标是：
> **在严重信息缺失下重建世界**

📌 这意味着：
- latent 偏向 **空间几何**
- 对 identity 不敏感
- 对连续变化友好

# DINOV2
* 