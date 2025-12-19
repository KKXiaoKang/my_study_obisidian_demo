# ViT - Transformer

- 视觉Transformer编码器工作原理
    
    ![](https://zcnvwfjp427h.feishu.cn/space/api/box/stream/download/asynccode/?code=MzE2YjA0NjU4YjJhYWIzZGI3NzdlYWNlOTg0N2NhYTNfazFobHRWbzIzekpLUE1MSlQ4UWpBeXV0d3dKTGJSZDZfVG9rZW46Qk9semJ3T2RTb2trRE14RlE3YmN0RWJCbnZNXzE3NjYxMzIwMDQ6MTc2NjEzNTYwNF9WNA)
    
    - 1) 图像分割
        
        - 输入图像首先被切分成固定大小的若干小块（patch）。每个小块通过展平变成一个向量，尺寸为 P * P * C （p 是 patch 的边长，CC是通道数）。这些 patch 通常会被展平为一个长度为 p^2 * C 的一维向量
            
    - 2) 线性投影
        
        - 每个展平后的 patch 向量通过一个线性层（线性投影）映射到一个较低维度的空间，通常称为 embedding 维度。这一过程类似于词嵌入（word embeddings）在 NLP 中的使用
            
    - 3) 加入位置编码
        
        - ViT 不像卷积神经网络（CNN）那样有局部感知机制，因此需要添加位置编码（Position Embedding）来保留图像中空间位置信息。每个 patch 向量都会加上一个对应的位置信息向
            
    - 4) Transformer Encoder
        
        - 每个经过位置编码的 patch 向量被输入到 Transformer 的 encoder 中。Transformer 由多层自注意力机制（Self-attention）和前馈神经网络（Feed Forward Network）构成。在这里，每个 patch 可以根据其与其他 patch 的关系进行上下文信息的建模
            
    - 5) 分类 Token
        
        - 类似于 BERT 中的 [CLS] token，ViT 在输入的 patch 序列的开始添加一个额外的 token，通常称为 "classification token"（[CLS] token）。这个 token 的最终表示会用于图像分类任务的预测
            
    - 6) 输出层
        
        - 经过多层 Transformer 编码后，最终得到的 [CLS] token 被送入一个分类头（通常是一个全连接层）进行最终的分类预测
            

  

# OpenX (RT-X)

![](https://zcnvwfjp427h.feishu.cn/space/api/box/stream/download/asynccode/?code=Y2EzZGQ0MWRiMDg1NDJiZTMyZmI0ZDg5N2MyZDAxYTFfNzVac2FCeVJ6Q0pkbjZ2S3UyaTQ3enZtejh6cThlMkVfVG9rZW46UmVncGJrZUV1b1FtUnN4a0tPYmM0VWJYbkJiXzE3NjYxMzIwMDQ6MTc2NjEzNTYwNF9WNA)

- Transformer 架构：
    
    - Action 空间 ： 位置和姿态都使用 relative Delta 增量
        
        - (x, y, z, roll, pitch, yaw, gripper) - 统一为7向量的动作空间（但都是基于增量scale的动作）
            
        - 额外一个维度，用于（terminating episode）中止回合
            
        - 关于向量的动作空间的坐标系是基于机器人坐标系还是基于末端坐标系上没有明确定义
            
    - Obs 空间：
        
        - RGB image - 只能是第一视角画面（不支持腕部相机）- 始终用最新的图像传入，不存在历史history图像
            
        - a task string - 机器人应该执行的任务的任务字符串作为输入
            
            - 比如`` move to top right / put banana on top of the pan` `` - 作为提示词
                
    - 局限性【数据只能用于单臂，但是可以参考一些比较详细的设计】
        
        - RT-X 架构并非不能处理双臂，而是它当前的 Transformer 模型被设定为输出单臂末端的动作（discretized end-effector commands），因而“只能”对应单臂单末端的控制
            
        - **RT-1-X 和 RT-2-X 模型设计**：这两种 Transformer-based 模型的架构，输入为图像 + 语言指令，输出为**经离散编码的末端执行器（end-effector）动作**（即关节末端的动作）
            
        - 虽然 Open X-Embodiment Dataset 包含了来自 22 种“embodiments”（包括双臂机器人和四足机器人）的数据[arXiv](https://arxiv.org/pdf/2310.08864?utm_source=chatgpt.com)[Robotics Transformer X](https://robotics-transformer-x.github.io/?mibextid=Zxz2cZ&utm_source=chatgpt.com)，但 RT-X 的模型结构仍以单一末端执行器为输出目标。这意味着它在训练和推理中是以“一个末端动作”为单位进行的，不具备天然处理双臂协同的能力
            
- 重点🚩： 如何将机器人的连续动作编码为tokens？之后在让model预测tokens，将其转换为机器人的连续动作？
    
    - 1）机器人的连续值（某一个帧截取到的）- 原始值
        
        - Δx = 0.012 m, Δy = -0.008 m, Δz = 0.020 m, Δyaw = 0.10 rad, gripper = 1 (张开)
            
    - 2) 归一化到action[-1, 1] （减去均值/标准差）
        
        - Δx_norm = 0.60， Δy_norm = -0.40， Δz_norm = 1.00， Δyaw_norm = 0.25， gripper_norm = 1.00
            
    - 3) 离散化 (discretization / quantization)（假设每个动作空间-1到1，被 256 个离散 bin（像词典里的 256 个单词））
        
        - Δx_norm = 0.60 → bin 205
            
        - Δy_norm = -0.40 → bin 77
            
        - Δz_norm = 1.00 → bin 255
            
        - Δyaw_norm = 0.25 → bin 160
            
        - gripper_norm = 1.00 → bin 255
            
    - 4) 编码为tokens序列
        
        - 每个 bin ID 就像 NLP 中的一个单词，组合起来就是一句“动作句子”：
            
        - [205] [77] [255] [160] [255]
            
    - 5) 执行时的解码 (de-tokenizer)
        
        - 模型输出 token → 还原为离散 bin → 再映射回连续动作：
            
        
        ```Bash
        [205] [77] [255] [160] [255]
            ↓
        Δx = 0.012 m, Δy = -0.008 m, ...
        ```
        
- Model详细介绍 : 介绍RT-1-X 和 RT-2-X 工作流
    
    - RT-1-X
        
        ![](https://zcnvwfjp427h.feishu.cn/space/api/box/stream/download/asynccode/?code=NjhmODhiNjBmZmQ4NjM2MjVjODg4YzFlZjBmYjQ5NmZfc094b3lQQzdGMDhEOTZyUkRUVlJXOXcyYkNibTByeTlfVG9rZW46QUp2TWJNcTlJb0VuYmZ4bUlvT2NxajdybmJmXzE3NjYxMzIwMDQ6MTc2NjEzNTYwNF9WNA)
        
        - 35M parameter network
            
        - Transformer-based - 简化的Transformer构型
            
        - obs空间：
            
            - 输入15张图像（每张图像都用ImageNet预训练的EfficientNet进行处理)
                
            - 自然语言的描述token，转化为USE embedding
                
        - 视觉和语言通过通过FiLM层组合，产生81个vision-language tokens，这些tokens输入到decoder-only Transformer（这个trans layer用于输出tokenized actions标记化的动作）
            
        
          
        
    - RT-2-X
        
        ![](https://zcnvwfjp427h.feishu.cn/space/api/box/stream/download/asynccode/?code=MmQ0OGRiYzM0ZmJkNTllODQ0ZmRiODRhYTc4NjdjYmZfTGwxWjMzcEVJMXJjZWd6b2pGT3Z1U3JEaHZaQlJMNGhfVG9rZW46SnN1QWJ0MkdYb0JQaHp4a01EeGNZU3g2bmJnXzE3NjYxMzIwMDQ6MTc2NjEzNTYwNF9WNA)
        
        - VLLM-based 视觉多模态感知大模型 - VLA
            
            - 视觉模型
                
                - Vit
                    
            - 文本模型
                
                - UL2
                    
            - 训练集dataset分布
                
                - WebLI
                    
            - 使用网络 Internet-scale vision and language data 的数据进行pre-trained
                
            - RT-2将tokenized action 转换为 text tokens（自动转换）
                
            - 比如（a possible action may be “1 128 91 241 5 101 127”.）
                
        - 得以于可以将action space通过词组离散化到每个不同的tokens
            
            - 任务pre-trained model VLM 都可以经过微调后给机器人控制
                
- 训练损失：标准交叉熵Loss 流程说明【如何将action转换为bin，将动作输出转换为多分类输出问题？】
    
    - 1） 定义如下动作空间离散
        
        - at=(x,y,z,roll,pitch,yaw,gripper)，根据上述255的bin化，将其转换为tokens
            
        - at=[token(x),token(y),token(z),token(roll),token(pitch),token(yaw),token(gripper)]
            
    - 2) Ground Truth 动作（离散化后）
        
        - 假设真实动作是：
            
            - x = -0.23 ⇒ token(x) = 42
                
            - y=0.10 ⇒ token(y)=140
                
            - z=0.50 ⇒ token(z)=200
                
            - Roll = 0.0 ⇒ token(roll)=128
                
            - Pitch = -0.15 ⇒ token(pitch)=100
                
            - Yaw = 0.33 ⇒ token(yaw)=180
                
            - gripper=open ⇒ token(gripper)=0
                
            - 所以groud truth 动作 token序列
                
                - a_t = [42, 140, 200, 128, 100, 180, 0]
                    
    - 3) 模型预测分布（softmax分布预测）
        
        - 模型对于每一个维度都会输出一个softmax概率分布
            
        - e.p.
            
            - 对于x维度（256bins）【控制手臂的x position】
                
                - p(x) = [..., 0.02, 0.85, 0.05, ...]
                    
                - -> 在token42上的概率=0.85
                    
            - 对于y维度【控制手臂的y position】
                
                - p(y)[140] = 0.60
                    
            - 对于z维度【控制手臂的z position】
                
                - p(z)[200] = 0.40
                    
            
                  ...
            
            - Gripper
                
                - p(gripper=open)=0.90, p(close)=0.10
                    
    - 4）🚩：交叉熵损失计算公式 - e.g.演示 【离散多类别预测的最好loss计算方式】
        
        - $$L = -\sum_{d=1}^{7} \log p_d(a_t^{gt}[d])$$
            
            - $$\text{d}$$= 动作维度（x, y, z, roll, pitch, yaw, gripper）
                
            - $$p_d() $$ = 模型对该维度的预测分布
                
            - $${a_t}^{gt}[d]$$ = 真实token
                
        - e.g.
            
            - x : 真实token=42，预测概率 = 0.85 -> loss_x = -log（0.85）= 0.16
                
            - y：真实 token=140，预测概率 = 0.60 → loss_y = -log(0.60) ≈ 0.51
                
            - z：真实 token=200，预测概率 = 0.40 → loss_z = -log(0.40) ≈ 0.92
                
            - roll：真实 token=128，预测概率 = 0.70 → loss_roll = -log(0.70) ≈ 0.36
                
            - pitch：真实 token=100，预测概率 = 0.55 → loss_pitch = -log(0.55) ≈ 0.60
                
            - yaw：真实 token=180，预测概率 = 0.20 → loss_yaw = -log(0.20) ≈ 1.61
                
            - gripper：真实=0 (open)，预测概率=0.90 → loss_gripper = -log(0.90) ≈ 0.11
                
        - 最终计算e.g. - 总loss - 该过程针对one-shot问题进行定义
            
            - $$L = \frac{1}{7} \sum_{d=1}^{7} \text{loss}_d$$
                
            - $$L=\frac{1}{7}（0.16+0.51+0.92+0.36+00.60+1.61+0.11）= 0.61$$
                
    - 解析：
        
        - 模型预测的分布越接近真实 token → loss 越小
            
        - 所以训练目标就是让每个维度的 softmax 更加集中在真实动作 token 上
            
        - 这就等价于在动作空间中做 **多维度的分类问题**，而不是回归
            
    - （后续） - 了解交叉熵和KL散度的区别
        
        - **公式如下**
            
            - 交叉熵CE
                
                - $$H(P, Q) = -\sum_i（p_i \cdot \log{q}_i）$$
                    
            - KL散度
                
                - $$D_{KL}(P || Q) = \sum_i（p_i \cdot \log{p}_i） -\sum_i（p_i \cdot \log{q}_i）$$
                    
            - 真实分布 H(P)
                
                - $$- \sum_i（p_i \cdot \log{p}_i） = H(P)$$ （真实分布的熵，常数）
                    
                    - H(P)与模型参数无关（它只依赖于真实分布），所以在训练中只需要最小化KL和最小化CE是等价的
                        
            - KL散度 = 交叉熵 - 真实分布H（P）
                
                - $$ -\sum_i（p_i \cdot \log{q}_i） = H(P, Q)$$（交叉熵）
                    
                - **KL = CE − H(P)**
                    
        - e.g. 示例
            
            - Background - 假设
                
                - 动作维度：(x,y,z,roll,pitch,yaw,gripper)
                    
                - 模型在**真实 token**上的 _预测概率_ （逐维）：
                    
                    - q=[0.85,0.60,0.40,0.70,0.55,0.20,0.90]
                        
                - 离散桶数 $$K = 256$$
                    
                - soft标签采用label smoothing： $$\varepsilon = 0.1$$, 所以
                    
                    - $$p_\text{true} = 1 - \varepsilon = 0.9$$ 【目标分布】
                        
                    - $$p_\text{other} = \frac{\varepsilon}{K-1} = \frac{0.1}{255} = 0.000392$$ 【目标分布】
                        
                - 我们假设预测分布 $$Q$$在非真类上均匀分配（仅为演示），即
                    
                    - $$q_{other} = \frac{1 - q_{\text{true}}}{255}$$【预测分布】
                        
            - `one-shot标签问题`中 KL(one-shot）= CE(one-shot) 【pi0等统一做法】
                
                - 设真实token是k
                    
                    - 真实分布： $$P = [0, ..., 1_k, ..., 0]$$
                        
                    - 预测分布: $$Q = [q_0, ..., q_k, ...]$$
                        
                - KL定义， 对于one-shot来说只有 $$p_k= 1$$， 其余全为0， 所以变为
                    
                    - $$D_{KL}(P||Q) = \sum_i p_i \log \frac{p_i}{q_i} = \log \frac{1}{q_k} = -\log{q_k}$$
                        
                    - 👉 **这与交叉熵完全相同**
                        
                    
                    ![](https://zcnvwfjp427h.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDliM2M0MGMxNTFhM2Y3OTdmNGU5OWE0MmIyZWE4NThfODJsRDFncUJNa3NSNW1yZlF1ZFJ0ek83U0ZsVHRlVzFfVG9rZW46UEZqUGJCVDBWb2d4NFF4RUtkWWNldkNlbkNiXzE3NjYxMzIwMDQ6MTc2NjEzNTYwNF9WNA)
                    
            - `Soft 标签问题`上：（KL(soft标签)， CE(soft标签)）：该问题的定义不再围绕某个one-shot，而是soft 标签， soft标签上 KL散度和交叉熵有区别
                
                - CE【soft标签】（带 smoothing 的目标 PPP）：
                    
                    - $$CE_\text{soft} = - \sum_{i} p_i \log_{q_i} = -（p_\text{true} \log q_{\text{true}} + （K-1）p_\text{other}\log q_\text{other}）$$
                        
                    - 因为 $$（K-1）p_\text{other} = 0.1$$，所以可写成：
                        
                    - $$CE_\text{soft} = -（0.9 \times \log q_{\text{true}} + 0.1 \times \log q_\text{other}）$$
                        
                - 因为目标分布的熵 $$H(P) $$【常数，只与P有关】
                    
                    - $$H(P) = -\sum p_i \log p_i = -（0.9\log 0.9 + 0.1 \log p_\text{other}）$$
                        
                - KL(soft)
                    
                    - $$KL_\text{soft} = CE_\text{soft} - H(P)$$
                        
                - 公式介绍完毕 - 接下来详细介绍soft标签是如何进行loss计算的【以计算动作空间delta pose当中的x方向作为计算参考】
                    
                    - 接下来填充数据 - 假设目标分布十分确定，就是0.9， H(P)的值固定
                        
                        - $$p_\text{other} = \frac{0.1}{255} = 0.000392$$【目标其他token的分布】
                            
                        - $$p_\text{true}=0.9 $$【目标正确token的分布】
                            
                        - $$\log p_\text{true} = \log 0.9 = -0.1053605 $$
                            
                        - $$\log p_\text{other} ≈-7.84384 $$
                            
                        - $$H(P) = -\sum p_i \log p_i = -（0.9\log 0.9 + 0.1 \log 0.000392）= 0.8792093$$
                            
                    - 然后带入公式计算对比 KL散度loss 以及 交叉熵loss
                        
                        - 对于action当中的计算x维度的loss，后续不管是KL和CE都要把所有的action都求和然后取平均
                            
                            - $$q_\text{true}=0.85 $$ - 这里用最上面的one-shot的数据进行假设，假设现在该动作的预测分布的概率为0.85
                                
                            - $$q_\text{other} = \frac{（1 - 0.85）}{255} ≈0.000588235$$ 【预测其他bin的概率，假设是均匀分布的】
                                
                            - $$\log q_\text{true} = \log 0.85 ≈ -0.1625189$$
                                
                            - $$\log q_\text{other} = \log （0.0005882352941176471）≈ −7.438383530044$$
                                
                            - $$CE_\text{one-shot} = - \log q_\text{true} = - \log 0.85 = 0.16251892949777494$$
                                
                            - $$CE_\text{soft} = -（0.9 \times \log q_{\text{true}} + 0.1 \times \log q_\text{other}） = - （0.9 \times \log 0.85 + 0.1 \times \log 0.000588235）≈ 0.8901053895524282$$
                                
                            - $$KL_{\text{soft}} = CE_\text{soft} - H(P) ≈0.8901053895524282−0.8792093279072909=0.010896061645137345$$
                                
                        - 。。。。此处省略其他维度的action的计算
                            
                
                ####         汇总（平均）
                
                        把 7 维逐项平均，得到：【】
                
                - 平均 $$CE_\text{one-shot} = KL_\text{one-shot} ≈ 0.608421$$
                    
                - 平均 $$CE_\text{soft} ≈ 1.213883$$
                    
                - 平均 $$KL_\text{soft} ≈ 0.334674$$
                    
                
                        （以上均为自然对数下的数值，保留小数用于可复现性。）
                
                ---
                
                ####         重要的直观要点（总结）
                
                - 当目标是 **one-hot** 时，**KL** **与** **CE** **完全相同**，每维损失就是
                    
                - 当目标是 **soft（smoothing）** 时：
                    
                    - $$CE_\text{soft} = - 0.9 \log q_\text{true} - 0.1 \log q_\text{other}$$
                        
                    - $$KL_\text{soft} = CE_\text{soft} - H(P)$$, 因为H(P)是与目标分布有关的参数，所以最小化CE_soft与最小化KL_soft在优化角度上是一致的（梯度相同），但数值上会相差这个常数
                        
                    - soft 目标对模型的“过度自信”惩罚更温和（例如 one-hot 下 yaw 带来的大惩罚在 soft 下被显著削弱）。
                        

  

# RDT-1B

![](https://zcnvwfjp427h.feishu.cn/space/api/box/stream/download/asynccode/?code=NDFjYzA5NzAyMDQ0MDM3ZmNlMGQ5ZjMyNTI3NDA1N2JfSjlSZWd0d1pPSVdOMnpWc2xrVHp5SG44UEdzZkpRTE5fVG9rZW46VENacmJIRFEzb1BxWmN4ZkxueGNReGRqbmFkXzE3NjYxMzIwMDQ6MTc2NjEzNTYwNF9WNA)

### 架构说明

- Diffusion 架构下 - 统一物理可解释动作空间（Physically Interpretable Unified Action Space）
    
    - RDT-1B 架构引入了一个能够统一不同机器人形式（如双臂、单臂、车轮机器人等）的动作表示方式，同时保持每个机器人的物理意义不变
        
    - 索引如下
        
    
    ![](https://zcnvwfjp427h.feishu.cn/space/api/box/stream/download/asynccode/?code=OTQzNzIzMzI3YjgyYTVhOWY1MzQ5OThjYWQwNTJiMTRfRjBxR0pPcWRkZ0JJYmVGWm84VzYzemNuaEJ3MkJvNTdfVG9rZW46VUMxb2JSd0tUb1EwUGZ4OTRLRGNoMk5MbnhiXzE3NjYxMzIwMDQ6MTc2NjEzNTYwNF9WNA)
    
- Diffusion + Transformer 的组合架构
    
    - RDT-1B 使用扩散模型（diffusion models）与 Transformer 结合，特别擅长表示并生成具有**多模态性**（multi-modal action distributions，比如双臂协作）的动作轨迹
        
    - Diffusion Modeling. 扩散建模
        
        - 多模态说明，给定 语言指导$$ℓ$$和 观测 $$o_t$$，可能会有许多可能的动作 $$a_t$$来继续执行任务。
            
            - 如果将其建模为确定性映射 $$（ℓ, o_t）-> a_t$$
                

  

  

  

  

  

  

# RLDG - robotic generalist policy distillation via Reinforcement learning

- Target：聚焦于如何训练一个通才策略
    
- pipeline：训练多个rl policy（针对某个任务的）- rl policy的mlp网络一般只能用于过拟合到单分布
    
    - 收集这些policy的experience collection经验收集
        
    - 通过OpenVLA 或者 Octo 这一类可以拟合多分布action的model将经验蒸馏总结
        

  

  

# Π0 / Π0-fast 对比

## Π0 - 动作编码

- 可以参考传统baseline的对每个维度进行dim化，分割为0-256个块，每个块预测概率 + 计算交叉熵CE
    

  

## Π0-fast / 频域压缩 Tokenization

FAST 不再逐维离散化，而是借鉴了 **信号处理里的频域压缩**：

1. **DCT 变换**：
    
    1. 对整个动作序列做 **离散余弦变换 (DCT)**，把它从时间域转换到频域。
        
    2. 在频域里，信号的大部分能量集中在低频部分（轨迹变化平滑），高频部分可以压缩掉。
        
2. **量化 & 编码**：
    
    1. 频域系数经过缩放、舍入 → 得到整数序列。
        
    2. 再用 BPE（Byte Pair Encoding）做压缩 → 生成 token。
        
3. **自回归预测**：
    
    1. 预测的目标不再是「每个关节的0–255分类」，而是 **预测一串频域 token**。
        
    2. 本质上更像 NLP 的 token 预测，而不是逐维 softmax over bins。
        

👉 优点：

- **压缩性强**：一个长动作序列可以压缩成很少 token。
    
- **更平滑**：低频 token 保留了动作的整体趋势，高频只补充细节。
    
- **效率高**：预测序列更短、词表更小，训练加速 5x。