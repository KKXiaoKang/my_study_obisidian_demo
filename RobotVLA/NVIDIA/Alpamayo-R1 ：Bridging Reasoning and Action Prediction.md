![[Alpamayo-R1_Bridging Reasoning and Action Prediction.pdf]]


# KeyNote
* 训练分为4个阶段
	* ![[Pasted image 20260106141534.png]]
* VLM training :  使用370万个视觉问答（VQA）样本在VLM中后训练，让Cosmos-Reason VLM 模型具备一定的物理常识和具体推理能力（在原本训练好的VLM当中进行post-train， 让其具备更多的物理常识和具体推理能力）
	*  Cosmos-Reason is a VLM specifically designed for Physical AI applications, post-trained on 3.7M Visual Question Answering (VQA) samples to develop physical common sense and embodied reasoning capabilities

* VLM training: 使用24700个专注于驾驶场景的精选视频 VQA样本，包含场景描述，驾驶难度注释以及从DeepSeek-R1 中提取的推理轨迹及预测下一步动作
	* The model incorporates 24.7K curated video VQA samples focused on driving scenarios, including scene descriptions, driving difficulty annotations, and reasoning traces distilled from DeepSeek-R1 (DeepSeek-AI, 2025) to predict the next action

### vision encoder 
* Domain-Specific Adaptations ： 特定领域的小组件升级
	* Vision Encoding ：目标机载部署，所以在vision token中要生成尽可能少的token
		*  Single-Image Tokenization
			* 传统 SigLip2 （2025）
			* ReWiND （2023）
		*  Multi-Camera Tokenization
			* 参考: [Efficient Multi-Camera Tokenization with Triplanes for End-to-End Driving](https://arxiv.org/abs/2506.12251)
		*  Multi-Camera Video Tokenization 多相机视频编码
			*  参考: [Towards Efficient and Effective Multi-Camera Encoding for End-to-End Driving](https://www.arxiv.org/pdf/2512.10947)  

### action decoder
* 对于VLA reasoning traces 来说，
	* 哪些 reasoning traces 是低信息密度的 - 黄色标识
	* 哪些 reasoning traces 是和当前自身行为相反或者冲突的 - 红色标识
	* 哪些 reasoning traces 是思考的的不够深刻 - 蓝色标识
	* ![[Pasted image 20260106161603.png]]
* CoC labeling pipeline :
	* ![[Pasted image 20260106161312.png]]
	* 1 ) Clip 剪辑选择：思维链裁剪
		*  针对刚刚 reasoning traces  的一些情况，选择包含明确驾驶决策的剪辑，过滤掉提供有限并且低信息密度 因果信息
	* 2) Keyframe Labeling：关键帧标签
		* 识别每个视频剪辑中的决策时刻，最大限度地减少潜在的因果混乱
	* 3) - 5) Structured CoC Labeling
		* 结构化 CoC 标签，为了构建最终的 CoC 并进一步减轻因果混乱，我们首先注释观察中的关键组件，同时避免引用未来帧中的因果因素，然后标记相应的驾驶决策。然后，我们用自然语言从驱动决策和因果因素中构建推理轨迹。

## Chain of Causation Dataset: Learning Causally Grounded Reasoning VLAs - 从因果链数据集当中构建VLA
### labeling framework - 因果链标注框架
#### 结构化的标注格式 - Structured Chain of Causation
* the driving decision - 驾驶决策
* the causal factors (critical components) - 因果关系组成部分
* the composed CoC trace : 思维链
*  CoT的最终结果必定是结构化的driving decision ： 包含了纵向选择 和 横向选择
##### driving decision - 驾驶决策
* ![[Pasted image 20260106164626.png]]
#####  因果关系组成部分
* ![[Pasted image 20260106165612.png]]

##### 因果链：确定了因果关系和驾驶决策后，就可以结构化思维链输出了
![[Pasted image 20260106165910.png]]


### Hybrid Labeling Procedure - 混合标签注入
#### 只在决策时刻时介入标注CoT（给定一些元动作，当元动作发生变换时该帧作为keyframe进行标注）
* driving decisions (驾驶决策) - 下图中使用绿色标识  
* critical components (因果组成) - 下图中使用蓝色标识  
* CoT标注示例如下
	* ![[Pasted image 20260106170607.png]]
* Meta action 元动作示例：
	* ![[Pasted image 20260106172836.png]]
#### Labeling Procedure
* **Next, we employ state-of-the-art VLMs such as GPT-5 (OpenAI, 2025) to perform offline auto-labeling through a multi-step reasoning process.** 
* 为了减轻因果混乱，在识别关键组件时，系统会提示 VLM 使用 2 秒历史视频。随后的 6 秒未来视频以及自我的轨迹和元动作，然后用于解决多模态问题并确定相应的驾驶决策。在此过程中，模型对已识别的因果因素进行重要性排序，并在最终推理轨迹中仅保留那些直接影响驾驶决策的因素。


## 训练方法
### pre-training VLA - 训练时参考Physical Intelligence的做法，把KV Cache的梯度进行截断，防止梯度传播到VLM当中
* **![[Pasted image 20260106175640.png]]**