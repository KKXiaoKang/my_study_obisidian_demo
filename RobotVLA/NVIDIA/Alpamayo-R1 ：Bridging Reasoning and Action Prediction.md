![[Alpamayo-R1_Bridging Reasoning and Action Prediction.pdf]]


# KeyNote
* 训练分为4个阶段
	* ![[Pasted image 20260106141534.png]]
* VLM training :  使用370万个视觉问答（VQA）样本在VLM中后训练，让Cosmos-Reason VLM 模型具备一定的物理常识和具体推理能力（在原本训练好的VLM当中进行post-train， 让其具备更多的物理常识和具体推理能力）
	*  Cosmos-Reason is a VLM specifically designed for Physical AI applications, post-trained on 3.7M Visual Question Answering (VQA) samples to develop physical common sense and embodied reasoning capabilities

* VLM training: 使用24700个专注于驾驶场景的精选视频 VQA样本，包含场景描述，驾驶难度注释以及从DeepSeek-R1 中提取的推理轨迹及预测下一步动作
	* The model incorporates 24.7K curated video VQA samples focused on driving scenarios, including scene descriptions, driving difficulty annotations, and reasoning traces distilled from DeepSeek-R1 (DeepSeek-AI, 2025) to predict the next action

* Domain-Specific Adaptations ： 特定领域的小组件升级
	* Vision Encoding ：目标机载部署，所以在vision token中要生成尽可能少的token
		*  Single-Image Tokenization
			* 传统 SigLip2 （2025）
			* ReWiND （2023）
		*  Multi-Camera Tokenization
			* 参考: [Efficient Multi-Camera Tokenization with Triplanes for End-to-End Driving](https://arxiv.org/abs/2506.12251)
		*  Multi-Camera Video Tokenization 多相机视频编码
			*  参考: [Towards Efficient and Effective Multi-Camera Encoding for End-to-End Driving](https://www.arxiv.org/pdf/2512.10947)  
