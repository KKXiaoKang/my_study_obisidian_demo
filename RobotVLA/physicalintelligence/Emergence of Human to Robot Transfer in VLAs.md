![[Emergence of Human to Robot Transfer in.pdf]]

## 论文核心探究
* VLA如何让human示教的数据能够让机器人也学会？
	*  前提1：机器人数据有覆盖 human 示范的场景
		* 比如：基础动作，比如放鸡蛋到盒子
	*  前提2： 人类video数据包含 semantic 组合
		* 排序鸡蛋、收拾衣柜/香料架

* 📌 **机器人可能见过部分基础动作**  
	 * 但 human 示范构造出了 **机器人策略中没有见过的新的要求组合**（例如颜色排序、多对象逻辑）  
	* 真正考验的是模型 **能否根据 human 教示推断这些组合语义并执行动作**

## 如何证明human_to_robot是可行的？(通过benchmark实验证明)
### 1 - Scene Generalization
* 不同场景结构（例如某个具体房间布置）机器人没见过
### 2 - 