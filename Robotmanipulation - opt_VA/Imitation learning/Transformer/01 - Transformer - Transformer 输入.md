### Transformer 输入
* ![[Pasted image 20250805192602.png]]
*  $X = \text{Word}_{\text{Embedding}} + \text{Position}_{\text{Embedding}}$ 

#### 单词Embedding 
* 作用 ：单词Embedding有很多方式可以获取，可以通过*Word2Vec* 、*Glove* 等算法预训练得到，也可以通过*Transformer*中训练得到

#### 位置Embedding
* 作用 ：通过位置编码，来编码在一段句子中词语出现的顺序及位置【表示单词出现在句子的位置】
	* 因为*Transformer*不采用*RNN*的结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于*NLP*来说十分重要
* 如何获取：
	* *PE* 为position embedding的表示，PE和Word Embedding维度一致，PE也可以通过训练得到，也可以通过某种公式计算得到。在Transformer当中采用后者通过公式计算
		* $PE_{(pos, 2i)} = sin(pos / 10000^{2i/d})$
		* $PE_(pos,2i+1) = cos(pos / 10000^{2i / d})$ 
	* 其中，$pos$表示单词在句子中的位置， $d$ 表示PE的维度（与词Embedding一样）, $2i$ 表示偶数的维度，$2i+1$表示奇数维度（即2i <= d | 2i+1 <= d）。
* 使用该方式计算PE用如下两种好处
	* 使PE可以适应比训练集里面所有句子更长的句子，比如训练集里面最长的句子是20个word，但是突然来了一个长度为21的word，则使用公式计算的方法可以计算出第21位的Embedding
	* 可以让模型容易地计算出相对位置，对于固定长度的间距$k$， $PE(pos+k)$ 可以用 $PE(pos)$计算得到。 因为$Sin(A+B)=Sin(A)Cos(B) + Cos(A)Sin(B)$ $Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)$ 
* 将单词的词 Embedding 和位置 Embedding 相加，就可以得到单词的表示向量 **x**，**x** 就是 Transformer 的输入 
