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
	* $$$$