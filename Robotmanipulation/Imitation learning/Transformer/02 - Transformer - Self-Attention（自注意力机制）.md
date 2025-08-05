### Self-Attention （自注意力机制）
* ![[Pasted image 20250805201048.png]]
* 上图是论文中 Transformer 的内部结构图，左侧为 Encoder block，右侧为 Decoder block。红色圈中的部分为 **Multi-Head Attention**，是由多个 **Self-Attention**组成的，可以看到 Encoder block 包含一个 Multi-Head Attention，而 Decoder block 包含两个 Multi-Head Attention (其中有一个用到 Masked)。Multi-Head Attention 上方还包括一个 Add & Norm 层，Add 表示残差连接 ([Residual Connection](https://zhida.zhihu.com/search?content_id=163422979&content_type=Article&match_order=1&q=Residual+Connection&zhida_source=entity)) 用于防止网络退化，Norm 表示 [Layer Normalization](https://zhida.zhihu.com/search?content_id=163422979&content_type=Article&match_order=1&q=Layer+Normalization&zhida_source=entity)，用于对每一层的激活值进行归一化。
##### self-attention的结构
![[Pasted image 20250805195838.png]]
* 上图是 Self-Attention 的结构，在计算的时候需要用到矩阵**Q(查询),K(键值),V(值)**。在实际中，Self-Attention 接收的是输入(单词的表示向量x组成的矩阵X) 或者上一个 Encoder block 的输出。而**Q,K,V**正是通过 Self-Attention 的输入进行线性变换得到的
##### Q,K,V的计算
* ![[Pasted image 20250805200341.png]]
* ![[Pasted image 20250805200355.png]]
* ![[Pasted image 20250805200401.png]]
* Self-Attention 的输入用矩阵X进行表示，则可以使用线性变阵矩阵**WQ,WK,WV**计算得到**Q,K,V**。计算如下图所示，**注意 X, Q, K, V 的每一行都表示一个单词。**

##### Self-Attention的输出
* 得到矩阵 Q, K, V之后就可以计算出 Self-Attention 的输出了，计算的公式如下：
* $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
	* $d_k$ 是 $Q$ , $K$ 矩阵的列数，即向量维度
###### 公式计算
* 公式中计算矩阵$Q$ 和$K$ 每一行向量的内积，为了防止内积过大，因此除于$d_{k}$的平方根
* **Q**乘以**K**的转置后，得到的矩阵行列数都为 n，n 为句子单词数，这个矩阵可以表示单词之间的 attention 强度。下图为**Q**乘以  ，1234 表示的是句子中的单词
![[Pasted image 20250805211012.png]]
* 得到 之后，使用 Softmax 计算每一个单词对于其他单词的 attention 系数，公式中的 Softmax 是对矩阵的每一行进行 Softmax，即每一行的和都变为 1
* ![[Pasted image 20250805211107.png]]
* 得到 Softmax 矩阵之后可以和**V**相乘，得到最终的输出**Z**。
*  ![[Pasted image 20250805211127.png]]
* 上图中 Softmax 矩阵的第 1 行表示单词 1 与其他所有单词的 attention 系数，最终单词 1 的输出  等于所有单词 i 的值  根据 attention 系数的比例加在一起得到，如下图所示：
* ![[Pasted image 20250805211242.png]]


### 多头注意力机制  Multi-Head Attention
* 在上一步，我们已经知道怎么通过 Self-Attention 计算得到输出矩阵 Z，而 Multi-Head Attention 是由多个 Self-Attention 组合形成的，下图是论文中 Multi-Head Attention 的结构图
* ![[Pasted image 20250805211752.png]]
* 从上图可以看到 Multi-Head Attention 包含多个 Self-Attention 层，首先将输入**X**分别传递到 h 个不同的 Self-Attention 中，计算得到 h 个输出矩阵**Z**。下图是 h=8 时候的情况，此时会得到 8 个输出矩阵**Z**。
* ![[Pasted image 20250805211855.png]]
* 得到 8 个输出矩阵 $Z_1$ 到 $Z_8$  之后，Multi-Head Attention 将它们拼接在一起 **(Concat)**，然后传入一个**Linear**层，得到 Multi-Head Attention 最终的输出**Z**
* ![[Pasted image 20250805211959.png]]
* 可以看到 Multi-Head Attention 输出的矩阵**Z**与其输入的矩阵**X**的维度是一样的。