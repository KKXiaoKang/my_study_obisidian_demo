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